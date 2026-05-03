"""Stage 2.0 — Adapter spec lock + smoke (norm matching against Qwen3-8B).

Re-measures `embed_tokens.weight.std()` for Qwen/Qwen3-8B AT RUNTIME (the 1.7B
value of 0.0345 does NOT transfer; this is rule R2 from the kickoff). Then:

  1. Initializes the Stage-2 adapter with last_linear_std = std_8B / sqrt(d_llm).
  2. Forwards a small batch of WavLM-L16 hidden states through the adapter.
  3. Computes ||A(H)||_2 / ||embed_row||_2 on every position.
  4. If the median ratio is outside [0.5, 2.0], retunes RMSNorm.scale ← scale
     × (target / median_ratio) so the band closes in one step.
  5. Re-runs the forward, asserts the new median ratio is in [0.5, 2.0].
  6. Saves the calibrated init values to outputs/stage2/adapter_init.json so
     scripts/stage2_train.py reuses the SAME init across all 5 seeds.

Run:
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/stage2_adapter_smoke.py
"""
from __future__ import annotations

import json
import math
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_env, banner, report_check  # noqa: E402

setup_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.models.adapter import AdapterConfig, BLSPAdapter  # noqa: E402

QWEN3_MODEL = "Qwen/Qwen3-8B"
WAVLM_MODEL = "microsoft/wavlm-large"
TARGET_BAND = (0.5, 2.0)

OUT_PATH = ROOT / "outputs" / "stage2" / "adapter_init.json"


def main() -> int:
    fails: list = []
    print(f"transformers=={__import__('transformers').__version__}, torch=={torch.__version__}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES','?')}", flush=True)
    if not torch.cuda.is_available():
        report_check("CUDA available", False, "GPU 6 required", fails); return 1
    device = "cuda"

    # ---- Re-measure std_8B (rule R2) ---- #
    banner(f"Re-measuring embed_tokens.weight.std() on {QWEN3_MODEL}")
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(QWEN3_MODEL, torch_dtype=torch.float32).eval().to(device)
    embed = model.get_input_embeddings()
    std_8b = embed.weight.float().std().item()
    embed_row_norm_mean = embed.weight.float().norm(dim=-1).mean().item()
    d_llm = embed.weight.shape[1]
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
    print(f"  d_llm                = {d_llm}", flush=True)
    print(f"  embed_tokens.std()   = {std_8b:.5f}", flush=True)
    print(f"  mean ||embed_row||_2 = {embed_row_norm_mean:.4f}", flush=True)
    print(f"  std × √d_llm         = {std_8b * math.sqrt(d_llm):.4f}  (sanity: ≈ ||row||)", flush=True)

    # We don't need the LM during the smoke — release to free GPU mem for WavLM.
    del model, embed
    torch.cuda.empty_cache()

    # ---- WavLM-L16 hidden states for a small batch ---- #
    banner(f"Forwarding 8 audios through WavLM (layer 16) for the smoke batch")
    from transformers import WavLMModel, AutoFeatureExtractor
    from src.data.stress_data import load_stress17k

    wavlm = WavLMModel.from_pretrained(WAVLM_MODEL, torch_dtype=torch.float32).eval().to(device)
    feat = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL)

    items = load_stress17k(splits=("train_full",))[:8]
    wavs = [it.audio_array for it in items]
    proc = feat(wavs, sampling_rate=16000, return_tensors="pt", padding=True,
                return_attention_mask=True)
    iv = proc["input_values"].to(device)
    am = proc["attention_mask"].to(device)
    with torch.no_grad():
        out = wavlm(input_values=iv, attention_mask=am, output_hidden_states=True)
    H_l16 = out.hidden_states[16].clone()    # (B, T_s, 1024) on GPU
    print(f"  WavLM-L16 H shape = {tuple(H_l16.shape)}", flush=True)
    sample_lengths = am.sum(dim=1)
    valid_T_s = wavlm._get_feat_extract_output_lengths(sample_lengths).long()
    print(f"  valid_T_s         = {valid_T_s.tolist()}", flush=True)

    # release WavLM
    del wavlm, feat, proc, out
    torch.cuda.empty_cache()

    # ---- Build adapter with R2 init (last_linear_std = std_8B / √d_llm) ---- #
    banner("Building adapter with last_linear_std = std_8B / √d_llm")
    last_linear_std = std_8b / math.sqrt(d_llm)
    cfg = AdapterConfig(
        d_enc=1024, d_llm=d_llm, conv_kernel=4, conv_stride=4,
        mlp_hidden_mult=2,
        last_linear_std=last_linear_std,
        rmsnorm_init_scale=1.0,
        modality_token_std=std_8b,
    )
    adapter = BLSPAdapter(cfg).to(device).to(torch.float32)
    n_trainable = adapter.n_trainable_params()
    print(f"  d_llm={d_llm}, last_linear_std={last_linear_std:.6f}", flush=True)
    print(f"  trainable params = {n_trainable/1e6:.2f}M (target ≤ 60M)", flush=True)
    report_check("trainable params ≤ 60M", n_trainable <= 60_000_000,
                 f"got {n_trainable/1e6:.2f}M", fails)

    # ---- Smoke #1: forward, check ||A(H)|| / ||embed_row|| ---- #
    banner("Smoke #1 — pre-retune ratio band")
    with torch.no_grad():
        K, valid_T_k = adapter(H_l16, valid_T_s=valid_T_s)
    print(f"  K shape = {tuple(K.shape)}, valid_T_k = {valid_T_k.tolist()}", flush=True)

    # Per-position ||K_i||_2 over the valid range; ratio against ||embed_row||_2 mean.
    ratios = []
    for b in range(K.shape[0]):
        Tk = int(valid_T_k[b].item())
        norms = K[b, :Tk].float().norm(dim=-1)   # (Tk,)
        ratios.append((norms / embed_row_norm_mean).cpu().numpy())
    flat = np.concatenate(ratios)
    median_ratio = float(np.median(flat))
    p10 = float(np.quantile(flat, 0.1))
    p90 = float(np.quantile(flat, 0.9))
    print(f"  per-position ratio: median={median_ratio:.3f}, p10={p10:.3f}, p90={p90:.3f}", flush=True)

    # ---- Retune RMSNorm.scale if outside band ---- #
    if not (TARGET_BAND[0] <= median_ratio <= TARGET_BAND[1]):
        banner(f"Smoke #1 OUT of band {TARGET_BAND}; retuning RMSNorm scale")
        target = 1.0   # bring median ratio to 1.0 → middle of band
        new_scale = (target / median_ratio) * cfg.rmsnorm_init_scale
        print(f"  retune: scale {cfg.rmsnorm_init_scale} → {new_scale:.5f}", flush=True)
        adapter.set_rmsnorm_scale(new_scale)
        cfg = AdapterConfig(**{**cfg.__dict__, "rmsnorm_init_scale": new_scale})

        # Re-smoke
        with torch.no_grad():
            K2, _ = adapter(H_l16, valid_T_s=valid_T_s)
        ratios2 = []
        for b in range(K2.shape[0]):
            Tk = int(valid_T_k[b].item())
            norms = K2[b, :Tk].float().norm(dim=-1)
            ratios2.append((norms / embed_row_norm_mean).cpu().numpy())
        flat2 = np.concatenate(ratios2)
        median_ratio = float(np.median(flat2))
        p10 = float(np.quantile(flat2, 0.1))
        p90 = float(np.quantile(flat2, 0.9))
        print(f"  post-retune ratio: median={median_ratio:.3f}, p10={p10:.3f}, p90={p90:.3f}", flush=True)
    else:
        print(f"  ratio already in band {TARGET_BAND}; no retune.", flush=True)

    in_band = TARGET_BAND[0] <= median_ratio <= TARGET_BAND[1]
    report_check(
        f"||A(H)|| / ||embed_row|| median ∈ {TARGET_BAND}",
        in_band, f"median={median_ratio:.3f}", fails,
    )

    # ---- Save calibrated init values (reused by every seed in stage2_train.py) ---- #
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "qwen3_8b": {
            "model_id": QWEN3_MODEL,
            "d_llm": d_llm,
            "embed_tokens_std": std_8b,
            "embed_row_norm_mean": embed_row_norm_mean,
        },
        "adapter_init": {
            "d_enc": cfg.d_enc,
            "d_llm": cfg.d_llm,
            "conv_kernel": cfg.conv_kernel,
            "conv_stride": cfg.conv_stride,
            "mlp_hidden_mult": cfg.mlp_hidden_mult,
            "last_linear_std": cfg.last_linear_std,
            "rmsnorm_init_scale": cfg.rmsnorm_init_scale,
            "modality_token_std": cfg.modality_token_std,
        },
        "smoke": {
            "median_ratio": median_ratio,
            "p10_ratio": p10,
            "p90_ratio": p90,
            "target_band": TARGET_BAND,
            "n_trainable_params": n_trainable,
        },
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\n  saved calibrated init → {OUT_PATH}", flush=True)

    if fails:
        print(f"\n  {len(fails)} smoke check(s) FAILED:")
        for f in fails:
            print(f"    - {f}")
        return 1
    print("\n  Stage 2.0 adapter smoke PASS.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
