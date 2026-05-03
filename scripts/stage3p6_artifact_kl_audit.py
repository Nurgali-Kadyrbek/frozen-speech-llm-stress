"""Stage 3.6 / B3.a — measure D(K_a, K_a'') distribution on Expresso pairs.

For each adapter checkpoint in {A_BLSP_seed1234 (R0), A_R1_seed1234 (R1
original pilot)}, sample N artifact pairs from cf_pairs_artifact.jsonl,
forward both members through frozen WavLM + adapter + frozen Qwen3-8B with
the transcript as response, compute mean KL(p_a || p_a'') over response
positions, and dump the distribution.

This sets δ_artifact for Stage 3.6 training (kickoff B3.b):
  - if 75th percentile of R0 D < 0.10: set δ = max(75th-pct, 0.02)
  - if 75th percentile is above 0.10 but R1 didn't fire: investigate

Run:
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/stage3p6_artifact_kl_audit.py \
      --n_pairs 100
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_env, banner  # noqa: E402

setup_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.models.adapter import AdapterConfig, BLSPAdapter  # noqa: E402
from src.utils.prompts import build_training_halves, DEFAULT_SYSTEM  # noqa: E402
from src.data.audio_pool import ASR_QUESTION  # noqa: E402
from src.data.cf_pairs import read_artifact_pairs_jsonl  # noqa: E402

QWEN3_MODEL = "Qwen/Qwen3-8B"
WAVLM_MODEL = "microsoft/wavlm-large"
SR = 16000

CACHE = Path("/raid/nurgaly/datasets/Beyond_Transcript_Alignment/cache")
EXPR_POOL_PATH = CACHE / "expresso_pool_n6000.pkl"
CF_ARTIFACT_JSONL = ROOT / "outputs" / "stage3" / "cf_pairs_artifact.jsonl"

ADAPTER_CHECKPOINTS = {
    "R0_BLSP_seed1234":   ROOT / "outputs" / "stage2" / "A_BLSP_seed1234.pt",
    "R1_original_seed1234": ROOT / "outputs" / "stage3" / "A_R1_seed1234.pt",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_pairs", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_audio_seconds", type=float, default=10.0)
    p.add_argument("--out_path", type=str,
                   default=str(ROOT / "outputs" / "stage3" / "artifact_kl_audit.json"))
    return p.parse_args()


@torch.no_grad()
def forward_response_logits(
    *, adapter, wavlm, feat_ex, llm, embed_layer, tok,
    audio: np.ndarray, transcript: str, vis_start_id: int, vis_end_id: int,
    device: str, max_audio_seconds: float, max_response_tokens: int = 64,
):
    """Run K = adapter(WavLM(audio)); forward [E_left, K, E_right, E_resp];
    return log_p_response (T_resp, vocab)."""
    max_n = int(max_audio_seconds * SR)
    if audio.shape[0] > max_n:
        audio = audio[:max_n]
    proc = feat_ex([audio], sampling_rate=SR, return_tensors="pt",
                   padding=True, return_attention_mask=True)
    iv = proc["input_values"].to(device).to(torch.bfloat16)
    am = proc["attention_mask"].to(device)
    out = wavlm(input_values=iv, attention_mask=am, output_hidden_states=True)
    H = out.hidden_states[16].float()
    sample_lengths = am.sum(dim=1)
    valid_T_s = wavlm._get_feat_extract_output_lengths(sample_lengths).long()
    K, valid_T_k = adapter(H, valid_T_s=valid_T_s)
    Tk = int(valid_T_k[0].item())
    K_speech_bf = K[0, :Tk, :].to(torch.bfloat16)

    halves = build_training_halves(
        tok, system=DEFAULT_SYSTEM, question=ASR_QUESTION, response=transcript,
    )
    left_ids = tok(halves.left_text, return_tensors="pt",
                   add_special_tokens=False).input_ids[0].to(dtype=torch.long, device=device)
    right_ids = tok(halves.right_text, return_tensors="pt",
                    add_special_tokens=False).input_ids[0].to(dtype=torch.long, device=device)
    response_ids = tok(transcript, return_tensors="pt",
                       add_special_tokens=False).input_ids[0].to(dtype=torch.long, device=device)
    if response_ids.shape[0] > max_response_tokens:
        response_ids = response_ids[:max_response_tokens]
    T_resp = response_ids.shape[0]

    E_left  = embed_layer(left_ids)
    E_right = embed_layer(right_ids)
    E_resp  = embed_layer(response_ids)
    vs = embed_layer(torch.tensor([vis_start_id], device=device))
    ve = embed_layer(torch.tensor([vis_end_id],   device=device))
    full = torch.cat([E_left, vs, K_speech_bf, ve, E_right, E_resp], dim=0).unsqueeze(0)
    attn = torch.ones(1, full.shape[1], dtype=torch.long, device=device)
    logits = llm(inputs_embeds=full, attention_mask=attn).logits[0]
    T_total = logits.shape[0]
    start = T_total - T_resp - 1
    response_pos_logits = logits[start:start+T_resp].float()
    return F.log_softmax(response_pos_logits, dim=-1), response_ids


def kl_directional(log_p_a: torch.Tensor, log_p_b: torch.Tensor) -> float:
    """KL(p_a || p_b) averaged over response positions."""
    T = min(log_p_a.shape[0], log_p_b.shape[0])
    p_a = log_p_a[:T].exp()
    return float((p_a * (log_p_a[:T] - log_p_b[:T])).sum(dim=-1).mean().item())


def measure_one_adapter(*, label: str, ckpt_path: Path, art_pairs, expr_pool,
                        wavlm, feat_ex, llm, embed_layer, tok,
                        vis_start_id, vis_end_id, n_pairs: int, seed: int,
                        max_audio_seconds: float, device: str) -> dict:
    banner(f"Adapter: {label} ({ckpt_path.name})")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = AdapterConfig(**ckpt["adapter_config"])
    adapter = BLSPAdapter(cfg).to(device).to(torch.float32)
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    adapter.eval()
    print(f"  loaded; trainable params = {adapter.n_trainable_params()/1e6:.2f}M", flush=True)

    rng = random.Random(seed)
    pairs = list(art_pairs)
    rng.shuffle(pairs)
    take = pairs[:n_pairs]

    kls = []
    t0 = time.time()
    for i, p in enumerate(take):
        ra = expr_pool[p.pool_idx_a]
        rb = expr_pool[p.pool_idx_a_prime]
        canonical_transcript = ra.transcript
        with torch.no_grad():
            log_p_a, ids_a = forward_response_logits(
                adapter=adapter, wavlm=wavlm, feat_ex=feat_ex,
                llm=llm, embed_layer=embed_layer, tok=tok,
                audio=ra.audio, transcript=canonical_transcript,
                vis_start_id=vis_start_id, vis_end_id=vis_end_id,
                device=device, max_audio_seconds=max_audio_seconds,
            )
            log_p_b, ids_b = forward_response_logits(
                adapter=adapter, wavlm=wavlm, feat_ex=feat_ex,
                llm=llm, embed_layer=embed_layer, tok=tok,
                audio=rb.audio, transcript=canonical_transcript,
                vis_start_id=vis_start_id, vis_end_id=vis_end_id,
                device=device, max_audio_seconds=max_audio_seconds,
            )
        if ids_a.shape[0] != ids_b.shape[0]:
            continue
        kl = kl_directional(log_p_a, log_p_b)
        kls.append(kl)
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(take)} done in {elapsed:.1f}s "
                  f"(running KL mean={np.mean(kls):.4f})", flush=True)
        del log_p_a, log_p_b
    elapsed = time.time() - t0
    arr = np.asarray(kls, dtype=np.float64)
    if arr.size == 0:
        stats = {k: float("nan") for k in
                 ("mean", "sigma", "p25", "p50", "p75", "p95", "frac_above_0_10")}
    else:
        stats = {
            "mean":  float(arr.mean()),
            "sigma": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "p25":   float(np.percentile(arr, 25)),
            "p50":   float(np.percentile(arr, 50)),
            "p75":   float(np.percentile(arr, 75)),
            "p95":   float(np.percentile(arr, 95)),
            "frac_above_0_10": float((arr > 0.10).mean()),
        }
    print(f"  → measured {len(kls)} pairs in {elapsed:.1f}s", flush=True)
    print(f"  → KL stats: mean={stats['mean']:.4f}  σ={stats['sigma']:.4f}  "
          f"p25={stats['p25']:.4f}  p50={stats['p50']:.4f}  "
          f"p75={stats['p75']:.4f}  p95={stats['p95']:.4f}  "
          f"frac>0.10={stats['frac_above_0_10']:.2%}", flush=True)

    del adapter
    torch.cuda.empty_cache()

    return {
        "label":     label,
        "checkpoint": str(ckpt_path),
        "n_pairs":   len(kls),
        "stats":     stats,
        "kls":       kls,
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("ERROR: no CUDA visible.", flush=True); return 1
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not EXPR_POOL_PATH.exists():
        print(f"ERROR: missing {EXPR_POOL_PATH}", flush=True); return 1
    print(f"loading {EXPR_POOL_PATH.name}…", flush=True)
    with open(EXPR_POOL_PATH, "rb") as f:
        expr_payload = pickle.load(f)
    # `audio_pool.SimpleSample` round-trip: pool stored as dicts, hydrate to objs.
    from src.data.audio_pool import SimpleSample
    expr_pool = [SimpleSample(**rec) for rec in expr_payload]
    print(f"  {len(expr_pool)} expresso rows", flush=True)

    art_pairs = read_artifact_pairs_jsonl(CF_ARTIFACT_JSONL)
    print(f"  {len(art_pairs)} artifact pairs", flush=True)

    banner(f"Loading Qwen3-8B (bf16) and WavLM-Large (bf16)")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QWEN3_MODEL)
    llm = AutoModelForCausalLM.from_pretrained(QWEN3_MODEL, torch_dtype=torch.bfloat16).eval().to(device)
    embed_layer = llm.get_input_embeddings()
    for p in llm.parameters():
        p.requires_grad_(False)

    from transformers import WavLMModel, AutoFeatureExtractor
    feat_ex = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL)
    wavlm = WavLMModel.from_pretrained(WAVLM_MODEL, torch_dtype=torch.bfloat16).eval().to(device)
    for p in wavlm.parameters():
        p.requires_grad_(False)

    vis_start_id = int(tok.convert_tokens_to_ids("<|vision_start|>"))
    vis_end_id   = int(tok.convert_tokens_to_ids("<|vision_end|>"))

    results = {}
    for label, ckpt_path in ADAPTER_CHECKPOINTS.items():
        if not ckpt_path.exists():
            print(f"WARNING: {ckpt_path} missing — skipping", flush=True)
            continue
        results[label] = measure_one_adapter(
            label=label, ckpt_path=ckpt_path,
            art_pairs=art_pairs, expr_pool=expr_pool,
            wavlm=wavlm, feat_ex=feat_ex,
            llm=llm, embed_layer=embed_layer, tok=tok,
            vis_start_id=vis_start_id, vis_end_id=vis_end_id,
            n_pairs=args.n_pairs, seed=args.seed,
            max_audio_seconds=args.max_audio_seconds, device=device,
        )

    # ---- Recommendation per kickoff B3.b ---- #
    banner("Recommendation for Stage 3.6 δ_artifact")
    if "R0_BLSP_seed1234" in results:
        r0_p75 = results["R0_BLSP_seed1234"]["stats"]["p75"]
        if r0_p75 < 0.10:
            new_delta = max(r0_p75, 0.02)
            print(f"  R0 75th-pct = {r0_p75:.4f} < 0.10 → set δ_artifact = max(p75, 0.02) = {new_delta:.4f}", flush=True)
        else:
            new_delta = 0.10
            print(f"  R0 75th-pct = {r0_p75:.4f} ≥ 0.10 → keep δ_artifact = 0.10; "
                  f"investigate why R1 training never crossed it", flush=True)
        results["recommended_delta_artifact"] = new_delta
    else:
        print("  R0 baseline missing — cannot compute recommendation", flush=True)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
