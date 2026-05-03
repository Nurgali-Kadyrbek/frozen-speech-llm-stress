"""Stage 0 — Documentation-verification smoke (PRE_IMPLEMENTATION_DESIGN §9).

Verifies the locked stack matches the design:
  - WavLM-Large hidden_states tuple len = 25
  - Whisper-large-v3 encoder hidden_states tuple len = 33
  - Qwen3-1.7B inputs_embeds forward (loss + generate) works
  - Qwen3 embed_tokens.weight.std() ≈ 0.02
  - StressPresso / Stress-17K-raw / Expresso-read / LibriSpeech-clean-100 loadable

Run on GPU 6:
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/stage0_smoke.py
"""
from __future__ import annotations

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
import transformers  # noqa: E402

QWEN3_MODEL = "Qwen/Qwen3-1.7B"  # unified Qwen3 instruct (post arXiv:2505.09388)
WAVLM_MODEL = "microsoft/wavlm-large"
WHISPER_MODEL = "openai/whisper-large-v3"
STD_TOL = (0.012, 0.030)  # design doc says ≈ 0.02; allow generous band


def gpu_summary() -> str:
    if not torch.cuda.is_available():
        return "CPU"
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    return (
        f"{torch.cuda.get_device_name(0)} (visible idx 0)"
        f"  free={free/1e9:.1f}GB / total={total/1e9:.1f}GB"
    )


def smoke_wavlm(device: str, fails: list) -> None:
    banner("WavLM-Large smoke (microsoft/wavlm-large)")
    from transformers import WavLMModel, AutoFeatureExtractor

    t0 = time.time()
    model = (
        WavLMModel.from_pretrained(WAVLM_MODEL, torch_dtype=torch.float32)
        .eval()
        .to(device)
    )
    feat = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL)
    print(f"  loaded in {time.time()-t0:.1f}s; n_params={sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    audio = np.random.randn(64000).astype(np.float32) * 0.02  # 4 s @ 16 kHz, near-silent
    inputs = feat(audio, sampling_rate=16000, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hs = out.hidden_states

    report_check("hidden_states is tuple of length 25", len(hs) == 25,
                 f"got {len(hs)}", fails)
    report_check("hidden_states[0] dtype/shape sane",
                 hs[0].ndim == 3 and hs[0].shape[-1] == 1024,
                 f"shape={tuple(hs[0].shape)}, dtype={hs[0].dtype}", fails)
    report_check("hidden_states[12] (mid) shape (B, T_s, 1024)",
                 hs[12].ndim == 3 and hs[12].shape[-1] == 1024,
                 f"shape={tuple(hs[12].shape)}", fails)
    report_check("hidden_states[24] (last) shape (B, T_s, 1024)",
                 hs[24].ndim == 3 and hs[24].shape[-1] == 1024,
                 f"shape={tuple(hs[24].shape)}", fails)

    print(f"  T_s = {hs[12].shape[1]} frames at ~50 Hz (expected ~199 for 4s)", flush=True)
    print(f"  hidden_states[12] std = {hs[12].std().item():.4f}, mean = {hs[12].mean().item():+.4f}", flush=True)

    del model, feat, out, hs
    if device == "cuda":
        torch.cuda.empty_cache()


def smoke_whisper(device: str, fails: list) -> None:
    banner("Whisper-large-v3 smoke (openai/whisper-large-v3)")
    from transformers import WhisperModel, WhisperProcessor

    t0 = time.time()
    model = (
        WhisperModel.from_pretrained(WHISPER_MODEL, torch_dtype=torch.float32)
        .eval()
        .to(device)
    )
    proc = WhisperProcessor.from_pretrained(WHISPER_MODEL)
    print(f"  loaded in {time.time()-t0:.1f}s; n_params={sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    audio = np.random.randn(64000).astype(np.float32) * 0.02  # 4 s @ 16 kHz; padded to 30 s by processor
    inputs = proc(audio, sampling_rate=16000, return_tensors="pt").to(device)
    feat = inputs["input_features"]
    report_check("WhisperProcessor input_features shape (B, 128, 3000)",
                 tuple(feat.shape) == (1, 128, 3000),
                 f"got {tuple(feat.shape)}", fails)

    with torch.no_grad():
        enc_out = model.encoder(feat, output_hidden_states=True)
    hs = enc_out.hidden_states

    report_check("encoder hidden_states is tuple of length 33", len(hs) == 33,
                 f"got {len(hs)}", fails)
    report_check("encoder hidden_states[22] shape (B, 1500, 1280) (mid-layer)",
                 tuple(hs[22].shape) == (1, 1500, 1280),
                 f"got {tuple(hs[22].shape)}", fails)
    report_check("encoder hidden_states[32] shape (B, 1500, 1280) (final)",
                 tuple(hs[32].shape) == (1, 1500, 1280),
                 f"got {tuple(hs[32].shape)}", fails)

    print(f"  hidden_states[22] std = {hs[22].std().item():.4f}", flush=True)
    print(f"  hidden_states[32] std = {hs[32].std().item():.4f}", flush=True)

    del model, proc, enc_out, hs
    if device == "cuda":
        torch.cuda.empty_cache()


def smoke_qwen3(device: str, fails: list) -> None:
    banner(f"Qwen3-1.7B smoke ({QWEN3_MODEL}, inputs_embeds + generate)")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(QWEN3_MODEL)
    model = (
        AutoModelForCausalLM.from_pretrained(QWEN3_MODEL, torch_dtype=torch.float32)
        .eval()
        .to(device)
    )
    print(f"  loaded in {time.time()-t0:.1f}s; n_params={sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    embed = model.get_input_embeddings()
    embed_std = embed.weight.std().item()
    report_check(f"embed_tokens.weight.std() ≈ 0.02 (in {STD_TOL[0]}..{STD_TOL[1]})",
                 STD_TOL[0] < embed_std < STD_TOL[1],
                 f"got {embed_std:.4f}", fails)
    print(f"  embed_tokens.weight: shape={tuple(embed.weight.shape)}, "
          f"std={embed_std:.4f}, mean={embed.weight.mean().item():+.4f}, "
          f"||row||_2={embed.weight.norm(dim=-1).mean().item():.3f}", flush=True)

    # Build a short ChatML user turn for inputs_embeds smoke
    msgs = [{"role": "user", "content": "What is 2 + 2?"}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print(f"  chat template (first 120 chars): {text[:120]!r}", flush=True)

    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    print(f"  input_ids.shape = {tuple(ids.shape)}", flush=True)

    # forward with inputs_embeds + labels => returns finite loss
    embeds_in = embed(ids)
    labels = ids.clone()
    with torch.no_grad():
        out = model(inputs_embeds=embeds_in,
                    attention_mask=torch.ones_like(ids),
                    labels=labels)
    report_check("Qwen3 forward(inputs_embeds, labels) returns finite loss",
                 torch.isfinite(out.loss).item(),
                 f"loss={out.loss.item():.4f}", fails)

    # generate(inputs_embeds=...) returns NEW tokens only (per design §3.2)
    with torch.no_grad():
        gen = model.generate(inputs_embeds=embeds_in,
                             max_new_tokens=8,
                             do_sample=False,
                             pad_token_id=tok.eos_token_id)
    report_check("generate(inputs_embeds) returns only new tokens (≤ max_new_tokens+1)",
                 gen.shape[1] <= 9,
                 f"shape={tuple(gen.shape)}", fails)
    print(f"  generated tokens: {gen[0].tolist()}", flush=True)
    print(f"  decoded:          {tok.decode(gen[0], skip_special_tokens=False)!r}", flush=True)

    del model, tok, embed, out, gen
    if device == "cuda":
        torch.cuda.empty_cache()


def smoke_datasets(fails: list) -> None:
    banner("Dataset smoke (HF Hub streaming, one record each)")
    from datasets import load_dataset

    targets = [
        ("slprl/StressPresso",       {"streaming": True}),
        ("slprl/Stress-17K-raw",     {"streaming": True}),
        ("ylacombe/expresso",        {"streaming": True}),
        ("openslr/librispeech_asr",  {"streaming": True, "name": "clean", "split": "train.100"}),
    ]
    for hf_id, kwargs in targets:
        t0 = time.time()
        try:
            ds = load_dataset(hf_id, **kwargs)
            # If no split arg, ds is IterableDatasetDict — pick first split.
            if hasattr(ds, "keys"):
                split = next(iter(ds.keys()))
                stream = ds[split]
            else:
                split = kwargs.get("split", "<default>")
                stream = ds
            first = next(iter(stream))
            keys = list(first.keys())[:8]
            report_check(f"{hf_id} loadable", True,
                         f"split={split}, keys={keys}, dt={time.time()-t0:.1f}s", fails)
        except Exception as exc:
            report_check(f"{hf_id} loadable", False,
                         f"{type(exc).__name__}: {exc}", fails)


def main() -> int:
    fails: list = []
    print(
        f"transformers=={transformers.__version__}, "
        f"torch=={torch.__version__}, "
        f"CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES','?')}",
        flush=True,
    )
    print(f"GPU: {gpu_summary()}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        report_check("CUDA available", False, "running on CPU; design requires GPU 6", fails)
        return 1

    smoke_wavlm(device, fails)
    smoke_whisper(device, fails)
    smoke_qwen3(device, fails)
    smoke_datasets(fails)

    banner("Stage 0 summary")
    if not fails:
        print("  ALL Stage 0 checks PASS.", flush=True)
        return 0
    print(f"  {len(fails)} Stage 0 check(s) FAILED:", flush=True)
    for f in fails:
        print(f"    - {f}", flush=True)
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
