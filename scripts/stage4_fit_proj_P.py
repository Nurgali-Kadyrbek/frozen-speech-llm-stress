"""Stage 4 / Control A prerequisite — fit P : R^4096 → R^1024.

Frozen linear projection mapping pool(K_T) → pool(H), fit by minimizing
MSE on a 500-sample batch from Stress-17K probe-train (per kickoff §4.0
moment-matched projection spec).

Saves outputs/stage4/proj_P.pt with the fitted weight + bias.

Run:
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/stage4_fit_proj_P.py
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_env, banner  # noqa: E402

setup_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

QWEN3_MODEL = "Qwen/Qwen3-8B"
WAVLM_MODEL = "microsoft/wavlm-large"
SR = 16000

CACHE = Path("/raid/nurgaly/datasets/Beyond_Transcript_Alignment/cache")
STRESS_POOL_PATH = CACHE / "stress17k_probe_train_pool.pkl"
OUT_P_PATH       = ROOT / "outputs" / "stage4" / "proj_P.pt"


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: no CUDA visible.", flush=True); return 1
    device = "cuda"
    torch.manual_seed(0)
    np.random.seed(0)

    OUT_P_PATH.parent.mkdir(parents=True, exist_ok=True)

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

    banner("Loading 500-sample batch from Stress-17K probe-train")
    with open(STRESS_POOL_PATH, "rb") as f:
        stress_pool = pickle.load(f)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(stress_pool))[:500]
    samples = [stress_pool[i] for i in idx]
    print(f"  {len(samples)} samples", flush=True)

    banner("Computing pool(K_T) and pool(H) per sample")
    X_kt   = []   # (N, 4096) — per-sample pooled K_T (text)
    Y_h    = []   # (N, 1024) — per-sample pooled WavLM-L16 H (audio)
    t0 = time.time()
    max_audio_seconds = 8.0
    max_n = int(max_audio_seconds * SR)
    for i, r in enumerate(samples):
        # K_T: pool token embeddings of transcript.
        ids = tok(r["transcription"], return_tensors="pt",
                  add_special_tokens=False).input_ids[0].to(dtype=torch.long, device=device)
        with torch.no_grad():
            kt = embed_layer(ids).float().mean(dim=0).cpu().numpy()
        X_kt.append(kt)
        # H: pool WavLM-L16 over time.
        audio = np.asarray(r["audio_array"], dtype=np.float32)
        if audio.shape[0] > max_n:
            audio = audio[:max_n]
        with torch.no_grad():
            proc = feat_ex([audio], sampling_rate=SR, return_tensors="pt",
                           padding=True, return_attention_mask=True)
            iv = proc["input_values"].to(device).to(torch.bfloat16)
            am = proc["attention_mask"].to(device)
            wav_out = wavlm(input_values=iv, attention_mask=am, output_hidden_states=True)
            sample_lengths = am.sum(dim=1)
            valid_T_s = wavlm._get_feat_extract_output_lengths(sample_lengths).long()
            T = int(valid_T_s[0].item())
            h = wav_out.hidden_states[16][0, :T, :].float().mean(dim=0).cpu().numpy()
        Y_h.append(h)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(samples)} done in {time.time()-t0:.1f}s", flush=True)
    X_kt = np.stack(X_kt).astype(np.float32)
    Y_h  = np.stack(Y_h).astype(np.float32)
    print(f"  X_kt {X_kt.shape}, Y_h {Y_h.shape}", flush=True)

    banner("Fitting linear P : 4096 → 1024 via least-squares")
    # Add bias by appending a 1.
    Xb = np.hstack([X_kt, np.ones((X_kt.shape[0], 1), dtype=np.float32)])
    # Solve W (4097, 1024) such that Xb @ W ≈ Y_h
    W_full, _, rank, _ = np.linalg.lstsq(Xb.astype(np.float64), Y_h.astype(np.float64), rcond=None)
    W = W_full[:-1, :].astype(np.float32)   # (4096, 1024)
    b = W_full[-1, :].astype(np.float32)    # (1024,)
    print(f"  W shape {W.shape}, b shape {b.shape}, rank {rank}", flush=True)

    # Diagnostic: training MSE + cosine.
    pred = X_kt @ W + b
    mse_train = float(((pred - Y_h) ** 2).mean())
    cos_train = float((pred * Y_h).sum(axis=-1).mean()
                       / (np.linalg.norm(pred, axis=-1)
                          * np.linalg.norm(Y_h, axis=-1) + 1e-9).mean())
    rms_pred = float(np.sqrt((pred ** 2).mean()))
    rms_y    = float(np.sqrt((Y_h ** 2).mean()))
    print(f"  train MSE     = {mse_train:.6f}", flush=True)
    print(f"  train mean cos sim = {cos_train:.4f}", flush=True)
    print(f"  rms(pred)={rms_pred:.4f}  rms(Y)={rms_y:.4f}", flush=True)

    payload = {
        "weight": torch.from_numpy(W),    # (4096, 1024)
        "bias":   torch.from_numpy(b),    # (1024,)
        "n_train": len(samples),
        "train_mse": mse_train,
        "train_cos_sim": cos_train,
        "rms_pred": rms_pred,
        "rms_y":    rms_y,
        "fit_method": "numpy.linalg.lstsq with bias term",
    }
    torch.save(payload, OUT_P_PATH)
    print(f"  saved → {OUT_P_PATH} ({OUT_P_PATH.stat().st_size/1e6:.2f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
