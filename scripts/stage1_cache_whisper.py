"""Cache Whisper-large-v3 encoder layer-22 + layer-32 mean-pooled hidden states
for all Stress-17K-raw + StressPresso audios.

Per design §2.2: Whisper-large-v3 always pads/crops to 30 s → emits 1500
encoder frames at 50 Hz. We mean-pool ONLY over the valid-frame range
`ceil(true_seconds × 50)` so padding does NOT contaminate the cell. Encoder
self-attention DOES mix across positions, so this is an imperfect mitigation
(documented confound, design §1) — but raw Probe-K is the only Stage-1 use of
Whisper, and the alternative (full 1500-frame pool) would inject 30 s of
encoded silence into every short stimulus.

Output: outputs/cache/whisper_pooled.pt with the same row schema as the WavLM
cache; 'pooled' has shape (N, 2, 1280) for [layer 22, layer 32].
"""
from __future__ import annotations

import math
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_env, banner  # noqa: E402

setup_env()

import numpy as np  # noqa: E402
import soxr  # noqa: E402
import torch  # noqa: E402

from src.data.stress_data import (  # noqa: E402
    load_stress17k, load_stresspresso_test,
    Stress17kItem, StressPressoItem,
)

WHISPER_MODEL = "openai/whisper-large-v3"
TARGET_SR = 16000
LAYERS_KEEP = (22, 32)
ENC_FRAME_RATE_HZ = 50  # Whisper post-conv-stem
DTYPE = torch.float32
BATCH_SIZE = 8
CACHE_PATH = ROOT / "outputs" / "cache" / "whisper_pooled.pt"


def to_target_sr(arr, sr: int) -> np.ndarray:
    if arr is None:
        return np.zeros(TARGET_SR, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    if sr == TARGET_SR:
        return arr
    return soxr.resample(arr, sr, TARGET_SR).astype(np.float32)


def collate_records(items_s17: list[Stress17kItem], items_sp: list[StressPressoItem]) -> list[dict]:
    rows = []
    for it in items_s17:
        rows.append({
            "audio_id": it.audio_id, "source": "stress17k",
            "transcription_id": it.transcription_id,
            "interpretation_id": it.interpretation_id,
            "split_origin": it.split_origin,
            "voice_name": it.voice_name, "speaker_id": "",
            "transcript": it.transcription, "words": list(it.words),
            "n_words": it.n_words, "stress_index": it.stress_index,
            "options": (it.options[0], it.options[1]), "label": it.label,
            "audio_array": it.audio_array, "audio_sr": it.audio_sr,
        })
    for it in items_sp:
        words = it.transcription.split()
        rows.append({
            "audio_id": it.audio_path or it.interpretation_id, "source": "stresspresso",
            "transcription_id": it.transcription_id,
            "interpretation_id": it.interpretation_id, "split_origin": "test",
            "voice_name": "", "speaker_id": it.speaker_id,
            "transcript": it.transcription, "words": words,
            "n_words": len(words), "stress_index": it.stress_index,
            "options": (it.options[0], it.options[1]), "label": it.label,
            "audio_array": it.audio_array, "audio_sr": it.audio_sr,
        })
    return rows


def main() -> int:
    print(f"transformers=={__import__('transformers').__version__}, torch=={torch.__version__}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES','?')}", flush=True)
    if not torch.cuda.is_available():
        print("ERROR: no CUDA visible.", flush=True); return 1
    device = "cuda"

    banner("Loading datasets")
    t0 = time.time()
    items_s17 = load_stress17k()
    items_sp  = load_stresspresso_test()
    print(f"  Stress-17K: {len(items_s17)}  StressPresso: {len(items_sp)}  ({time.time()-t0:.1f}s)", flush=True)
    rows = collate_records(items_s17, items_sp)
    print(f"  total audios to encode: {len(rows)}", flush=True)

    banner("Loading Whisper-large-v3 encoder")
    from transformers import WhisperModel, WhisperProcessor
    t0 = time.time()
    proc  = WhisperProcessor.from_pretrained(WHISPER_MODEL)
    model = WhisperModel.from_pretrained(WHISPER_MODEL, torch_dtype=DTYPE).eval().to(device)
    encoder = model.encoder
    n_params = sum(p.numel() for p in encoder.parameters()) / 1e6
    print(f"  loaded in {time.time()-t0:.1f}s, encoder n_params={n_params:.1f}M", flush=True)

    N = len(rows)
    pooled = torch.zeros(N, len(LAYERS_KEEP), 1280, dtype=DTYPE)
    valid_T = torch.zeros(N, dtype=torch.long)

    t_loop = time.time()
    for start in range(0, N, BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        wavs = [to_target_sr(r["audio_array"], r["audio_sr"]) for r in batch]
        feats = proc(wavs, sampling_rate=TARGET_SR, return_tensors="pt")["input_features"].to(device)
        # Compute valid encoder frames per sample BEFORE padding-to-30s.
        # 50 Hz ≈ valid_T = ceil(true_seconds * 50). Cap at 1500.
        valid_frames = []
        for w in wavs:
            t_sec = len(w) / TARGET_SR
            valid_frames.append(min(1500, max(1, math.ceil(t_sec * ENC_FRAME_RATE_HZ))))

        with torch.no_grad():
            out = encoder(feats, output_hidden_states=True)
        hs = out.hidden_states  # tuple of 33; use indices 22 and 32 only.

        for j, vt in enumerate(valid_frames):
            for li_idx, li in enumerate(LAYERS_KEEP):
                pooled[start + j, li_idx] = hs[li][j, :vt, :].mean(dim=0).cpu()
            valid_T[start + j] = vt

        done = min(start + BATCH_SIZE, N)
        if done % 200 == 0 or done == N:
            elapsed = time.time() - t_loop
            rate = done / max(elapsed, 1e-3)
            eta = (N - done) / max(rate, 1e-3)
            print(f"  encoded {done}/{N}  rate={rate:.1f}/s  ETA={eta:.0f}s", flush=True)

    print(f"  total forward time: {time.time()-t_loop:.1f}s", flush=True)

    banner("Saving cache")
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "audio_ids":         [r["audio_id"]         for r in rows],
        "sources":           [r["source"]           for r in rows],
        "transcription_ids": [r["transcription_id"] for r in rows],
        "interpretation_ids":[r["interpretation_id"]for r in rows],
        "split_origin":      [r["split_origin"]     for r in rows],
        "voice_names":       [r["voice_name"]       for r in rows],
        "speaker_ids":       [r["speaker_id"]       for r in rows],
        "transcripts":       [r["transcript"]       for r in rows],
        "words":             [r["words"]            for r in rows],
        "n_words":           torch.tensor([r["n_words"]      for r in rows], dtype=torch.long),
        "stress_index":      torch.tensor([r["stress_index"] for r in rows], dtype=torch.long),
        "options":           [r["options"]          for r in rows],
        "label":             torch.tensor([r["label"]        for r in rows], dtype=torch.long),
        "pooled":            pooled,
        "layer_indices":     list(LAYERS_KEEP),
        "valid_T_s":         valid_T,
        "model_id":          WHISPER_MODEL,
    }
    torch.save(payload, CACHE_PATH)
    print(f"  saved → {CACHE_PATH}  ({CACHE_PATH.stat().st_size/1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
