"""Cache WavLM-Large per-layer mean-pooled hidden states for all Stress-17K-raw
+ StressPresso audios. One forward pass per audio; output is a single .pt file
that downstream Probe-K runs read repeatedly.

Per design §2.1: WavLM hidden_states is a 25-tuple; index 0 is post-CNN-stem
(pre-block-1) and is excluded. Indices 1..24 are post-block residuals at d=1024.
We cache layers 1..24 mean-pooled over VALID frames (using the feature
extractor's attention_mask).

Storage layout (single torch.save dict):
  {
    'audio_ids':     list[str]               # primary key
    'sources':       list[str]               # 'stress17k' | 'stresspresso'
    'transcription_ids': list[str]
    'interpretation_ids': list[str]
    'split_origin': list[str]                # 'train_full' | 'train_fine' | 'test'
    'voice_names':  list[str]                # TTS voice for stress17k; '' for sp
    'speaker_ids':  list[str]                # speaker for stresspresso; '' for s17k
    'transcripts':  list[str]                # raw transcription
    'words':        list[list[str]]          # whistress_transcription where avail
    'n_words':      LongTensor[N]
    'stress_index': LongTensor[N]
    'options':      list[tuple[str,str]]
    'label':        LongTensor[N]
    'pooled':       FloatTensor[N, 24, 1024] # main payload (layers 1..24)
    'valid_T_s':    LongTensor[N]            # number of WavLM frames before mean
  }

Run:
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/stage1_cache_wavlm.py
"""
from __future__ import annotations

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
    load_stress17k, load_stresspresso_test, parse_stress17k_record, parse_stresspresso_record,
    Stress17kItem, StressPressoItem,
)

WAVLM_MODEL = "microsoft/wavlm-large"
TARGET_SR = 16000
BATCH_SIZE = 8
DTYPE = torch.float32
CACHE_PATH = ROOT / "outputs" / "cache" / "wavlm_pooled.pt"


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
            "audio_id": it.audio_id,
            "source": "stress17k",
            "transcription_id": it.transcription_id,
            "interpretation_id": it.interpretation_id,
            "split_origin": it.split_origin,
            "voice_name": it.voice_name,
            "speaker_id": "",
            "transcript": it.transcription,
            "words": list(it.words),
            "n_words": it.n_words,
            "stress_index": it.stress_index,
            "options": (it.options[0], it.options[1]),
            "label": it.label,
            "audio_array": it.audio_array,
            "audio_sr": it.audio_sr,
        })
    for it in items_sp:
        # StressPresso has no whistress_transcription; rebuild a word list from
        # transcription.split() so n_words / stress_index align.
        words = it.transcription.split()
        rows.append({
            "audio_id": it.audio_path or it.interpretation_id,  # fallback
            "source": "stresspresso",
            "transcription_id": it.transcription_id,
            "interpretation_id": it.interpretation_id,
            "split_origin": "test",
            "voice_name": "",
            "speaker_id": it.speaker_id,
            "transcript": it.transcription,
            "words": words,
            "n_words": len(words),
            "stress_index": it.stress_index,
            "options": (it.options[0], it.options[1]),
            "label": it.label,
            "audio_array": it.audio_array,
            "audio_sr": it.audio_sr,
        })
    return rows


def main() -> int:
    print(f"transformers=={__import__('transformers').__version__}, torch=={torch.__version__}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES','?')}", flush=True)
    if not torch.cuda.is_available():
        print("ERROR: no CUDA visible.", flush=True)
        return 1
    device = "cuda"

    banner("Loading datasets (non-streaming, audio decoded)")
    t0 = time.time()
    items_s17 = load_stress17k()
    print(f"  Stress-17K-raw: {len(items_s17)} single-stress items kept "
          f"(filtering multi-stress and out-of-range), {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    items_sp  = load_stresspresso_test()
    print(f"  StressPresso test: {len(items_sp)} items, {time.time()-t0:.1f}s", flush=True)
    rows = collate_records(items_s17, items_sp)
    print(f"  total audios to encode: {len(rows)}", flush=True)

    banner("Loading WavLM-Large")
    from transformers import WavLMModel, AutoFeatureExtractor
    t0 = time.time()
    feat = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL)
    model = WavLMModel.from_pretrained(WAVLM_MODEL, torch_dtype=DTYPE).eval().to(device)
    print(f"  loaded in {time.time()-t0:.1f}s, n_params={sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    banner(f"Forward pass + mean-pool (batch_size={BATCH_SIZE})")
    N = len(rows)
    pooled = torch.zeros(N, 24, 1024, dtype=DTYPE)  # layers 1..24
    valid_T = torch.zeros(N, dtype=torch.long)

    t_loop = time.time()
    for start in range(0, N, BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        wavs = [to_target_sr(r["audio_array"], r["audio_sr"]) for r in batch]
        proc = feat(wavs, sampling_rate=TARGET_SR, return_tensors="pt", padding=True,
                    return_attention_mask=True)
        input_values = proc["input_values"].to(device)
        # WavLM-Large needs an attention mask for padding-aware attention.
        attn = proc["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_values=input_values, attention_mask=attn,
                        output_hidden_states=True)
        hs = out.hidden_states  # tuple of 25; idx 0 is pre-block, skip it.
        # WavLM 50 Hz: number of valid frames per item = floor((n_samples - kernel + 1) / 320 + 1)
        # easier: use the model's `_get_feat_extract_output_lengths` if exposed,
        # else compute via the conv-stem pattern. WavLM has stride=[5,2,2,2,2,2,2],
        # kernel=[10,3,3,3,3,2,2] giving total stride 320 → 50 Hz.
        # Use the attention_mask broadcast: feature extractor returns a sample-level
        # attention mask of length T_audio_samples; we need it converted to T_s.
        # transformers 4.51.3's WavLMModel internally converts the sample-mask to a
        # frame-mask via `_get_feature_vector_attention_mask`. Reuse that.
        sample_lengths = attn.sum(dim=1)  # (B,)
        frame_lengths = model._get_feat_extract_output_lengths(sample_lengths)  # (B,)
        for j, lj in enumerate(frame_lengths.tolist()):
            lj = int(lj)
            if lj <= 0:
                continue
            # mean-pool layers 1..24 over the valid frame range
            for li in range(1, 25):
                pooled[start + j, li - 1] = hs[li][j, :lj, :].mean(dim=0).cpu()
            valid_T[start + j] = lj

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
        "valid_T_s":         valid_T,
        "model_id":          WAVLM_MODEL,
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
