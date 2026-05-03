"""Stage 2 training data pool: 60 % Stress-17K-raw probe-train +
20 % LibriSpeech-clean-100 (sampled) + 20 % Expresso-read (sampled).

The pool is deterministic given a seed for the per-step composition so the
five training seeds are comparable. Each sample is a SimpleSample with the
fields the Stage 2 training loop needs:

    audio:      np.ndarray (float32 16 kHz, mono)
    transcript: str
    response:   str
    question:   str          (per-source instruction text)
    source:     'stress17k' | 'librispeech' | 'expresso'
    meta:       dict         (transcription_id / audio_id / etc.)
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from src.data.stress_data import (
    Stress17kItem, load_stress17k, partition_transcript_ids,
)
from src.data.augment import augment_one, AugmentConfig, SR


# ---------- One-canonical-row dataclass ---------- #

@dataclass
class SimpleSample:
    audio:      np.ndarray
    transcript: str
    response:   str
    question:   str
    source:     str
    meta:       dict


STRESS_QUESTION = "Based on the speaker's emphasis, what is the underlying meaning?"
ASR_QUESTION    = "Transcribe what the speaker says."


# ---------- Loaders for each source ---------- #

def load_stress17k_train(stage1_train_ids: set[str]) -> list[SimpleSample]:
    """Load Stress-17K-raw rows whose transcription_id is in the Stage-1
    probe-TRAIN partition. (Stage-1 probe-eval transcripts are held out.)"""
    items = load_stress17k()
    out: list[SimpleSample] = []
    for it in items:
        if it.transcription_id not in stage1_train_ids:
            continue
        # `description` is the canonical correct interpretation
        # (the response text the BLSP teacher should match).
        out.append(SimpleSample(
            audio=np.asarray(it.audio_array, dtype=np.float32),
            transcript=it.transcription,
            response=_get_description_from_options(it),
            question=STRESS_QUESTION,
            source="stress17k",
            meta={
                "transcription_id":  it.transcription_id,
                "interpretation_id": it.interpretation_id,
                "audio_id":          it.audio_id,
                "voice_name":        it.voice_name,
                "stress_index":      it.stress_index,
            },
        ))
    return out


def _get_description_from_options(it: Stress17kItem) -> str:
    """Return the option text matching `it.label`."""
    return it.options[it.label]


def _save_audio_pool_cache(rows: list[SimpleSample], cache_path) -> None:
    import pickle
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for r in rows:
        payload.append({
            "audio": r.audio,           # np.ndarray float32 16 kHz
            "transcript": r.transcript,
            "response": r.response,
            "question": r.question,
            "source": r.source,
            "meta": r.meta,
        })
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_audio_pool_cache(cache_path) -> list[SimpleSample]:
    import pickle
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)
    return [SimpleSample(**rec) for rec in payload]


def load_librispeech_subset(n: int, seed: int = 0) -> list[SimpleSample]:
    """Stream the first `n` valid LibriSpeech-clean-100 rows.

    Pool is seed-AGNOSTIC by default (uses seed=0 for the streaming traversal)
    so that every training seed sees the same pool. The MixedSampler is what
    introduces seed-specific batch composition. Cache to disk after first load.
    """
    from pathlib import Path
    cache_path = Path("/raid/nurgaly/datasets/Beyond_Transcript_Alignment/cache") / f"librispeech_pool_n{n}.pkl"
    if cache_path.exists():
        return _load_audio_pool_cache(cache_path)

    from datasets import load_dataset
    ds = load_dataset("openslr/librispeech_asr", "clean",
                      split="train.100", streaming=True)
    rng = random.Random(0)   # fixed seed across all training seeds
    accept_p = max(0.05, min(1.0, (n / 28539) * 1.5))
    out: list[SimpleSample] = []
    import soxr
    for rec in ds:
        if len(out) >= n:
            break
        if rng.random() > accept_p:
            continue
        audio = rec["audio"]
        if audio is None:
            continue
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr  = int(audio.get("sampling_rate", SR))
        if sr != SR:
            arr = soxr.resample(arr, sr, SR).astype(np.float32)
        text = rec["text"]
        out.append(SimpleSample(
            audio=arr,
            transcript=text,
            response=text,
            question=ASR_QUESTION,
            source="librispeech",
            meta={
                "id":         rec["id"],
                "speaker_id": rec["speaker_id"],
                "chapter_id": rec["chapter_id"],
            },
        ))
    _save_audio_pool_cache(out, cache_path)
    return out


def load_expresso_subset(n: int, seed: int = 0) -> list[SimpleSample]:
    """Stream the Expresso 'read' subset until we have `n` rows.

    Same seed-agnostic policy as `load_librispeech_subset`.
    """
    from pathlib import Path
    cache_path = Path("/raid/nurgaly/datasets/Beyond_Transcript_Alignment/cache") / f"expresso_pool_n{n}.pkl"
    if cache_path.exists():
        return _load_audio_pool_cache(cache_path)

    from datasets import load_dataset
    ds = load_dataset("ylacombe/expresso", split="train", streaming=True)
    rng = random.Random(1)   # fixed seed
    out: list[SimpleSample] = []
    import soxr
    for rec in ds:
        if len(out) >= n:
            break
        text = rec.get("text") or ""
        if not text.strip():
            continue
        if rng.random() < 0.5:
            continue
        audio = rec["audio"]
        if audio is None:
            continue
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr  = int(audio.get("sampling_rate", SR))
        if sr != SR:
            arr = soxr.resample(arr, sr, SR).astype(np.float32)
        out.append(SimpleSample(
            audio=arr,
            transcript=text,
            response=text,
            question=ASR_QUESTION,
            source="expresso",
            meta={
                "id":          rec["id"],
                "speaker_id":  rec["speaker_id"],
                "style":       rec.get("style", ""),
            },
        ))
    _save_audio_pool_cache(out, cache_path)
    return out


# ---------- Mixed sampler (60 / 20 / 20 per step) ---------- #

class MixedSampler:
    """Yields per-step batches with prescribed source proportions.

    The per-step batch composition is: floor(B*0.60) from Stress-17K, the
    rest split 50/50 between LibriSpeech and Expresso (so for B=32 we get
    19 / 6 / 7 → rebalanced to 19 / 7 / 6 in code below). The remainder
    is rounded toward Stress-17K when ambiguous.
    """

    def __init__(
        self,
        stress: list[SimpleSample],
        libri:  list[SimpleSample],
        expr:   list[SimpleSample],
        *,
        seed: int,
        stress_frac: float = 0.60,
        libri_frac:  float = 0.20,
        expr_frac:   float = 0.20,
    ):
        self.stress, self.libri, self.expr = stress, libri, expr
        self.rng = random.Random(seed)
        self.stress_frac = stress_frac
        self.libri_frac  = libri_frac
        self.expr_frac   = expr_frac

    def _draw(self, pool: list[SimpleSample], n: int) -> list[SimpleSample]:
        if not pool:
            return []
        return [pool[self.rng.randrange(len(pool))] for _ in range(n)]

    def batch(self, batch_size: int) -> list[SimpleSample]:
        n_stress = max(1, round(batch_size * self.stress_frac))
        n_libri  = max(1, round(batch_size * self.libri_frac))
        n_expr   = max(0, batch_size - n_stress - n_libri)
        rows = (
            self._draw(self.stress, n_stress)
            + self._draw(self.libri, n_libri)
            + self._draw(self.expr,  n_expr)
        )
        # Local shuffle within the batch so source order isn't a confound.
        self.rng.shuffle(rows)
        return rows


def with_augmentation(
    rows: list[SimpleSample],
    *,
    cfg: AugmentConfig | None = None,
    seed: int = 0,
) -> list[SimpleSample]:
    """Return new rows with `audio` replaced by an augmented version."""
    rng = random.Random(seed)
    out = []
    for r in rows:
        out.append(SimpleSample(
            audio=augment_one(r.audio, cfg=cfg, rng=rng),
            transcript=r.transcript,
            response=r.response,
            question=r.question,
            source=r.source,
            meta=r.meta,
        ))
    return out
