"""Dataset adapters for StressPresso and Stress-17K-raw.

Goal: produce typed Python records that downstream code (Stage 1a oracle,
Stage 1b/1c Probe-K) can use without re-deriving fields from raw HF dicts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence
import hashlib


# ---------- StressPresso (202 test items, real recordings, 48 kHz) ---------- #

@dataclass(frozen=True)
class StressPressoItem:
    transcription_id: str
    interpretation_id: str
    transcription: str         # raw transcript
    transcript_marked: str     # transcript with [[stressed_word]] markup
    stressed_word: str
    stress_index: int          # word position (0-indexed) in transcription.split()
    options: tuple[str, str]   # (option_A, option_B)
    label: int                 # 0 or 1, index into options
    audio_array: object | None
    audio_sr: int
    audio_path: str            # original filename (no path on /raid)
    speaker_id: str


def _wrap_word(transcription: str, position: int, *, lhs: str = "[[", rhs: str = "]]") -> str:
    words = transcription.split()
    if position >= len(words):
        return transcription
    out = list(words)
    out[position] = f"{lhs}{out[position]}{rhs}"
    return " ".join(out)


def parse_stresspresso_record(rec: dict) -> StressPressoItem:
    sp = rec["stress_pattern"]
    indices = sp["indices"]
    if len(indices) != 1:
        # Multi-stress items: keep first; downstream caller may filter.
        pass
    stress_index = int(indices[0])
    stressed_word = sp["words"][0] if sp["words"] else ""
    transcript_marked = _wrap_word(rec["transcription"], stress_index)
    audio = rec.get("audio") or {}
    speaker = (rec.get("metadata") or {}).get("speaker_id", "")
    return StressPressoItem(
        transcription_id=str(rec["transcription_id"]),
        interpretation_id=str(rec["interpretation_id"]),
        transcription=rec["transcription"],
        transcript_marked=transcript_marked,
        stressed_word=stressed_word,
        stress_index=stress_index,
        options=(rec["possible_answers"][0], rec["possible_answers"][1]),
        label=int(rec["label"]),
        audio_array=audio.get("array"),
        audio_sr=int(audio.get("sampling_rate", 48000)),
        audio_path=str(audio.get("path", "")),
        speaker_id=str(speaker),
    )


def load_stresspresso_test() -> list[StressPressoItem]:
    """Load all 202 StressPresso test items (non-streaming so audio arrays decode)."""
    from datasets import load_dataset
    ds = load_dataset("slprl/StressPresso", split="test")
    return [parse_stresspresso_record(r) for r in ds]


# ---------- Stress-17K-raw (4400 train_full + 1311 train_fine, TTS, 16 kHz) -- #

@dataclass(frozen=True)
class Stress17kItem:
    transcription_id: str
    interpretation_id: str
    audio_id: str
    transcription: str
    words: tuple[str, ...]      # whistress_transcription, the canonical word list
    stress_index: int           # gt_stress_indices[0]
    n_words: int
    options: tuple[str, str]
    label: int
    audio_array: object | None
    audio_sr: int
    voice_name: str             # nova / echo / ... — for TTS-confound tracking
    split_origin: str           # 'train_full' | 'train_fine'


def parse_stress17k_record(rec: dict, split_origin: str) -> Stress17kItem | None:
    # Some records may have multi-stress; design assumes single. Filter those.
    gt = rec.get("gt_stress_indices") or []
    if len(gt) != 1:
        return None
    words = tuple(rec.get("whistress_transcription") or [])
    stress_index = int(gt[0])
    if stress_index >= len(words):
        return None
    audio = rec.get("audio") or {}
    md = (rec.get("metadata") or {}).get("tts_metadata") or {}
    return Stress17kItem(
        transcription_id=str(rec["transcription_id"]),
        interpretation_id=str(rec["interpretation_id"]),
        audio_id=str(rec.get("audio_id", "")),
        transcription=rec["transcription"],
        words=words,
        stress_index=stress_index,
        n_words=len(words),
        options=(rec["possible_answers"][0], rec["possible_answers"][1]),
        label=int(rec["label"]),
        audio_array=audio.get("array"),
        audio_sr=int(audio.get("sampling_rate", 16000)),
        voice_name=str(md.get("voice_name", "")),
        split_origin=split_origin,
    )


def load_stress17k(splits: Sequence[str] = ("train_full", "train_fine")) -> list[Stress17kItem]:
    from datasets import load_dataset
    items: list[Stress17kItem] = []
    for s in splits:
        ds = load_dataset("slprl/Stress-17K-raw", split=s)
        for r in ds:
            it = parse_stress17k_record(r, s)
            if it is not None:
                items.append(it)
    return items


# ---------- Transcript-id partitioning (deterministic, hash-based) ---------- #

def partition_transcript_ids(
    items: Iterable[Stress17kItem],
    eval_frac: float = 0.20,
    seed: str = "BTA-2026-05-02",
) -> tuple[set[str], set[str]]:
    """Deterministic 80/20 split of unique transcription_ids.

    Using a stable hash means the partition is reproducible across re-runs
    without requiring a saved random seed alongside the cache.
    """
    ids = sorted({it.transcription_id for it in items})
    eval_ids: set[str] = set()
    for tid in ids:
        h = hashlib.sha256(f"{seed}:{tid}".encode()).digest()
        # First 8 bytes as uint64 → uniform [0, 1)
        u = int.from_bytes(h[:8], "big") / (1 << 64)
        if u < eval_frac:
            eval_ids.add(tid)
    train_ids = set(ids) - eval_ids
    return train_ids, eval_ids


def split_items(
    items: Iterable[Stress17kItem],
    train_ids: set[str],
    eval_ids: set[str],
) -> tuple[list[Stress17kItem], list[Stress17kItem]]:
    train, evalu = [], []
    for it in items:
        if it.transcription_id in train_ids:
            train.append(it)
        elif it.transcription_id in eval_ids:
            evalu.append(it)
    return train, evalu
