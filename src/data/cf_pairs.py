"""Same-transcript counterfactual pair construction (Stage 3.0).

Two pair families:

  CfPair  — Stress-17K-raw probe-train pairs (a, a') with shared
            transcription_id but different gt_stress_indices. Carries the
            two AFC option strings that L_cf scores against. The 2-AFC
            option set is shared across all interpretation_ids of a
            transcription_id in this dataset, so y_phi_a and y_phi_a'
            equal the two `possible_answers` regardless of which audio
            realization we pick.

  ArtifactPair — Expresso-read pairs (a, a'') with same (speaker_id,
                 transcript) and different `style`. By construction
                 Phi(a) == Phi(a'') (no stress label changes for these
                 pairs); used for L_artifact upper-bounding response
                 divergence.

Both pair indices are JSON-serializable; audio is fetched at training
time via lookup tables built from the existing pool caches.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

from src.data.stress_data import Stress17kItem


# ---------- pair dataclasses ---------- #

@dataclass(frozen=True)
class CfPair:
    """One ordered counterfactual pair from Stress-17K-raw."""
    transcription_id:        str
    transcription:           str         # shared across pair
    audio_id_a:              str
    audio_id_a_prime:        str
    interpretation_id_a:     str
    interpretation_id_a_prime: str
    stress_index_a:          int
    stress_index_a_prime:    int
    n_words:                 int
    options:                 tuple[str, str]   # 2-AFC option set, shared
    label_a:                 int           # 0 or 1 — index into options
    label_a_prime:           int           # 0 or 1
    voice_name_a:            str
    voice_name_a_prime:      str

    @property
    def y_phi_a(self) -> str:
        return self.options[self.label_a]

    @property
    def y_phi_a_prime(self) -> str:
        return self.options[self.label_a_prime]


@dataclass(frozen=True)
class ArtifactPair:
    """One ordered artifact-matched pair from Expresso-read.

    Phi is preserved (same speaker, same text); only `style` differs.
    Used for L_artifact's response-divergence upper bound.
    """
    speaker_id:        str
    transcript:        str
    style_a:           str
    style_a_prime:     str
    pool_idx_a:        int       # index into expresso_pool cache
    pool_idx_a_prime:  int


# ---------- Stress-17K cf pair construction ---------- #

def build_stress_cf_pairs(
    items: Sequence[Stress17kItem],
    *,
    max_pairs_per_transcript: int | None = None,
    seed: int = 0,
) -> list[CfPair]:
    """Enumerate ordered same-transcript pairs differing in stress_index.

    Required invariants enforced:
      - transcription_id matches
      - stress_index_a != stress_index_a_prime
      - both items single-stress (already filtered by parse_stress17k_record)
      - the two `possible_answers` strings differ (skip degenerate pairs)
    """
    rng = random.Random(seed)
    by_tid: dict[str, list[Stress17kItem]] = defaultdict(list)
    for it in items:
        by_tid[it.transcription_id].append(it)

    pairs: list[CfPair] = []
    for tid, group in by_tid.items():
        if len(group) < 2:
            continue
        local: list[CfPair] = []
        for i, a in enumerate(group):
            for j, ap in enumerate(group):
                if i == j:
                    continue
                if a.stress_index == ap.stress_index:
                    continue
                # Require shared 2-AFC option set; skip degenerate pairs.
                if a.options != ap.options:
                    continue
                # Skip if option strings are identical (would make L_cf trivial).
                if a.options[0].strip() == a.options[1].strip():
                    continue
                # Both members must reference distinct labels for L_cf to be a
                # meaningful contrast (true in the Stress-17K schema for
                # different stress indices but defend against schema drift).
                if a.label == ap.label:
                    continue
                local.append(CfPair(
                    transcription_id=tid,
                    transcription=a.transcription,
                    audio_id_a=a.audio_id,
                    audio_id_a_prime=ap.audio_id,
                    interpretation_id_a=a.interpretation_id,
                    interpretation_id_a_prime=ap.interpretation_id,
                    stress_index_a=a.stress_index,
                    stress_index_a_prime=ap.stress_index,
                    n_words=a.n_words,
                    options=a.options,
                    label_a=a.label,
                    label_a_prime=ap.label,
                    voice_name_a=a.voice_name,
                    voice_name_a_prime=ap.voice_name,
                ))
        if max_pairs_per_transcript is not None and len(local) > max_pairs_per_transcript:
            rng.shuffle(local)
            local = local[:max_pairs_per_transcript]
        pairs.extend(local)
    rng.shuffle(pairs)
    return pairs


# ---------- Expresso artifact pair construction ---------- #

def build_expresso_artifact_pairs(
    pool_rows: Sequence[dict],
    *,
    max_pairs_per_group: int = 6,
    seed: int = 0,
) -> list[ArtifactPair]:
    """Enumerate (speaker_id, transcript) groups with ≥2 distinct styles.

    `pool_rows` is the list-of-dict payload saved by audio_pool's pkl cache
    (NOT SimpleSample objects, since this module avoids importing from
    `audio_pool` to keep cold-import cheap).
    """
    rng = random.Random(seed)
    by_key: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for idx, r in enumerate(pool_rows):
        meta = r.get("meta") or {}
        sp = str(meta.get("speaker_id", ""))
        txt = (r.get("transcript") or "").strip().lower()
        style = str(meta.get("style", ""))
        if not sp or not txt or not style:
            continue
        by_key[(sp, txt)].append((idx, style))

    pairs: list[ArtifactPair] = []
    for (sp, txt), entries in by_key.items():
        # Need at least two distinct styles in this group.
        styles = {s for _, s in entries}
        if len(styles) < 2:
            continue
        local: list[ArtifactPair] = []
        for i, (idx_a, style_a) in enumerate(entries):
            for j, (idx_b, style_b) in enumerate(entries):
                if i == j or style_a == style_b:
                    continue
                local.append(ArtifactPair(
                    speaker_id=sp,
                    transcript=txt,
                    style_a=style_a,
                    style_a_prime=style_b,
                    pool_idx_a=idx_a,
                    pool_idx_a_prime=idx_b,
                ))
        if len(local) > max_pairs_per_group:
            rng.shuffle(local)
            local = local[:max_pairs_per_group]
        pairs.extend(local)
    rng.shuffle(pairs)
    return pairs


# ---------- jsonl I/O ---------- #

def write_pairs_jsonl(pairs: Iterable, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for p in pairs:
            d = asdict(p)
            # tuples become lists in JSON; preserve options as a 2-list.
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_cf_pairs_jsonl(path: Path) -> list[CfPair]:
    out: list[CfPair] = []
    valid_keys = set(CfPair.__dataclass_fields__.keys())
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            # Drop any extra fields (e.g., Stage 4 shuffled jsonl writes
            # `_shuffle_audio_a_origin_tid` for provenance); CfPair dataclass
            # is frozen and rejects unknown kwargs.
            d = {k: v for k, v in d.items() if k in valid_keys}
            d["options"] = tuple(d["options"])
            out.append(CfPair(**d))
    return out


def read_artifact_pairs_jsonl(path: Path) -> list[ArtifactPair]:
    out: list[ArtifactPair] = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            out.append(ArtifactPair(**d))
    return out
