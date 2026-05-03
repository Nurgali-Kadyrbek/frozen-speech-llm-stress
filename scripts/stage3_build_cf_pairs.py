"""Stage 3.0 — build same-transcript counterfactual pair indices.

Two output files (both JSON Lines):
  outputs/stage3/cf_pairs_train.jsonl
      Stress-17K-raw probe-train ordered pairs (a, a') with same
      transcription_id, different gt_stress_indices.
  outputs/stage3/cf_pairs_artifact.jsonl
      Expresso-read ordered pairs (a, a'') with same (speaker, text)
      and different `style`. Indexes into the existing expresso pool
      pickle so audio is fetched at training time without re-streaming.

Also writes a one-time stress-pool cache to
  /raid/nurgaly/datasets/Beyond_Transcript_Alignment/cache/stress17k_probe_train_pool.pkl
which downstream training scripts use to skip the slow HF re-decode.

Run:
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/stage3_build_cf_pairs.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_env  # noqa: E402

setup_env()

from src.data.cf_pairs import (  # noqa: E402
    build_stress_cf_pairs, build_expresso_artifact_pairs,
    write_pairs_jsonl,
)
from src.data.stress_data import (  # noqa: E402
    load_stress17k, partition_transcript_ids,
)


CACHE = Path("/raid/nurgaly/datasets/Beyond_Transcript_Alignment/cache")
STRESS_POOL_PATH = CACHE / "stress17k_probe_train_pool.pkl"
EXPRESSO_POOL_PATH = CACHE / "expresso_pool_n6000.pkl"

STAGE3_DIR = ROOT / "outputs" / "stage3"
CF_TRAIN_JSONL = STAGE3_DIR / "cf_pairs_train.jsonl"
CF_ARTIFACT_JSONL = STAGE3_DIR / "cf_pairs_artifact.jsonl"


def cache_stress_probe_train_pool() -> int:
    """Decode Stress-17K once, partition by Stage-1 transcription_ids, and
    write a numpy-array-bearing pkl keyed by audio_id for fast training-time
    lookup."""
    print(f"loading Stress-17K-raw (full HF decode)…", flush=True)
    items = load_stress17k()
    train_ids, _ = partition_transcript_ids(items, eval_frac=0.20, seed="BTA-2026-05-02")
    train_items = [it for it in items if it.transcription_id in train_ids]
    payload = []
    for it in train_items:
        payload.append({
            "transcription_id":  it.transcription_id,
            "interpretation_id": it.interpretation_id,
            "audio_id":          it.audio_id,
            "transcription":     it.transcription,
            "words":             list(it.words),
            "stress_index":      it.stress_index,
            "n_words":           it.n_words,
            "options":           list(it.options),
            "label":             it.label,
            "audio_array":       it.audio_array,        # np.ndarray
            "audio_sr":          it.audio_sr,
            "voice_name":        it.voice_name,
            "split_origin":      it.split_origin,
        })
    STRESS_POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STRESS_POOL_PATH, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  {len(payload)} probe-train rows ({len(train_ids)} transcription_ids) cached → "
          f"{STRESS_POOL_PATH}  ({STRESS_POOL_PATH.stat().st_size/1e6:.1f} MB)",
          flush=True)
    return len(payload)


def main() -> int:
    # ---- 1. Cache stress probe-train pool if not already there ---- #
    if STRESS_POOL_PATH.exists():
        print(f"stress probe-train pool already cached at {STRESS_POOL_PATH}", flush=True)
    else:
        cache_stress_probe_train_pool()

    # ---- 2. Reload pool, build cf pairs ---- #
    print("loading cached stress probe-train pool…", flush=True)
    with open(STRESS_POOL_PATH, "rb") as f:
        stress_pool = pickle.load(f)
    print(f"  {len(stress_pool)} rows", flush=True)

    # Convert dict → Stress17kItem-shaped tuple for build_stress_cf_pairs.
    # We can synthesize a lightweight typed view since CfPair only uses fields.
    from src.data.stress_data import Stress17kItem
    items = [
        Stress17kItem(
            transcription_id=r["transcription_id"],
            interpretation_id=r["interpretation_id"],
            audio_id=r["audio_id"],
            transcription=r["transcription"],
            words=tuple(r["words"]),
            stress_index=r["stress_index"],
            n_words=r["n_words"],
            options=(r["options"][0], r["options"][1]),
            label=r["label"],
            audio_array=None,       # NOT needed for pair construction
            audio_sr=r["audio_sr"],
            voice_name=r["voice_name"],
            split_origin=r["split_origin"],
        )
        for r in stress_pool
    ]
    cf_pairs = build_stress_cf_pairs(items, seed=0)
    print(f"cf_pairs_train: {len(cf_pairs)} ordered pairs", flush=True)

    # Sanity: for each pair, label_a + label_a_prime should be 1 in this
    # dataset (because the 2-AFC option set is canonical and the labels flip).
    n_label_flip = sum(1 for p in cf_pairs if (p.label_a + p.label_a_prime) == 1)
    print(f"  label-flip invariant: {n_label_flip}/{len(cf_pairs)} pairs satisfy label_a + label_a' == 1",
          flush=True)
    # Distribution of stress_index combinations for sanity.
    from collections import Counter
    si_pairs = Counter((p.stress_index_a, p.stress_index_a_prime) for p in cf_pairs)
    top = si_pairs.most_common(5)
    print(f"  top (s_a, s_a') combos: {top}", flush=True)

    n_train_jsonl = write_pairs_jsonl(cf_pairs, CF_TRAIN_JSONL)
    print(f"  wrote → {CF_TRAIN_JSONL}  ({n_train_jsonl} rows, "
          f"{CF_TRAIN_JSONL.stat().st_size/1e6:.2f} MB)", flush=True)

    # ---- 3. Build artifact pairs from existing expresso cache ---- #
    print(f"\nloading expresso pool from {EXPRESSO_POOL_PATH}…", flush=True)
    if not EXPRESSO_POOL_PATH.exists():
        raise FileNotFoundError(
            f"expected expresso pool cache from Stage 2 at {EXPRESSO_POOL_PATH}; "
            f"run Stage 2 audio_pool.load_expresso_subset(6000) first."
        )
    with open(EXPRESSO_POOL_PATH, "rb") as f:
        expr_pool = pickle.load(f)
    print(f"  {len(expr_pool)} expresso rows", flush=True)

    art_pairs = build_expresso_artifact_pairs(expr_pool, max_pairs_per_group=6, seed=0)
    print(f"cf_pairs_artifact: {len(art_pairs)} ordered pairs", flush=True)

    # Style coverage sanity.
    from collections import Counter
    style_pairs = Counter((p.style_a, p.style_a_prime) for p in art_pairs)
    top_st = style_pairs.most_common(5)
    print(f"  top (style_a, style_a') combos: {top_st}", flush=True)

    n_art_jsonl = write_pairs_jsonl(art_pairs, CF_ARTIFACT_JSONL)
    print(f"  wrote → {CF_ARTIFACT_JSONL}  ({n_art_jsonl} rows, "
          f"{CF_ARTIFACT_JSONL.stat().st_size/1e6:.2f} MB)", flush=True)

    # ---- 4. Final summary ---- #
    print("\n=== Stage 3.0 done ===", flush=True)
    print(json.dumps({
        "stress_probe_train_rows":      len(stress_pool),
        "cf_pairs_train":               len(cf_pairs),
        "expresso_pool_rows":           len(expr_pool),
        "cf_pairs_artifact":            len(art_pairs),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
