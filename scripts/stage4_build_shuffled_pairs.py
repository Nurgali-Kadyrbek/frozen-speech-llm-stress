"""Stage 4 / Control B prerequisite — build shuffled cf_pairs.

For each (a, a') in outputs/stage3/cf_pairs_train.jsonl, replace the
audio side with audio from a DIFFERENT transcript while keeping the
label structure (transcription_id, stress_index_a, stress_index_a_prime,
options, label_a, label_a_prime) intact.

Result: audio is decorrelated from (transcript, Φ) — Control B's
falsification test of "is audio doing real work in R1.8?".

Run:
  /raid/nurgaly/conda_envs/BTA/bin/python scripts/stage4_build_shuffled_pairs.py
"""
from __future__ import annotations

import json
import pickle
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = Path("/raid/nurgaly/datasets/Beyond_Transcript_Alignment/cache")
STRESS_POOL_PATH    = CACHE / "stress17k_probe_train_pool.pkl"
CF_TRAIN_JSONL      = ROOT / "outputs" / "stage3" / "cf_pairs_train.jsonl"
CF_SHUFFLED_JSONL   = ROOT / "outputs" / "stage4" / "cf_pairs_train_shuffled.jsonl"


def main() -> int:
    print(f"loading {STRESS_POOL_PATH.name}…", flush=True)
    with open(STRESS_POOL_PATH, "rb") as f:
        stress_pool = pickle.load(f)
    print(f"  {len(stress_pool)} rows", flush=True)
    by_audio_id = {r["audio_id"]: r for r in stress_pool}
    by_tid: dict[str, list[dict]] = {}
    for r in stress_pool:
        by_tid.setdefault(r["transcription_id"], []).append(r)
    all_audio_ids = list(by_audio_id.keys())
    print(f"  {len(by_tid)} unique transcription_ids", flush=True)

    print(f"loading {CF_TRAIN_JSONL.name}…", flush=True)
    pairs: list[dict] = []
    with CF_TRAIN_JSONL.open() as f:
        for line in f:
            pairs.append(json.loads(line))
    print(f"  {len(pairs)} cf pairs", flush=True)

    rng = random.Random(0)
    n_skipped = 0
    CF_SHUFFLED_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with CF_SHUFFLED_JSONL.open("w") as out_f:
        for cp in pairs:
            tid = cp["transcription_id"]
            # Pick two random audio_ids whose transcription_id != tid.
            tries = 0
            shuf_a = shuf_b = None
            while tries < 20:
                cand_a = rng.choice(all_audio_ids)
                cand_b = rng.choice(all_audio_ids)
                ra = by_audio_id[cand_a]
                rb = by_audio_id[cand_b]
                if (ra["transcription_id"] != tid
                        and rb["transcription_id"] != tid
                        and ra["transcription_id"] != rb["transcription_id"]):
                    shuf_a = cand_a
                    shuf_b = cand_b
                    break
                tries += 1
            if shuf_a is None:
                n_skipped += 1
                continue
            new_cp = dict(cp)  # shallow copy
            new_cp["audio_id_a"]            = shuf_a
            new_cp["audio_id_a_prime"]      = shuf_b
            new_cp["interpretation_id_a"]   = by_audio_id[shuf_a]["interpretation_id"]
            new_cp["interpretation_id_a_prime"] = by_audio_id[shuf_b]["interpretation_id"]
            new_cp["voice_name_a"]          = by_audio_id[shuf_a]["voice_name"]
            new_cp["voice_name_a_prime"]    = by_audio_id[shuf_b]["voice_name"]
            new_cp["_shuffle_audio_a_origin_tid"]      = by_audio_id[shuf_a]["transcription_id"]
            new_cp["_shuffle_audio_a_prime_origin_tid"] = by_audio_id[shuf_b]["transcription_id"]
            out_f.write(json.dumps(new_cp, ensure_ascii=False) + "\n")
    print(f"  wrote {len(pairs) - n_skipped} shuffled pairs ({n_skipped} skipped)",
          flush=True)
    print(f"  → {CF_SHUFFLED_JSONL}  "
          f"({CF_SHUFFLED_JSONL.stat().st_size/1e6:.2f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
