"""One-off inspection of StressPresso + Stress-17K-raw schemas.

Goal: understand the meaning of each field so the Stage-1 prompt builder and
Probe-K label extractor are unambiguous.
"""
from __future__ import annotations
import sys, json, pprint
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.env import setup_env, banner
setup_env()

from datasets import load_dataset


def show(label, ds_iter, n=3):
    banner(label)
    for i, rec in enumerate(ds_iter):
        if i >= n:
            break
        print(f"\nrecord #{i}:")
        for k, v in rec.items():
            if k == "audio":
                if isinstance(v, dict):
                    arr = v.get("array")
                    sr = v.get("sampling_rate")
                    path = v.get("path")
                    arr_info = f"array shape={getattr(arr,'shape', None)} dtype={getattr(arr,'dtype', None)}" if arr is not None else "array=None"
                    print(f"  audio: {arr_info}, sr={sr}, path={path}")
                else:
                    print(f"  audio: {type(v).__name__}={v}")
            else:
                s = str(v)
                if len(s) > 240:
                    s = s[:240] + "…(truncated)"
                print(f"  {k}: {s}")


def main():
    sp_test = load_dataset("slprl/StressPresso", split="test", streaming=True)
    show("StressPresso test, first 3 records", sp_test, n=3)

    s17_full = load_dataset("slprl/Stress-17K-raw", split="train_full", streaming=True)
    show("Stress-17K-raw train_full, first 3 records", s17_full, n=3)

    # Confirm split sizes (need them for partitioning).
    banner("Split metadata (non-streaming load_dataset_builder)")
    from datasets import load_dataset_builder
    for hf_id, splits in [
        ("slprl/StressPresso", ["test"]),
        ("slprl/Stress-17K-raw", None),  # auto
    ]:
        try:
            b = load_dataset_builder(hf_id)
            info = b.info
            print(f"\n{hf_id}: features = {list(info.features.keys()) if info.features else 'n/a'}")
            if info.splits:
                for sn, sinfo in info.splits.items():
                    print(f"  split {sn}: num_examples={sinfo.num_examples}, num_bytes={sinfo.num_bytes}")
        except Exception as exc:
            print(f"\n{hf_id}: builder load failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
