"""Centralized env setup for Stage 0 / 0.5 scripts.

Storage (per /home/nurgaly/CLAUDE.md): all caches on /raid; never under /home.
GPU (per project rules): GPU 6 only.
"""
from __future__ import annotations
import os
import sys


HF_HOME = "/raid/nurgaly/hf_home"
HF_DATASETS_CACHE = "/raid/nurgaly/hf_cache/datasets"
DEFAULT_GPU = "6"


def setup_env(gpu: str | None = None) -> None:
    os.environ.setdefault("HF_HOME", HF_HOME)
    os.environ.setdefault("HF_DATASETS_CACHE", HF_DATASETS_CACHE)
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # HF Hub Xet backend has been returning 500s on xet-read-token; force LFS fallback.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    requested = gpu if gpu is not None else os.environ.get("CUDA_VISIBLE_DEVICES", DEFAULT_GPU)
    os.environ["CUDA_VISIBLE_DEVICES"] = requested


def banner(title: str, file=sys.stdout) -> None:
    line = "=" * 72
    print(f"\n{line}\n  {title}\n{line}", file=file, flush=True)


def report_check(name: str, passed: bool, detail: str = "", fail_log: list | None = None) -> bool:
    tag = "PASS" if passed else "FAIL"
    suffix = f" -- {detail}" if detail else ""
    print(f"  [{tag}] {name}{suffix}", flush=True)
    if not passed and fail_log is not None:
        fail_log.append(name)
    return passed
