"""Stage 1b + 1c — raw Probe-K layer sweep on cached encoder features.

Reads outputs/cache/{wavlm_pooled, whisper_pooled, kt_pooled}.pt and runs the
Probe-K protocol per the session prompt:

  - Stress-17K-raw partitioned by transcription_id (deterministic 80/20).
  - Probe head families: LINEAR (selector) + MLP-2 (256 hidden, GeLU; diagnostic).
  - Stratifications:
      held-out-transcript            — transcription_id ∉ train_ids.
      within-transcript-counterfactual — restrict argmax to the candidate
                                         stress positions for that transcript.
  - Robustness eval columns: fp32 default, fp16-cast, additive Gaussian σ∈{0.01,0.1}.
  - StressPresso held-out: scored with the same probes AFTER the layer is
    locked. NOT used to drive cell selection (governing rule).

Output: outputs/stage1bc/results.json + a printed Markdown table.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_env, banner  # noqa: E402

setup_env()

import torch  # noqa: E402

from src.data.stress_data import partition_transcript_ids  # noqa: E402
from src.probes.probe_k import (  # noqa: E402
    ProbeK, FitConfig, fit_probe, predict, masked_softmax_logits,
    within_transcript_argmax, accuracy,
)


# ---------- Cell specifications ---------- #

WAVLM_CELLS = [
    {"name": "WavLM-L8",        "mode": "single", "layers": [8]},
    {"name": "WavLM-L12",       "mode": "single", "layers": [12]},
    {"name": "WavLM-L16",       "mode": "single", "layers": [16]},
    {"name": "WavLM-WS{8..16}", "mode": "wsum",   "layers": list(range(8, 17))},
    {"name": "WavLM-WS{1..24}", "mode": "wsum",   "layers": list(range(1, 25))},
]
WHISPER_CELLS = [
    {"name": "Whisper-L22",     "mode": "single", "layers": [22]},
    {"name": "Whisper-L32",     "mode": "single", "layers": [32]},
]
KT_CELLS = [
    # The K_T cache stores pool(embed_tokens(transcript)) as a single pseudo-layer.
    # 1-indexed to match the WavLM-cache convention used by select_layers().
    {"name": "K_T (Qwen3-embed)", "mode": "single", "layers": [1]},
]


# ---------- Cache I/O ---------- #

def load_cache(path: Path) -> dict:
    print(f"  loading {path} ...", flush=True)
    return torch.load(path, weights_only=False, map_location="cpu")


def select_layers(pooled: torch.Tensor, cache_layer_index: list[int],
                  layers_wanted: list[int]) -> torch.Tensor:
    """pooled: (N, L_cached, d). cache_layer_index lists the layer numbers stored
    in axis 1, in order. Returns (N, len(layers_wanted), d) selecting in order."""
    name_to_pos = {li: pos for pos, li in enumerate(cache_layer_index)}
    positions = [name_to_pos[li] for li in layers_wanted]
    return pooled[:, positions, :]


# ---------- Probe-K runner ---------- #

def split_indices(cache: dict, train_ids: set[str], eval_ids: set[str], source: str) -> tuple[list[int], list[int], list[int]]:
    train_idx, eval_idx, sp_idx = [], [], []
    for i, src in enumerate(cache["sources"]):
        if src != source and src != "stresspresso":
            continue
        if src == "stress17k":
            tid = cache["transcription_ids"][i]
            if tid in train_ids:
                train_idx.append(i)
            elif tid in eval_ids:
                eval_idx.append(i)
        elif src == "stresspresso":
            sp_idx.append(i)
    return train_idx, eval_idx, sp_idx


def build_within_transcript_lookup(cache: dict, indices: list[int]) -> dict[str, list[int]]:
    """For each transcription_id present in `indices`, list the stress positions
    that appear across its rows. argmax restricted to this set is the
    within-transcript-counterfactual prediction.
    """
    lookup: dict[str, set[int]] = defaultdict(set)
    for i in indices:
        tid = cache["transcription_ids"][i]
        lookup[tid].add(int(cache["stress_index"][i].item()))
    return {k: sorted(v) for k, v in lookup.items()}


def run_cell(*, cell: dict, cache: dict, cache_layers: list[int],
             train_idx: list[int], eval_idx: list[int], sp_idx: list[int],
             fit_cfg: FitConfig, device: str, head: str,
             n_max_classes: int) -> dict:

    pooled = cache["pooled"]
    d_in = pooled.shape[-1]
    sub_train = select_layers(pooled[train_idx], cache_layers, cell["layers"])
    sub_eval  = select_layers(pooled[eval_idx],  cache_layers, cell["layers"])
    sub_sp    = select_layers(pooled[sp_idx],    cache_layers, cell["layers"])

    y_train = cache["stress_index"][train_idx]
    y_eval  = cache["stress_index"][eval_idx]
    y_sp    = cache["stress_index"][sp_idx]
    nw_train = cache["n_words"][train_idx]
    nw_eval  = cache["n_words"][eval_idx]
    nw_sp    = cache["n_words"][sp_idx]

    probe = ProbeK(
        d_in=d_in, n_classes=n_max_classes,
        cell_mode=cell["mode"], n_layers_used=len(cell["layers"]), head=head,
    )
    fit_info = fit_probe(probe, sub_train, y_train, nw_train,
                         sub_eval, y_eval, nw_eval,
                         fit_cfg, device)
    final_acc_eval = fit_info["final_eval_acc"]
    best_acc_eval  = fit_info["best"]["eval_acc"]

    # held-out-transcript predictions on probe-eval (default, fp32).
    p_eval_full = predict(probe, sub_eval, nw_eval, device)
    eval_acc_full = accuracy(p_eval_full, y_eval)

    # within-transcript-counterfactual on probe-eval.
    eval_tids = [cache["transcription_ids"][i] for i in eval_idx]
    eval_cands = build_within_transcript_lookup(cache, eval_idx)
    p_eval_within = within_transcript_argmax(probe, sub_eval, eval_tids, eval_cands, device)
    eval_acc_within = accuracy(p_eval_within, y_eval)

    # Robustness: fp16 cast + Gaussian σ.
    rob = {}
    rob["fp16"]    = accuracy(predict(probe, sub_eval, nw_eval, device, dtype=torch.float16), y_eval)
    for s in (0.01, 0.1):
        rob[f"sigma_{s}"] = accuracy(predict(probe, sub_eval, nw_eval, device, noise_sigma=s, noise_seed=0), y_eval)

    # StressPresso held-out (REPORT ONLY; not used for selection).
    sp_acc_full = float("nan")
    sp_acc_within = float("nan")
    if sp_idx:
        p_sp_full = predict(probe, sub_sp, nw_sp, device)
        sp_acc_full = accuracy(p_sp_full, y_sp)
        sp_tids = [cache["transcription_ids"][i] for i in sp_idx]
        sp_cands = build_within_transcript_lookup(cache, sp_idx)
        p_sp_within = within_transcript_argmax(probe, sub_sp, sp_tids, sp_cands, device)
        sp_acc_within = accuracy(p_sp_within, y_sp)

    # Random-baseline accuracy estimate (uniform over n_words per item).
    chance_eval   = float((1.0 / nw_eval.float()).mean().item())
    chance_eval_w = float((1.0 / torch.tensor([len(eval_cands.get(t, [1])) for t in eval_tids]).float()).mean().item())

    return {
        "cell": cell["name"],
        "head": head,
        "n_train": len(train_idx),
        "n_eval":  len(eval_idx),
        "n_sp":    len(sp_idx),
        "fit_best_eval_acc":  best_acc_eval,
        "fit_final_eval_acc": final_acc_eval,
        "eval_acc_full":      eval_acc_full,
        "eval_acc_within":    eval_acc_within,
        "sp_acc_full":        sp_acc_full,
        "sp_acc_within":      sp_acc_within,
        "robust_fp16":        rob["fp16"],
        "robust_sigma_001":   rob["sigma_0.01"],
        "robust_sigma_01":    rob["sigma_0.1"],
        "chance_eval_full":   chance_eval,
        "chance_eval_within": chance_eval_w,
    }


def run_section(name: str, *, cells: list[dict], cache_path: Path,
                cache_layers_attr: str, train_ids: set[str], eval_ids: set[str],
                fit_cfg: FitConfig, device: str, n_max_classes: int) -> list[dict]:
    banner(f"Stage 1b/c — {name} ({cache_path.name})")
    cache = load_cache(cache_path)
    cache_layers = cache.get(cache_layers_attr, list(range(1, cache["pooled"].shape[1] + 1)))
    train_idx, eval_idx, sp_idx = split_indices(cache, train_ids, eval_ids, source="stress17k")
    print(f"  train rows: {len(train_idx)}  eval rows: {len(eval_idx)}  StressPresso rows: {len(sp_idx)}", flush=True)

    rows = []
    for cell in cells:
        for head in ("linear", "mlp2"):
            t0 = time.time()
            row = run_cell(cell=cell, cache=cache, cache_layers=cache_layers,
                           train_idx=train_idx, eval_idx=eval_idx, sp_idx=sp_idx,
                           fit_cfg=fit_cfg, device=device, head=head,
                           n_max_classes=n_max_classes)
            row["wallclock_s"] = round(time.time() - t0, 2)
            rows.append(row)
            print(f"  {row['cell']:<20s} {row['head']:<6s}  "
                  f"eval_full={row['eval_acc_full']:.3f}  within={row['eval_acc_within']:.3f}  "
                  f"sp_full={row['sp_acc_full']:.3f}  fp16={row['robust_fp16']:.3f}  "
                  f"σ0.1={row['robust_sigma_01']:.3f}  ({row['wallclock_s']}s)", flush=True)
    return rows


def main() -> int:
    print(f"transformers=={__import__('transformers').__version__}, torch=={torch.__version__}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES','?')}", flush=True)
    if not torch.cuda.is_available():
        print("ERROR: no CUDA visible."); return 1
    device = "cuda"

    cache_dir = ROOT / "outputs" / "cache"
    wavlm_path   = cache_dir / "wavlm_pooled.pt"
    whisper_path = cache_dir / "whisper_pooled.pt"
    kt_path      = cache_dir / "kt_pooled.pt"
    for p in (wavlm_path, whisper_path, kt_path):
        if not p.exists():
            print(f"ERROR: missing {p}; run the corresponding cache script first.")
            return 1

    # Compute the train/eval id partition once from the WavLM cache (its row
    # set is canonical; the other caches share the same row order).
    banner("Partitioning Stress-17K-raw transcription_ids (80/20, deterministic)")
    src = torch.load(wavlm_path, weights_only=False, map_location="cpu")
    s17_tids = sorted({src["transcription_ids"][i] for i, s in enumerate(src["sources"]) if s == "stress17k"})
    print(f"  unique Stress-17K transcription_ids: {len(s17_tids)}", flush=True)

    # Use the deterministic-hash partitioner from src.data.stress_data; rebuild
    # an "items" iterable just for that helper.
    fake_items = [type("X", (), {"transcription_id": t})() for t in s17_tids]
    train_ids, eval_ids = partition_transcript_ids(fake_items, eval_frac=0.20)
    print(f"  partitioned: train_ids={len(train_ids)}  eval_ids={len(eval_ids)}", flush=True)

    n_max = max(int(src["n_words"].max().item()), int(src["stress_index"].max().item()) + 1)
    print(f"  n_max stress classes: {n_max}", flush=True)

    fit_cfg = FitConfig(epochs=80, lr=1e-3, batch_size=256, weight_decay=1e-4, seed=0)

    all_rows: list[dict] = []
    all_rows.extend(run_section("WavLM cells", cells=WAVLM_CELLS, cache_path=wavlm_path,
                                cache_layers_attr="layer_indices_unused",
                                train_ids=train_ids, eval_ids=eval_ids,
                                fit_cfg=fit_cfg, device=device, n_max_classes=n_max))
    all_rows.extend(run_section("Whisper cells", cells=WHISPER_CELLS, cache_path=whisper_path,
                                cache_layers_attr="layer_indices",
                                train_ids=train_ids, eval_ids=eval_ids,
                                fit_cfg=fit_cfg, device=device, n_max_classes=n_max))
    all_rows.extend(run_section("K_T baseline (Qwen3 embed_tokens)", cells=KT_CELLS,
                                cache_path=kt_path,
                                cache_layers_attr="layer_indices_kt_baseline",
                                train_ids=train_ids, eval_ids=eval_ids,
                                fit_cfg=fit_cfg, device=device, n_max_classes=n_max))

    # Save raw rows
    out_dir = ROOT / "outputs" / "stage1bc"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps({
        "rows": all_rows,
        "fit_cfg": asdict(fit_cfg),
        "n_train_ids": len(train_ids),
        "n_eval_ids":  len(eval_ids),
        "n_max_classes": n_max,
    }, indent=2))

    # Lock the highest-LINEAR-eval cell among WavLM cells (Stress-17K probe-eval).
    banner("Layer-selection decision (LINEAR probe on Stress-17K probe-eval)")
    wavlm_linear = [r for r in all_rows if r["cell"].startswith("WavLM-") and r["head"] == "linear"]
    wavlm_linear.sort(key=lambda r: r["eval_acc_full"], reverse=True)
    print("  ranked WavLM cells (linear probe, held-out-transcript probe-eval):")
    for r in wavlm_linear:
        print(f"    {r['cell']:<22s}  eval_acc_full={r['eval_acc_full']:.4f}  within={r['eval_acc_within']:.4f}", flush=True)
    locked = wavlm_linear[0] if wavlm_linear else None
    if locked is None:
        print("  ERROR: no WavLM linear results.")
        return 2
    print(f"\n  LOCKED: E-WavLM-train = {locked['cell']!r}", flush=True)

    # Was bottleneck pattern observed?
    by_head = {(r["cell"], r["head"]): r for r in all_rows}
    wp22 = by_head.get(("Whisper-L22", "linear"))
    wp32 = by_head.get(("Whisper-L32", "linear"))
    kt   = by_head.get(("K_T (Qwen3-embed)", "linear"))
    bottleneck_obs = (
        locked["eval_acc_full"] > (wp22["eval_acc_full"] if wp22 else 0)
        and (wp22 and wp22["eval_acc_full"] > wp32["eval_acc_full"] if (wp22 and wp32) else False)
    )
    print(f"  bottleneck pattern (WavLM-best > Whisper-22 > Whisper-32 ≈ K_T): {bottleneck_obs}", flush=True)

    summary = {
        "locked_wavlm_cell": locked["cell"],
        "locked_acc_eval_full": locked["eval_acc_full"],
        "locked_acc_eval_within": locked["eval_acc_within"],
        "locked_acc_sp_full":   locked["sp_acc_full"],
        "locked_acc_sp_within": locked["sp_acc_within"],
        "whisper_l22_linear":   (wp22 or {}).get("eval_acc_full"),
        "whisper_l32_linear":   (wp32 or {}).get("eval_acc_full"),
        "kt_linear":            (kt   or {}).get("eval_acc_full"),
        "bottleneck_pattern":   bottleneck_obs,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  rows    → {out_dir / 'results.json'}", flush=True)
    print(f"  summary → {out_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
