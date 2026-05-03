"""Stage 2.5 — aggregate across seeds, compute Δ_seeds, write results.md.

Reads outputs/stage2_eval/seed{S}/summary.json for each seed S available,
emits training_logs/stage2_results.md with three tables and a one-paragraph
EXPECTED / UNEXPECTED-{A..E} mapping.

Run after all per-seed evals are done:
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/stage2_finalize.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np


SEED_DIR = ROOT / "outputs" / "stage2_eval"
OUT_MD = ROOT / "training_logs" / "stage2_results.md"


def load_seed_summary(seed: int) -> dict | None:
    path = SEED_DIR / f"seed{seed}" / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def fmt(x, prec=4):
    if x is None or (isinstance(x, float) and (x != x)):
        return "—"
    return f"{x:.{prec}f}"


def cross_seed(values: list[float]) -> tuple[float, float, float]:
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0), float(arr.max() - arr.min())


def main() -> int:
    seeds = [1234, 1235, 1236, 1237, 1238]
    summaries = {}
    for s in seeds:
        sm = load_seed_summary(s)
        if sm is not None:
            summaries[s] = sm

    if not summaries:
        print("No per-seed summaries found yet — run stage2_eval.py first.", flush=True)
        return 0

    available = sorted(summaries.keys())
    print(f"available seeds: {available}", flush=True)

    # ---- Table 1: per-seed Probe-G on StressPresso (adapter) ---- #
    table1_rows = []
    table1_rows.append("| seed | accuracy | acc CI | signed margin | margin CI | acc neutral | acc explicit |")
    table1_rows.append("|---|---:|---|---:|---|---:|---:|")
    for s in available:
        a = summaries[s]["adapter"]
        table1_rows.append(
            f"| {s} | {fmt(a['accuracy'])} | "
            f"({fmt(a['accuracy_ci_lo'])}, {fmt(a['accuracy_ci_hi'])}) | "
            f"{a['signed_margin']:+.4f} | "
            f"({a['signed_margin_ci_lo']:+.4f}, {a['signed_margin_ci_hi']:+.4f}) | "
            f"{fmt(a['accuracy_neutral'])} | {fmt(a['accuracy_explicit'])} |"
        )

    # Δ_seeds row
    accs = [summaries[s]["adapter"]["accuracy"] for s in available]
    margs = [summaries[s]["adapter"]["signed_margin"] for s in available]
    accs_neut = [summaries[s]["adapter"]["accuracy_neutral"] for s in available]
    accs_expl = [summaries[s]["adapter"]["accuracy_explicit"] for s in available]
    macc, sacc, racc = cross_seed(accs)
    mmrg, smrg, rmrg = cross_seed(margs)
    mneut, sneut, _ = cross_seed(accs_neut)
    mexpl, sexpl, _ = cross_seed(accs_expl)
    table1_rows.append(
        f"| **Δ_seeds** | mean={fmt(macc)} σ={fmt(sacc)} range={fmt(racc)} | — | "
        f"mean={mmrg:+.4f} σ={fmt(smrg)} range={fmt(rmrg)} | — | "
        f"mean={fmt(mneut)} σ={fmt(sneut)} | mean={fmt(mexpl)} σ={fmt(sexpl)} |"
    )

    # ---- Table 2: Probe-K per seed ---- #
    table2_rows = []
    table2_rows.append("| seed | head | eval_full | within | sp_full | fp16 | σ=0.1 |")
    table2_rows.append("|---|---|---:|---:|---:|---:|---:|")
    for s in available:
        for head in ("linear", "mlp2"):
            r = summaries[s]["probe_k"][head]
            table2_rows.append(
                f"| {s} | {head} | {fmt(r['eval_acc_full'])} | "
                f"{fmt(r['eval_acc_within'])} | {fmt(r['sp_acc_full'])} | "
                f"{fmt(r['robust_fp16'])} | {fmt(r['robust_sigma_01'])} |"
            )

    # ---- Table 3: baselines per seed ---- #
    table3_rows = []
    table3_rows.append("| seed | variant | accuracy | acc CI | signed margin | margin CI |")
    table3_rows.append("|---|---|---:|---|---:|---|")
    for s in available:
        for variant_key, label in [
            ("K_T",       "K_T"),
            ("cascade_T", "Cascade-T"),
            ("oracle_reconfirm", "Oracle"),
        ]:
            v = summaries[s].get(variant_key)
            if v is None:
                table3_rows.append(f"| {s} | {label} | — | — | — | — |")
                continue
            table3_rows.append(
                f"| {s} | {label} | {fmt(v['accuracy'])} | "
                f"({fmt(v['accuracy_ci_lo'])}, {fmt(v['accuracy_ci_hi'])}) | "
                f"{v['signed_margin']:+.4f} | "
                f"({v['signed_margin_ci_lo']:+.4f}, {v['signed_margin_ci_hi']:+.4f}) |"
            )

    # ---- Decision mapping (mean across seeds) ---- #
    s = available[0]
    aa = summaries[s]["adapter"]
    cc = summaries[s].get("cascade_T") or {}
    kk = summaries[s].get("K_T") or {}
    oo = summaries[s].get("oracle_reconfirm") or {}
    a_acc_n = float(np.mean([summaries[s]["adapter"]["accuracy_neutral"]    for s in available]))
    c_acc_n = float(np.mean([(summaries[s].get("cascade_T") or {}).get("accuracy_neutral", float("nan")) for s in available])) if any(summaries[s].get("cascade_T") for s in available) else float("nan")

    # ---- Write the markdown ---- #
    md = []
    md.append("# Stage 2 Results — R0 BLSP baseline (frozen-frozen)\n")
    md.append("Run date: 2026-05-02 (Stage 2). Locked stack from Stage 1: "
              "`Qwen/Qwen3-8B`, `WavLM-Large` layer 16. Adapter from "
              "`outputs/stage2/A_BLSP_seed{S}.pt`; eval per seed under "
              "`outputs/stage2_eval/seed{S}/summary.json`.\n")
    md.append(f"\nSeeds present in this report: {', '.join(map(str, available))}.\n")
    md.append("\n## Table 1 — Probe-G(A_BLSP) on StressPresso (signed margin + 95 % CI)\n")
    md.extend(table1_rows)
    md.append("\n## Table 2 — Probe-K(A_BLSP) on Stress-17K probe-eval & StressPresso\n")
    md.append("\nLinear is the held-over selector; MLP-2 (256 hidden, GeLU) is upper-bound diagnostic.\n")
    md.extend(table2_rows)
    md.append("\n## Table 3 — Cascade-T / K_T baselines + Probe-G-oracle re-confirm\n")
    md.extend(table3_rows)
    md.append("\n## Δ_seeds summary\n")
    md.append(f"- accuracy mean = {macc:.4f}, σ = {sacc:.4f}, range = {racc:.4f}")
    md.append(f"- signed margin mean = {mmrg:+.4f}, σ = {smrg:.4f}, range = {rmrg:.4f}")
    md.append(f"- accuracy neutral mean = {mneut:.4f}, σ = {sneut:.4f}")
    md.append(f"- accuracy explicit mean = {mexpl:.4f}, σ = {sexpl:.4f}")

    md.append("\n## Outcome mapping\n")
    if not np.isnan(c_acc_n):
        gap_n = a_acc_n - c_acc_n
    else:
        gap_n = float("nan")
    md.append(f"- adapter accuracy_neutral mean across seeds: {a_acc_n:.4f}")
    md.append(f"- cascade-T accuracy_neutral mean: {c_acc_n:.4f}")
    md.append(f"- Δ(adapter − cascade-T)_neutral: {gap_n:+.4f}")
    if abs(gap_n) <= 0.05 if not np.isnan(gap_n) else False:
        md.append("- **EXPECTED — cascade-equivalence replicated.** Probe-G(A_BLSP) ≈ Probe-G(Cascade-T) in the neutral condition.")
    elif gap_n > 0.05:
        md.append("- **UNEXPECTED-A — adapter beats cascade-T by >5 pp (neutral).** Investigate prompt format or augmentation pipeline; may be an unfair audio-cue.")
    else:
        md.append("- **UNEXPECTED-A (negative direction) — adapter underperforms cascade-T.** Adapter is degrading the BLSP behaviour-alignment baseline; check init scale, training stability, or grad clipping.")
    md.append("\n(For full UNEXPECTED-{B..E} mapping see kickoff §2.5; this auto-generated note covers the dominant case.)")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"saved → {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
