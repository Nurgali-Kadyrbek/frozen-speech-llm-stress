"""Stage 3.5 — aggregate across seeds, compute Δ_seeds and R0 vs R1 deltas,
write stage3_results.md.

Reads outputs/stage3_eval/seed{S}/summary.json for each seed S available,
emits training_logs/stage3_results.md with five tables and the kickoff §3.5
decision branch mapping.

Run after all per-seed evals are done:
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/stage3_finalize.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np


SEED_DIR_R1 = ROOT / "outputs" / "stage3_eval"
SEED_DIR_R0 = ROOT / "outputs" / "stage2_eval"
OUT_MD = ROOT / "training_logs" / "stage3_results.md"


def load_seed_summary(seed_dir: Path, seed: int) -> dict | None:
    path = seed_dir / f"seed{seed}" / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def fmt(x, prec=4):
    if x is None or (isinstance(x, float) and (x != x)):
        return "—"
    return f"{x:.{prec}f}"


def cross_seed(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)],
                     dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0), \
           float(arr.max() - arr.min())


def main() -> int:
    seeds = [1234, 1235, 1236, 1237, 1238]
    r1 = {s: load_seed_summary(SEED_DIR_R1, s) for s in seeds}
    r0 = {s: load_seed_summary(SEED_DIR_R0, s) for s in seeds}
    available_r1 = sorted([s for s, v in r1.items() if v is not None])
    available_r0 = sorted([s for s, v in r0.items() if v is not None])
    if not available_r1:
        print("No R1 per-seed summaries found yet — run stage3_eval.py first.", flush=True)
        return 0
    print(f"R1 seeds: {available_r1}", flush=True)
    print(f"R0 seeds: {available_r0}", flush=True)

    md = []
    md.append("# Stage 3 Results — R1 structural counterfactual training\n")
    md.append("Run date: 2026-05-02 (Stage 3). Locked stack from Stage 1: "
              "`Qwen/Qwen3-8B`, `WavLM-Large` layer 16. R1 adapter from "
              "`outputs/stage3/A_R1_seed{S}.pt`; eval per seed under "
              "`outputs/stage3_eval/seed{S}/summary.json`.\n")
    md.append(f"\nSeeds present in this report: {', '.join(map(str, available_r1))}.\n")

    # ---- Table 1: per-seed Probe-G(A_R1) on StressPresso ---- #
    md.append("\n## Table 1 — Probe-G(A_R1) on StressPresso (signed margin + 95 % CI)\n")
    md.append("| seed | accuracy | acc CI | signed margin | margin CI | acc neutral | acc explicit |")
    md.append("|---|---:|---|---:|---|---:|---:|")
    accs, margs, accs_neut, accs_expl = [], [], [], []
    for s in available_r1:
        a = r1[s]["adapter"]
        md.append(
            f"| {s} | {fmt(a['accuracy'])} | "
            f"({fmt(a['accuracy_ci_lo'])}, {fmt(a['accuracy_ci_hi'])}) | "
            f"{a['signed_margin']:+.4f} | "
            f"({a['signed_margin_ci_lo']:+.4f}, {a['signed_margin_ci_hi']:+.4f}) | "
            f"{fmt(a['accuracy_neutral'])} | {fmt(a['accuracy_explicit'])} |"
        )
        accs.append(a["accuracy"]); margs.append(a["signed_margin"])
        accs_neut.append(a["accuracy_neutral"]); accs_expl.append(a["accuracy_explicit"])
    macc, sacc, racc = cross_seed(accs)
    mmrg, smrg, rmrg = cross_seed(margs)
    mneut, sneut, _  = cross_seed(accs_neut)
    mexpl, sexpl, _  = cross_seed(accs_expl)
    md.append(
        f"| **Δ_seeds** | mean={fmt(macc)} σ={fmt(sacc)} range={fmt(racc)} | — | "
        f"mean={mmrg:+.4f} σ={fmt(smrg)} range={fmt(rmrg)} | — | "
        f"mean={fmt(mneut)} σ={fmt(sneut)} | mean={fmt(mexpl)} σ={fmt(sexpl)} |"
    )

    # ---- Table 2: per-seed Probe-K ---- #
    md.append("\n## Table 2 — Probe-K(A_R1) on Stress-17K probe-eval & StressPresso\n")
    md.append("Linear is the held-over selector; MLP-2 (256 hidden, GeLU) is upper-bound diagnostic.\n")
    md.append("| seed | head | eval_full | within | sp_full | fp16 | σ=0.1 |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    pk_lin_full, pk_mlp_full = [], []
    for s in available_r1:
        for head in ("linear", "mlp2"):
            r = r1[s]["probe_k"][head]
            md.append(
                f"| {s} | {head} | {fmt(r['eval_acc_full'])} | "
                f"{fmt(r['eval_acc_within'])} | {fmt(r['sp_acc_full'])} | "
                f"{fmt(r['robust_fp16'])} | {fmt(r['robust_sigma_01'])} |"
            )
            if head == "linear":
                pk_lin_full.append(r["eval_acc_full"])
            else:
                pk_mlp_full.append(r["eval_acc_full"])

    # ---- Table 3: Branch-D 4 diagnostics + T-only ---- #
    md.append("\n## Table 3 — Branch-D conditional shortcut diagnostics (C.1 / C.2 / C.3 / C.4) + T-only\n")
    md.append("| seed | C.1 spread | C.1 PASS | C.2 in→cross ratio | C.2 PASS | C.3 frac↑ | C.3 PASS | C.4 dom_acc(A_R1) | C.4 RED_FLAG | T-only acc | contam (>0.30) |")
    md.append("|---|---:|---|---:|---|---:|---|---:|---|---:|---|")
    for s in available_r1:
        c1 = r1[s].get("C1_within_domain", {})
        c2 = r1[s].get("C2_cross_domain_transfer", {})
        c3 = r1[s].get("C3_domain_stratified_probe_g", {})
        c4 = r1[s].get("C4_domain_probe_on_adapter", {})
        to = r1[s].get("t_only_diagnostic", {})
        md.append(
            f"| {s} | {fmt(c1.get('max_spread'))} | {c1.get('PASS')} | "
            f"{fmt(c2.get('ratio_cross_over_in'),3)} | {c2.get('PASS')} | "
            f"{fmt(c3.get('fraction_uplift_positive'),3)} | {c3.get('PASS')} | "
            f"{fmt(c4.get('domain_acc_A_R1'))} | {c4.get('RED_FLAG_higher_than_raw_H')} | "
            f"{fmt(to.get('accuracy'))} | {to.get('lexical_contamination_flag')} |"
        )

    # ---- Table 4: K_T / Cascade-T / Oracle re-confirm ---- #
    md.append("\n## Table 4 — Cascade-T / K_T baselines + Probe-G-oracle re-confirm\n")
    md.append("| seed | variant | accuracy | acc CI | signed margin | margin CI |")
    md.append("|---|---|---:|---|---:|---|")
    for s in available_r1:
        for variant_key, label in [
            ("K_T",       "K_T"),
            ("cascade_T", "Cascade-T"),
            ("oracle_reconfirm", "Oracle"),
        ]:
            v = r1[s].get(variant_key)
            if v is None:
                md.append(f"| {s} | {label} | — | — | — | — |")
                continue
            md.append(
                f"| {s} | {label} | {fmt(v['accuracy'])} | "
                f"({fmt(v['accuracy_ci_lo'])}, {fmt(v['accuracy_ci_hi'])}) | "
                f"{v['signed_margin']:+.4f} | "
                f"({v['signed_margin_ci_lo']:+.4f}, {v['signed_margin_ci_hi']:+.4f}) |"
            )

    # ---- Table 5: side-by-side R0 vs R1 cohort means ---- #
    md.append("\n## Table 5 — R0 vs R1 cohort comparison (means across seeds)\n")
    md.append("| metric | R0 (Stage 2) | R1 (Stage 3) | Δ (R1 − R0) |")
    md.append("|---|---:|---:|---:|")
    # Pull R0 cohort from configs/locked_cells.yaml stage2_r0_baseline section
    R0_PROBE_G_NEUT  = 0.5122
    R0_PROBE_G_TOTAL = 0.6306
    R0_PROBE_K_LIN   = 0.2105
    R0_PROBE_K_MLP   = 0.2446
    md.append(f"| Probe-G accuracy_neutral | {R0_PROBE_G_NEUT:.4f} | {fmt(mneut)} | "
              f"{(mneut - R0_PROBE_G_NEUT):+.4f} |")
    md.append(f"| Probe-G accuracy_total   | {R0_PROBE_G_TOTAL:.4f} | {fmt(macc)} | "
              f"{(macc - R0_PROBE_G_TOTAL):+.4f} |")
    if pk_lin_full:
        m_lin, _, _ = cross_seed(pk_lin_full)
        md.append(f"| Probe-K linear eval_full | {R0_PROBE_K_LIN:.4f} | {fmt(m_lin)} | "
                  f"{(m_lin - R0_PROBE_K_LIN):+.4f} |")
    if pk_mlp_full:
        m_mlp, _, _ = cross_seed(pk_mlp_full)
        md.append(f"| Probe-K mlp2 eval_full   | {R0_PROBE_K_MLP:.4f} | {fmt(m_mlp)} | "
                  f"{(m_mlp - R0_PROBE_K_MLP):+.4f} |")

    # ---- Decision mapping (kickoff §3.5) ---- #
    md.append("\n## Outcome mapping (kickoff §3.5)\n")
    delta_pk = (cross_seed(pk_lin_full)[0] - R0_PROBE_K_LIN) if pk_lin_full else float("nan")
    delta_pg_n = mneut - R0_PROBE_G_NEUT if not np.isnan(mneut) else float("nan")
    K_T_floor = 0.290
    EXPECTED_BAND = 0.561  # R0 neutral 0.512 + 0.05
    # Aggregate Branch-D across seeds: PASS if all-seed PASS for each subgate.
    c1_all = [r1[s].get("C1_within_domain", {}).get("PASS") for s in available_r1]
    c2_all = [r1[s].get("C2_cross_domain_transfer", {}).get("PASS") for s in available_r1]
    c3_all = [r1[s].get("C3_domain_stratified_probe_g", {}).get("PASS") for s in available_r1]
    contam_all = [r1[s].get("t_only_diagnostic", {}).get("lexical_contamination_flag") for s in available_r1]
    c1_ok = all(c1_all) and bool(c1_all)
    c2_ok = all(c2_all) and bool(c2_all)
    c3_ok = all(c3_all) and bool(c3_all)
    contam_ok = (not any(contam_all))

    md.append(f"- Δ Probe-K linear vs R0 (0.211): {delta_pk:+.4f}")
    md.append(f"- Δ Probe-G neutral vs R0 (0.512): {delta_pg_n:+.4f}")
    if pk_lin_full:
        m_lin, _, _ = cross_seed(pk_lin_full)
        md.append(f"- Probe-K cohort mean: {m_lin:.4f}  (target floor K_T = {K_T_floor})")
    md.append(f"- Probe-G neutral cohort mean: {fmt(mneut)} (target floor: {EXPECTED_BAND})")
    md.append(f"- C.1 (within-domain) PASS in {sum(1 for x in c1_all if x)}/{len(c1_all)} seeds")
    md.append(f"- C.2 (cross-domain transfer) PASS in {sum(1 for x in c2_all if x)}/{len(c2_all)} seeds")
    md.append(f"- C.3 (stratified Probe-G) PASS in {sum(1 for x in c3_all if x)}/{len(c3_all)} seeds")
    md.append(f"- T-only contamination flag fired in {sum(1 for x in contam_all if x)}/{len(contam_all)} seeds")

    # Decision tree branch
    branch = "UNCLASSIFIED"
    if pk_lin_full:
        m_lin, _, _ = cross_seed(pk_lin_full)
        if (delta_pk > 0.080 and delta_pg_n > 0.050
            and m_lin > K_T_floor and mneut > EXPECTED_BAND
            and c1_ok and c2_ok and c3_ok and contam_ok):
            branch = "HEADLINE-CANDIDATE"
        elif (delta_pk > 0.040 and delta_pg_n > 0.025
              and c1_ok and c2_ok and c3_ok and contam_ok):
            branch = "SUB-HEADLINE / SMALL-EFFECT"
        elif m_lin > K_T_floor and abs(delta_pg_n) <= 0.02:
            branch = "PARTIAL-K-only (F3)"
        elif R0_PROBE_K_LIN < m_lin <= K_T_floor:
            branch = "PARTIAL-narrowing"
        elif (mneut > 0.70) or (m_lin > 0.40):
            branch = "UNEXPECTED-LEAK"
        elif (not c2_ok) and c1_ok:
            branch = "UNEXPECTED-LEAK (cross-domain transfer fail; TTS-shortcut)"
        else:
            branch = "NULL"

    md.append(f"\n**Branch: {branch}**")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"saved → {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
