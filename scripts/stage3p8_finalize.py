"""Stage 3.8.4 — aggregate cohort across 5 seeds (1234 from 3.7 + 1235-1238).

Reads outputs/stage3p8_eval/seed{S}/summary.json and Stage 3.7 single-seed
results to produce training_logs/stage3p8_results.md per kickoff §3.8.5.

Run:
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/stage3p8_finalize.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np


SEED_DIR_R1P8 = ROOT / "outputs" / "stage3p8_eval"
SEED_DIR_R0   = ROOT / "outputs" / "stage2_eval"
OUT_MD        = ROOT / "training_logs" / "stage3p8_results.md"

# R0 cohort baseline (from configs/locked_cells.yaml stage2_r0_baseline)
R0_PROBE_G_NEUT     = 0.5122
R0_PROBE_G_TOTAL    = 0.6306
R0_PROBE_G_EXPLICIT = 0.7491
R0_PROBE_K_LIN      = 0.2105
R0_PROBE_K_MLP      = 0.2446
K_T_FLOOR           = 0.290
EXPECTED_BAND       = 0.561  # R0 neutral 0.512 + 0.05


def load_seed_summary(seed: int) -> dict | None:
    p = SEED_DIR_R1P8 / f"seed{seed}" / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def fmt(x, prec=4):
    if x is None or (isinstance(x, float) and (x != x)):
        return "—"
    return f"{x:.{prec}f}"


def cs(values):
    """cohort summary: (mean, sigma, range)"""
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)],
                     dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    s = arr.std(ddof=1) if arr.size > 1 else 0.0
    return float(arr.mean()), float(s), float(arr.max() - arr.min())


def main() -> int:
    seeds = [1234, 1235, 1236, 1237, 1238]
    by_seed = {s: load_seed_summary(s) for s in seeds}
    available = sorted([s for s, v in by_seed.items() if v is not None])
    if not available:
        print("No per-seed summaries found.")
        return 0
    print(f"available seeds: {available}", flush=True)

    md = []
    md.append("# Stage 3.8 Results — R1.8 cohort (drop L_NCE; L_cf is the structural driver)\n")
    md.append("Run date: 2026-05-02 → 03 (multi-day cohort). Single-axis change from "
              "Stage 3.7: **λ_NCE 3.0 → 0.0** (drop). Everything else bit-identical "
              "(λ_cf=5, λ_artifact=1, λ_cond=0.5, λ_KL=1; aggressive_stage3_config aug; "
              "AdamW lr=5e-5, 500-step cosine, warmup=300, 400 steps; max_norm=1.0; "
              "δ_artifact=0.10).\n")
    md.append(f"Seeds: {', '.join(map(str, available))}.")
    md.append("Seed 1234 reused from Stage 3.7 (`outputs/stage3p7/A_R1p7_seed1234.pt`); "
              "Stage 3.7 cos-contrib log demonstrated L_NCE alignment 0.05-0.16 → "
              "λ_NCE=3 contributed effectively zero to the update direction.\n")

    # ---- Section A: per-seed training trajectory anchors ---- #
    md.append("\n## Section A — Per-seed training trajectory verification\n")
    md.append("Per-seed L_cf final (must be < uniform log 2 = 0.693 to confirm L_cf "
              "actually trains under λ=5), per-seed L_NCE confirmed not computed, "
              "per-seed clip_rate, per-seed mean L_cf cosine to total update.\n")
    md.append("| seed | L_cf final | < 0.693? | L_NCE final | clip_rate | L_cf cos→total (mean) |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for s in available:
        # These values are not part of summary.json; we record them at the
        # top of training_logs/stage3p8_train_seed{S}.log. Best-effort scrape.
        log_path = ROOT / "training_logs" / f"stage3p8_train_seed{s}.log"
        l_cf_final = None
        l_nce_final = None
        clip_rate = None
        cos_cf_mean = None
        if log_path.exists():
            text = log_path.read_text()
            # last `step  400` line
            for line in reversed(text.splitlines()):
                if "step  400" in line and "L_cf=" in line:
                    try:
                        l_cf_final = float(line.split("L_cf=")[1].split()[0])
                        l_nce_final = float(line.split("L_NCE=")[1].split()[0])
                    except Exception:
                        pass
                    break
            for line in reversed(text.splitlines()):
                if "main-step clip rate:" in line:
                    try:
                        clip_rate = float(line.split("=")[-1].strip().rstrip("%")) / 100.0
                    except Exception:
                        pass
                    break
            cos_cf_vals = []
            for line in text.splitlines():
                if "cos-contrib" in line and "L_cf=" in line:
                    try:
                        cos_cf_vals.append(float(line.split("L_cf=")[1].split()[0]))
                    except Exception:
                        pass
            if cos_cf_vals:
                cos_cf_mean = sum(cos_cf_vals) / len(cos_cf_vals)
        elif s == 1234:
            # Stage 3.7 carry-over.
            log_path_37 = ROOT / "training_logs" / "stage3p7_train_seed1234.log"
            if log_path_37.exists():
                text = log_path_37.read_text()
                for line in reversed(text.splitlines()):
                    if "step  400" in line and "L_cf=" in line:
                        l_cf_final = float(line.split("L_cf=")[1].split()[0])
                        l_nce_final = float(line.split("L_NCE=")[1].split()[0])
                        break
                clip_rate = 1.00
                cos_cf_vals = []
                for line in text.splitlines():
                    if "cos-contrib" in line and "L_cf=" in line:
                        try:
                            cos_cf_vals.append(float(line.split("L_cf=")[1].split()[0]))
                        except Exception:
                            pass
                if cos_cf_vals:
                    cos_cf_mean = sum(cos_cf_vals) / len(cos_cf_vals)
        below = ("✓" if (l_cf_final is not None and l_cf_final < 0.693) else
                 ("❌" if l_cf_final is not None else "—"))
        md.append(f"| {s} | {fmt(l_cf_final, 3)} | {below} | "
                  f"{fmt(l_nce_final, 3)} | {fmt(clip_rate, 3) if clip_rate is not None else '—'} | "
                  f"{fmt(cos_cf_mean, 3) if cos_cf_mean is not None else '—'} |")

    # ---- Section B: cohort Probe-G with neutral/explicit split ---- #
    md.append("\n## Section B — Cohort Probe-G with explicit/neutral split\n")
    md.append("| seed | acc total | acc neutral | acc explicit | margin total | margin neutral | margin explicit |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    accs_t, accs_n, accs_e = [], [], []
    margs_t, margs_n, margs_e = [], [], []
    for s in available:
        a = by_seed[s].get("adapter") or {}
        accs_t.append(a.get("accuracy"))
        accs_n.append(a.get("accuracy_neutral"))
        accs_e.append(a.get("accuracy_explicit"))
        margs_t.append(a.get("signed_margin"))
        margs_n.append(a.get("signed_margin_neutral"))
        margs_e.append(a.get("signed_margin_explicit"))
        md.append(f"| {s} | {fmt(a.get('accuracy'))} | "
                  f"{fmt(a.get('accuracy_neutral'))} | {fmt(a.get('accuracy_explicit'))} | "
                  f"{a.get('signed_margin', float('nan')):+.4f} | "
                  f"{a.get('signed_margin_neutral', float('nan')):+.4f} | "
                  f"{a.get('signed_margin_explicit', float('nan')):+.4f} |")
    m_t, sd_t, _   = cs(accs_t)
    m_n, sd_n, _   = cs(accs_n)
    m_e, sd_e, _   = cs(accs_e)
    mg_t, _, _     = cs(margs_t)
    mg_n, _, _     = cs(margs_n)
    mg_e, _, _     = cs(margs_e)
    md.append(f"| **cohort mean** | {fmt(m_t)} σ={fmt(sd_t)} | {fmt(m_n)} σ={fmt(sd_n)} | "
              f"{fmt(m_e)} σ={fmt(sd_e)} | {mg_t:+.4f} | {mg_n:+.4f} | {mg_e:+.4f} |")

    md.append(f"\nR0 cohort baseline: total={R0_PROBE_G_TOTAL:.4f}, neutral={R0_PROBE_G_NEUT:.4f}, "
              f"explicit={R0_PROBE_G_EXPLICIT:.4f}\n")
    md.append(f"Δ R1.8 cohort vs R0: total {(m_t - R0_PROBE_G_TOTAL):+.4f}, "
              f"neutral {(m_n - R0_PROBE_G_NEUT):+.4f}, "
              f"explicit {(m_e - R0_PROBE_G_EXPLICIT):+.4f}\n")

    # ---- Section C: cohort Probe-K ---- #
    md.append("\n## Section C — Cohort Probe-K (linear + MLP-2)\n")
    md.append("| seed | head | eval_full | within | sp_full | sp_within | fp16 | σ=0.1 |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|")
    pk_lin_full, pk_mlp_full = [], []
    pk_lin_sp,   pk_mlp_sp   = [], []
    pk_lin_within, pk_mlp_within = [], []
    for s in available:
        for head in ("linear", "mlp2"):
            r = (by_seed[s].get("probe_k") or {}).get(head, {})
            md.append(f"| {s} | {head} | {fmt(r.get('eval_acc_full'))} | "
                      f"{fmt(r.get('eval_acc_within'))} | {fmt(r.get('sp_acc_full'))} | "
                      f"{fmt(r.get('sp_acc_within'))} | {fmt(r.get('robust_fp16'))} | "
                      f"{fmt(r.get('robust_sigma_01'))} |")
            if head == "linear":
                pk_lin_full.append(r.get("eval_acc_full"))
                pk_lin_sp.append(r.get("sp_acc_full"))
                pk_lin_within.append(r.get("eval_acc_within"))
            else:
                pk_mlp_full.append(r.get("eval_acc_full"))
                pk_mlp_sp.append(r.get("sp_acc_full"))
                pk_mlp_within.append(r.get("eval_acc_within"))
    m_lin, sd_lin, range_lin = cs(pk_lin_full)
    m_mlp, sd_mlp, range_mlp = cs(pk_mlp_full)
    m_lin_sp, _, _ = cs(pk_lin_sp)
    m_mlp_sp, _, _ = cs(pk_mlp_sp)
    m_lin_w, _, _  = cs(pk_lin_within)
    m_mlp_w, _, _  = cs(pk_mlp_within)
    md.append(f"| **cohort mean** | linear | {fmt(m_lin)} σ={fmt(sd_lin)} | "
              f"{fmt(m_lin_w)} | {fmt(m_lin_sp)} | — | — | — |")
    md.append(f"| **cohort mean** | mlp2 | {fmt(m_mlp)} σ={fmt(sd_mlp)} | "
              f"{fmt(m_mlp_w)} | {fmt(m_mlp_sp)} | — | — | — |")
    md.append(f"\nR0 cohort baseline: linear eval_full={R0_PROBE_K_LIN:.4f} (σ=0.013), "
              f"mlp2 eval_full={R0_PROBE_K_MLP:.4f} (σ=0.034)")
    md.append(f"K_T floor: {K_T_FLOOR:.3f}")
    md.append(f"Δ R1.8 cohort vs R0: linear {(m_lin - R0_PROBE_K_LIN):+.4f}, "
              f"mlp2 {(m_mlp - R0_PROBE_K_MLP):+.4f}")
    md.append(f"Δ_seeds (R1.8 cohort range): linear {range_lin:.4f}, mlp2 {range_mlp:.4f}")

    # ---- Section D: Branch-D + T-only across cohort ---- #
    md.append("\n## Section D — Branch-D 4 diagnostics + T-only across cohort\n")
    md.append("| seed | C.1 PASS | C.2 PASS | C.2 ratio | C.3 frac↑ | C.3 PASS | C.4 dom_acc | T-only acc |")
    md.append("|---|---|---|---:|---:|---|---:|---:|")
    c1_all, c2_all, c3_all = [], [], []
    c2_ratios, t_onlys = [], []
    c4_doms = []
    for s in available:
        c1 = by_seed[s].get("C1_within_domain", {})
        c2 = by_seed[s].get("C2_cross_domain_transfer", {})
        c3 = by_seed[s].get("C3_domain_stratified_probe_g", {})
        c4 = by_seed[s].get("C4_domain_probe_on_adapter", {})
        to = by_seed[s].get("t_only_diagnostic", {})
        c1_all.append(c1.get("PASS"))
        c2_all.append(c2.get("PASS"))
        c3_all.append(c3.get("PASS"))
        if c2.get("ratio_cross_over_in") is not None:
            c2_ratios.append(c2.get("ratio_cross_over_in"))
        if to.get("accuracy") is not None:
            t_onlys.append(to.get("accuracy"))
        if c4.get("domain_acc_A_R1") is not None:
            c4_doms.append(c4.get("domain_acc_A_R1"))
        md.append(f"| {s} | {c1.get('PASS')} | {c2.get('PASS')} | "
                  f"{fmt(c2.get('ratio_cross_over_in'), 3)} | "
                  f"{fmt(c3.get('fraction_uplift_positive'), 3)} | {c3.get('PASS')} | "
                  f"{fmt(c4.get('domain_acc_A_R1'))} | {fmt(to.get('accuracy'))} |")
    n_c1 = sum(1 for x in c1_all if x)
    n_c2 = sum(1 for x in c2_all if x)
    n_c3 = sum(1 for x in c3_all if x)
    md.append(f"\nC.1 PASS in {n_c1}/{len(c1_all)} seeds; "
              f"C.2 (revised 3-condition) PASS in {n_c2}/{len(c2_all)} seeds; "
              f"C.3 PASS in {n_c3}/{len(c3_all)} seeds.")
    if c2_ratios:
        m_c2r, _, _ = cs(c2_ratios)
        md.append(f"C.2 ratio cohort mean: {m_c2r:.3f} (target ≥ 0.80).")
    if t_onlys:
        m_to, _, _ = cs(t_onlys)
        md.append(f"T-only cohort mean: {m_to:.3f} (must be < 0.30 — contamination floor).")
    if c4_doms:
        m_c4, _, _ = cs(c4_doms)
        md.append(f"C.4 domain probe cohort mean: {m_c4:.3f} (vs raw H ≈ 0.99).")

    # ---- Section E: B3 style probe across cohort ---- #
    md.append("\n## Section E — B3 case-discriminator (style probe) per seed\n")
    md.append("| seed | style eval_acc | chance | case |")
    md.append("|---|---:|---:|---|")
    style_accs = []
    for s in available:
        b3 = by_seed[s].get("B3_style_probe", {})
        if "error" in b3:
            md.append(f"| {s} | — | — | {b3.get('error')} |")
        else:
            style_accs.append(b3.get("eval_acc"))
            md.append(f"| {s} | {fmt(b3.get('eval_acc'))} | "
                      f"{fmt(b3.get('chance'))} | {b3.get('case')} |")
    if style_accs:
        m_st, _, _ = cs(style_accs)
        md.append(f"\nStyle-probe cohort mean: {m_st:.3f} (Stage 3.7 single seed: 0.425).")

    # ---- Section F: REVISED-* branch determination ---- #
    md.append("\n## Section F — REVISED-* branch determination\n")
    headline_thresholds = {
        "linear_cohort_above_R0_plus_020":  (m_lin > R0_PROBE_K_LIN + 0.020),
        "mlp2_cohort_above_K_T":            (m_mlp > K_T_FLOOR),
        "neutral_cohort_above_R0_plus_025": (m_n > R0_PROBE_G_NEUT + 0.025),
        "explicit_cohort_above_R0_plus_020": (m_e > R0_PROBE_G_EXPLICIT + 0.020),
        "C1_cohort_PASS":                   (n_c1 == len(c1_all) and len(c1_all) > 0),
        "C2_cohort_PASS":                   (n_c2 == len(c2_all) and len(c2_all) > 0),
        "C3_cohort_PASS":                   (n_c3 == len(c3_all) and len(c3_all) > 0),
        "delta_seeds_below_025":            (range_lin <= 0.025),
    }
    md.append("| condition | required for HEADLINE | observed |")
    md.append("|---|---|---|")
    for k, v in headline_thresholds.items():
        md.append(f"| {k} | true | {v} |")

    sub_thresholds = {
        "linear_cohort_above_R0_plus_020": headline_thresholds["linear_cohort_above_R0_plus_020"],
        "C1_cohort_PASS":                   headline_thresholds["C1_cohort_PASS"],
        "neutral_within_R0_pm_015":         abs(m_n - R0_PROBE_G_NEUT) <= 0.015,
        "explicit_within_R0_pm_010":        abs(m_e - R0_PROBE_G_EXPLICIT) <= 0.010,
    }
    explicit_thresholds = {
        "linear_cohort_above_R0_plus_020": headline_thresholds["linear_cohort_above_R0_plus_020"],
        "explicit_cohort_above_R0_plus_020": headline_thresholds["explicit_cohort_above_R0_plus_020"],
        "neutral_within_R0_pm_010":         abs(m_n - R0_PROBE_G_NEUT) <= 0.010,
    }

    branch = "UNCLASSIFIED"
    if all(headline_thresholds.values()):
        branch = "REVISED-HEADLINE-CONFIRMED"
    elif all(sub_thresholds.values()):
        branch = "REVISED-PARTIAL-K-only (F3 case)"
    elif all(explicit_thresholds.values()):
        branch = "REVISED-EXPLICIT-ONLY (LLM uses Φ when asked)"
    elif (m_lin <= R0_PROBE_K_LIN + 0.010):
        branch = "REVISED-FLUKE (R1.7 single-seed gain not reproducible)"
    elif (m_n < R0_PROBE_G_NEUT - 0.015) or (m_e < R0_PROBE_G_EXPLICIT - 0.020):
        branch = "REVISED-INSTABILITY"
    elif (m_lin > 0.40) or (m_n > 0.65):
        branch = "UNEXPECTED-CONVERGENCE (audit leakage)"
    else:
        branch = "UNCLASSIFIED — see threshold table above"
    md.append(f"\n**Branch: {branch}**\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"saved → {OUT_MD}")
    print(f"branch: {branch}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
