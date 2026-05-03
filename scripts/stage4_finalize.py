"""Stage 4.5 — aggregate Control A + B + shortcut-probe cohort.

Reads:
  outputs/stage3p8_eval/seed{1234..1238}/summary.json   (R1.8 cohort baseline)
  outputs/stage4_eval/textK_seed{1234,1235,1236}/summary.json   (Control A)
  outputs/stage4_eval/shuffle_seed{1234,1235,1236}/summary.json (Control B)
  outputs/stage4/cohort_shortcut_probes.json                    (per-seed tts/domain)

Writes training_logs/stage4_results.md with sections A-E and the
CONTROLS-PASS / *-FAIL / PARTIAL-PASS branch determination.

Run:
  /raid/nurgaly/conda_envs/BTA/bin/python scripts/stage4_finalize.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np


R1P8_EVAL_DIR     = ROOT / "outputs" / "stage3p8_eval"
STAGE4_EVAL_DIR   = ROOT / "outputs" / "stage4_eval"
COHORT_SHORTCUT   = ROOT / "outputs" / "stage4" / "cohort_shortcut_probes.json"
OUT_MD            = ROOT / "training_logs" / "stage4_results.md"

K_T_FLOOR              = 0.290    # K_T baseline Probe-K from Stage 1
R0_PROBE_K_LIN         = 0.2105
R0_PROBE_K_MLP         = 0.2446
R0_PROBE_G_NEUT        = 0.5122
R0_PROBE_G_TOTAL       = 0.6306
R0_PROBE_G_EXPLICIT    = 0.7491

# Stage 4 PASS thresholds (kickoff §4.0–4.2):
CONTROL_A_MLP_PASS_LE  = 0.310    # K_T + 0.02
CONTROL_B_MLP_PASS_LE  = 0.275    # R0 mlp2 cohort 0.245 + 0.030
SHORTCUT_TTS_PASS_LT   = 0.79     # raw H 0.89 - 0.10
SHORTCUT_DOM_PASS_LT   = 0.89     # raw H 0.99 - 0.10

R1P8_COHORT_LINEAR_MEAN = 0.2256
R1P8_COHORT_MLP2_MEAN   = 0.3059


def load_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def fmt(x, prec=4):
    if x is None or (isinstance(x, float) and (x != x)):
        return "—"
    return f"{x:.{prec}f}"


def cs(values):
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)],
                     dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    s = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), s


def main() -> int:
    md = []
    md.append("# Stage 4 Results — Controls A + B + shortcut-probe cohort validation\n")
    md.append("Run date: 2026-05-02 → 03 (multi-day session). Stage 4 runs three "
              "controls in parallel to falsification-test the Stage 3.8 R1.8 "
              "MLP-2 > K_T finding. PASS gates: Control A MLP-2 ≤ "
              f"{CONTROL_A_MLP_PASS_LE} (K_T + 0.02), Control B MLP-2 ≤ "
              f"{CONTROL_B_MLP_PASS_LE}, shortcut probes drop ≥ 0.10 vs raw H on ≥ 4/5 seeds.\n")

    # ---- Section A: Control A — text-only adapter ---- #
    md.append("\n## Section A — Control A (text-only A_textK)\n")
    md.append("Single change axis: replace audio path with P(embed_tokens(transcript)) "
              "where P is a frozen 4096→1024 linear projection fit on 500 "
              "Stress-17K probe-train pool(K_T)→pool(H) pairs (proj_P.pt). All "
              "other R1.8 hyperparameters (λ_cf=5, λ_NCE=0, etc.) bit-identical.\n")
    seeds_a = [1234, 1235, 1236]
    summaries_a = {}
    for s in seeds_a:
        path = STAGE4_EVAL_DIR / f"textK_seed{s}" / "summary.json"
        v = load_summary(path)
        if v is not None:
            summaries_a[s] = v
    print(f"Control A seeds present: {sorted(summaries_a.keys())}", flush=True)
    md.append("| seed | linear eval_full | mlp2 eval_full | linear sp_full | mlp2 sp_full | Probe-G_neutral | Probe-G_explicit |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    a_lin, a_mlp = [], []
    a_lin_sp, a_mlp_sp = [], []
    a_pgN, a_pgE = [], []
    for s in seeds_a:
        v = summaries_a.get(s)
        if v is None:
            md.append(f"| {s} | — | — | — | — | — | — |")
            continue
        pk = v.get("probe_k", {})
        lin = pk.get("linear", {})
        mlp = pk.get("mlp2",   {})
        ad  = v.get("adapter", {})
        a_lin.append(lin.get("eval_acc_full"))
        a_mlp.append(mlp.get("eval_acc_full"))
        a_lin_sp.append(lin.get("sp_acc_full"))
        a_mlp_sp.append(mlp.get("sp_acc_full"))
        a_pgN.append(ad.get("accuracy_neutral"))
        a_pgE.append(ad.get("accuracy_explicit"))
        md.append(f"| {s} | {fmt(lin.get('eval_acc_full'))} | "
                  f"{fmt(mlp.get('eval_acc_full'))} | {fmt(lin.get('sp_acc_full'))} | "
                  f"{fmt(mlp.get('sp_acc_full'))} | "
                  f"{fmt(ad.get('accuracy_neutral'))} | "
                  f"{fmt(ad.get('accuracy_explicit'))} |")
    a_lin_m, a_lin_sd = cs(a_lin)
    a_mlp_m, a_mlp_sd = cs(a_mlp)
    a_lin_sp_m, _ = cs(a_lin_sp)
    a_mlp_sp_m, _ = cs(a_mlp_sp)
    a_pgN_m, _ = cs(a_pgN)
    a_pgE_m, _ = cs(a_pgE)
    md.append(f"| **cohort mean** | {fmt(a_lin_m)} σ={fmt(a_lin_sd)} | "
              f"{fmt(a_mlp_m)} σ={fmt(a_mlp_sd)} | {fmt(a_lin_sp_m)} | "
              f"{fmt(a_mlp_sp_m)} | {fmt(a_pgN_m)} | {fmt(a_pgE_m)} |")
    a_pass = bool(a_mlp_m is not None and not np.isnan(a_mlp_m) and a_mlp_m <= CONTROL_A_MLP_PASS_LE)
    md.append(f"\nControl A MLP-2 cohort mean = {fmt(a_mlp_m)}; "
              f"PASS threshold ≤ {CONTROL_A_MLP_PASS_LE}; **PASS={a_pass}**")
    md.append(f"R1.8 cohort MLP-2 = {R1P8_COHORT_MLP2_MEAN}; gap "
              f"R1.8 − Control A = "
              f"{R1P8_COHORT_MLP2_MEAN - (a_mlp_m if a_mlp_m else 0):+.4f}")

    # ---- Section B: Control B — shuffled audio ---- #
    md.append("\n## Section B — Control B (shuffled-audio A_R1.8-shuffle)\n")
    md.append("Single change axis: cf_pairs_train_shuffled.jsonl decorrelates audio "
              "from (transcript, Φ) — for each pair (a, a'), audio is drawn from "
              "DIFFERENT transcripts while keeping label structure intact.\n")
    seeds_b = [1234, 1235, 1236]
    summaries_b = {}
    for s in seeds_b:
        path = STAGE4_EVAL_DIR / f"shuffle_seed{s}" / "summary.json"
        v = load_summary(path)
        if v is not None:
            summaries_b[s] = v
    print(f"Control B seeds present: {sorted(summaries_b.keys())}", flush=True)
    md.append("| seed | linear eval_full | mlp2 eval_full | linear sp_full | mlp2 sp_full | Probe-G_neutral | Probe-G_explicit |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    b_lin, b_mlp = [], []
    b_lin_sp, b_mlp_sp = [], []
    b_pgN, b_pgE = [], []
    for s in seeds_b:
        v = summaries_b.get(s)
        if v is None:
            md.append(f"| {s} | — | — | — | — | — | — |")
            continue
        pk = v.get("probe_k", {})
        lin = pk.get("linear", {})
        mlp = pk.get("mlp2",   {})
        ad  = v.get("adapter", {})
        b_lin.append(lin.get("eval_acc_full"))
        b_mlp.append(mlp.get("eval_acc_full"))
        b_lin_sp.append(lin.get("sp_acc_full"))
        b_mlp_sp.append(mlp.get("sp_acc_full"))
        b_pgN.append(ad.get("accuracy_neutral"))
        b_pgE.append(ad.get("accuracy_explicit"))
        md.append(f"| {s} | {fmt(lin.get('eval_acc_full'))} | "
                  f"{fmt(mlp.get('eval_acc_full'))} | {fmt(lin.get('sp_acc_full'))} | "
                  f"{fmt(mlp.get('sp_acc_full'))} | "
                  f"{fmt(ad.get('accuracy_neutral'))} | "
                  f"{fmt(ad.get('accuracy_explicit'))} |")
    b_lin_m, b_lin_sd = cs(b_lin)
    b_mlp_m, b_mlp_sd = cs(b_mlp)
    b_lin_sp_m, _ = cs(b_lin_sp)
    b_mlp_sp_m, _ = cs(b_mlp_sp)
    b_pgN_m, _ = cs(b_pgN)
    b_pgE_m, _ = cs(b_pgE)
    md.append(f"| **cohort mean** | {fmt(b_lin_m)} σ={fmt(b_lin_sd)} | "
              f"{fmt(b_mlp_m)} σ={fmt(b_mlp_sd)} | {fmt(b_lin_sp_m)} | "
              f"{fmt(b_mlp_sp_m)} | {fmt(b_pgN_m)} | {fmt(b_pgE_m)} |")
    b_pass = bool(b_mlp_m is not None and not np.isnan(b_mlp_m) and b_mlp_m <= CONTROL_B_MLP_PASS_LE)
    md.append(f"\nControl B MLP-2 cohort mean = {fmt(b_mlp_m)}; "
              f"PASS threshold ≤ {CONTROL_B_MLP_PASS_LE}; **PASS={b_pass}**")

    # ---- Section C: Shortcut probes on R1.8 cohort ---- #
    md.append("\n## Section C — Shortcut probes on R1.8 cohort A_R1.8(H)\n")
    sc = load_summary(COHORT_SHORTCUT) or {}
    by_seed = sc.get("by_seed", {})
    md.append("| seed | tts_vs_real | drop vs raw H 0.89 | domain | drop vs raw H 0.99 | per-seed PASS |")
    md.append("|---|---:|---:|---:|---:|---|")
    sc_pass_ct = 0
    for s in sorted(by_seed.keys()):
        v = by_seed[s]
        md.append(f"| {s} | {fmt(v['tts_vs_real_eval_acc'])} | "
                  f"{v['tts_vs_real_drop_vs_raw_H_baseline']:+.4f} | "
                  f"{fmt(v['domain_eval_acc'])} | "
                  f"{v['domain_drop_vs_raw_H_baseline']:+.4f} | "
                  f"{v['per_seed_PASS']} |")
        if v.get("per_seed_PASS"):
            sc_pass_ct += 1
    cohort_summary = sc.get("cohort_summary", {})
    sc_pass = bool(cohort_summary.get("cohort_PASS", False))
    md.append(f"\ntts_vs_real cohort mean = {fmt(cohort_summary.get('tts_vs_real_cohort_mean'))} "
              f"σ={fmt(cohort_summary.get('tts_vs_real_cohort_sigma'))}; "
              f"domain cohort mean = {fmt(cohort_summary.get('domain_cohort_mean'))} "
              f"σ={fmt(cohort_summary.get('domain_cohort_sigma'))}")
    md.append(f"Per-seed PASS in {sc_pass_ct}/{len(by_seed)} seeds; "
              f"cohort PASS (≥ 4/5): **{sc_pass}**")

    # ---- Section D: branch determination ---- #
    md.append("\n## Section D — Composite decision branch\n")
    md.append("| gate | required | observed | PASS? |")
    md.append("|---|---|---:|---|")
    md.append(f"| Control A MLP-2 ≤ {CONTROL_A_MLP_PASS_LE} | required | "
              f"{fmt(a_mlp_m)} | {a_pass} |")
    md.append(f"| Control B MLP-2 ≤ {CONTROL_B_MLP_PASS_LE} | required | "
              f"{fmt(b_mlp_m)} | {b_pass} |")
    md.append(f"| Shortcut probes cohort PASS (≥ 4/5) | required | "
              f"{sc_pass_ct}/{len(by_seed)} | {sc_pass} |")
    overall = a_pass and b_pass and sc_pass
    if overall:
        branch = "CONTROLS-PASS"
        narrative = (
            "**R1.8's MLP-2 > K_T finding is validated as audio-derived.** "
            "Control A confirms the loss is not optimizer regularization "
            "(text-only adapter doesn't reach K_T). Control B confirms "
            "audio is doing real work (shuffled-audio adapter doesn't "
            "reproduce R1.8's gain). Shortcut probes confirm the adapter "
            "compresses nuisance signal. Bounded F3 finding is "
            "publishable; Stage 6 minimal-LoRA upper-bound is the next step."
        )
    elif not a_pass:
        branch = "CONTROL-A-FAIL"
        narrative = (
            "Text-only adapter reaches/beats R1.8's MLP-2. The structural "
            "objective acts as optimizer regularization, not acoustic "
            "preservation. Hard pivot: rethink loss design — investigate "
            "whether L_cf actually uses audio."
        )
    elif not b_pass:
        branch = "CONTROL-B-FAIL"
        narrative = (
            "Shuffled-audio adapter reproduces R1.8's gain despite audio "
            "decorrelated from labels. Adapter is using transcript- or "
            "label-side signal that doesn't require correct audio. "
            "Hard pivot: audit which loss term carries the leakage; "
            "L_cf with K_T text branch is prime suspect."
        )
    elif not sc_pass:
        branch = "SHORTCUT-PROBE-FAIL"
        narrative = (
            "Adapter encodes domain or TTS-fingerprint at near-raw-H "
            "levels. The MLP-2 > K_T result may ride nuisance correlations "
            "with Φ. Hard pivot: investigate augmentation pipeline."
        )
    else:
        branch = "PARTIAL-PASS"
        narrative = "Some controls pass, others fail; see gates above."
    md.append(f"\n**Branch: {branch}**\n")
    md.append(f"\n{narrative}")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"saved → {OUT_MD}", flush=True)
    print(f"branch: {branch}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
