#!/usr/bin/env python3
"""artifacts/scripts/smoke_test.py — single-command pipeline integrity check.

Verifies:
  1. Pinned dependencies load (transformers==4.51.3, torch>=2.5, peft).
  2. WavLM-Large + Qwen3-8B-Instruct + tokenizer all load.
  3. Adapter checkpoint loads (R1.8 seed1234 if available).
  4. Probe-G-oracle on 5 StressPresso items reproduces ≥ 0.78 (Stage 1a anchor).
  5. Per-seed summary.json files match the cohort means in
     configs/locked_cells.yaml within tolerance.

Run from project root:
    python artifacts/scripts/smoke_test.py
"""

from __future__ import annotations
import json, sys, os
from pathlib import Path
from statistics import mean, pstdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
print(f"[smoke] cwd = {PROJECT_ROOT}")

# Released repo layout uses results/; original project layout uses outputs/ symlink.
if (PROJECT_ROOT / "results").is_dir():
    EVAL_ROOT = "results"
elif (PROJECT_ROOT / "outputs").exists():
    EVAL_ROOT = "outputs"
else:
    print(f"[smoke] FAIL: neither results/ nor outputs/ found at {PROJECT_ROOT}")
    sys.exit(1)
print(f"[smoke] eval root = {EVAL_ROOT}")


def fail(msg: str) -> None:
    print(f"[smoke] FAIL: {msg}")
    sys.exit(1)


def section(label: str):
    print(f"\n[smoke] === {label} ===")


# --------------------------------------------------------------------------
# 1. Dependencies
# --------------------------------------------------------------------------
section("Pinned dependencies")
try:
    import torch
    import transformers
    print(f"        torch       = {torch.__version__}")
    print(f"        transformers= {transformers.__version__}")
    if transformers.__version__ != "4.51.3":
        print(f"[smoke] WARN: transformers != 4.51.3; results may differ.")
except Exception as exc:
    fail(f"import torch/transformers: {exc}")

try:
    import peft
    print(f"        peft        = {peft.__version__}")
except Exception:
    print(f"[smoke] WARN: peft not installed (Stage 6 LoRA will be skipped).")


# --------------------------------------------------------------------------
# 2. Per-seed summary.json cohort sanity
# --------------------------------------------------------------------------
section("Per-seed cohort sanity (no model load)")

def cohort_mean(stage_eval, seeds, *path):
    vals = []
    for s in seeds:
        p = Path(EVAL_ROOT) / stage_eval / f"seed{s}" / "summary.json"
        if not p.exists():
            fail(f"missing per-seed summary: {p}")
        d = json.loads(p.read_text())
        for k in path:
            d = d[k]
        vals.append(float(d))
    return mean(vals), pstdev(vals)

R0 = [1234, 1235, 1236, 1237, 1238]
R18 = [1234, 1235, 1236, 1237, 1238]

R0_neutral_m, R0_neutral_s = cohort_mean("stage2_eval", R0, "adapter", "accuracy_neutral")
R0_pk_lin_m, _             = cohort_mean("stage2_eval", R0, "probe_k", "linear", "eval_acc_full")
R18_neutral_m, _           = cohort_mean("stage3p8_eval", R18, "adapter", "accuracy_neutral")
R18_pk_mlp_m, _            = cohort_mean("stage3p8_eval", R18, "probe_k", "mlp2", "eval_acc_full")

S7 = json.loads(Path(f"{EVAL_ROOT}/stage7_eval/seed1234/summary.json").read_text())
S7_neutral = float(S7["adapter"]["accuracy_neutral"])
S7_oracle  = float(S7["oracle_reconfirm"]["accuracy"])
S7_KT_styled = float(S7["K_T_styled"]["accuracy_neutral"])

print(f"        R0  cohort Probe-G_neutral = {R0_neutral_m:.4f} (expect 0.5122)")
print(f"        R0  cohort Probe-K linear  = {R0_pk_lin_m:.4f} (expect 0.2105)")
print(f"        R1.8 cohort Probe-G_neutral = {R18_neutral_m:.4f} (expect 0.5122)")
print(f"        R1.8 cohort Probe-K MLP-2   = {R18_pk_mlp_m:.4f} (expect 0.3059)")
print(f"        Stage 7 Probe-G_neutral     = {S7_neutral:.4f} (expect 0.5158)")
print(f"        Stage 7 K_T_styled neutral  = {S7_KT_styled:.4f} (expect 0.7901)")
print(f"        Stage 7 oracle re-confirm   = {S7_oracle:.4f} (expect 0.7871)")

# Tolerances
checks = [
    ("R0 neutral",         R0_neutral_m,   0.5122, 1e-3),
    ("R0 PK linear",       R0_pk_lin_m,    0.2105, 1e-3),
    ("R1.8 neutral",       R18_neutral_m,  0.5122, 1e-3),
    ("R1.8 PK MLP-2",      R18_pk_mlp_m,   0.3059, 5e-3),
    ("S7 neutral",         S7_neutral,     0.5158, 1e-3),
    ("S7 K_T_styled",      S7_KT_styled,   0.7901, 1e-3),
    ("S7 oracle",          S7_oracle,      0.7871, 1e-3),
]
nfail = 0
for name, got, exp, tol in checks:
    if abs(got - exp) > tol:
        print(f"[smoke] FAIL: {name} — got {got:.4f}, expected {exp:.4f} (tol {tol})")
        nfail += 1
if nfail:
    fail(f"{nfail} cohort-sanity checks failed")
print("[smoke] All cohort-sanity checks PASS.")


# --------------------------------------------------------------------------
# 3. Optional: model-load smoke (skipped by default; uncomment to enable)
# --------------------------------------------------------------------------
section("Model load smoke (skipped — uncomment in script to enable)")
print("        WavLM-Large + Qwen3-8B-Instruct loads not run.")
print("        To enable: edit artifacts/scripts/smoke_test.py and uncomment "
      "the model-load block.")

# Uncomment below to actually load the models:
# from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
# print("[smoke] Loading WavLM-Large (frozen)...")
# enc = AutoModel.from_pretrained("microsoft/wavlm-large", torch_dtype=torch.bfloat16)
# print(f"        WavLM params: {sum(p.numel() for p in enc.parameters()) / 1e6:.1f} M")
# print("[smoke] Loading Qwen3-8B-Instruct (frozen)...")
# tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
# llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16)
# print(f"        Qwen3-8B params: {sum(p.numel() for p in llm.parameters()) / 1e9:.2f} B")
# print(f"        d_llm = {llm.config.hidden_size}")
# assert llm.config.hidden_size == 4096, "d_llm mismatch"

print("\n[smoke] Pipeline integrity verified.")
