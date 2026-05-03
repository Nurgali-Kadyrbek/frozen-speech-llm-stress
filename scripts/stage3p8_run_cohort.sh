#!/bin/bash
# Stage 3.8 cohort orchestrator: train+eval seeds 1235-1238 serially.
# Single-axis change from Stage 3.7: λ_NCE 3.0 → 0.0. All else bit-identical.
# Per kickoff §3.8.0 LOCKED.
set -euo pipefail

cd /home/nurgaly/experiment/Beyond_Transcript_Alignment

VENV=/raid/nurgaly/conda_envs/BTA/bin/python
export CUDA_VISIBLE_DEVICES=6
export HF_HUB_DISABLE_XET=1

mkdir -p outputs/stage3p8 outputs/stage3p8_eval

# Seed 1234 carry-over from Stage 3.7. Document explicit rationale.
echo "======================================================================"
echo "  SEED 1234 — REUSE Stage 3.7 checkpoint (kickoff §3.8.2 carry-over)"
echo "======================================================================"
echo "  rationale: cos-contrib log in Stage 3.7 showed L_NCE alignment 0.05-0.16"
echo "  with total update direction; λ_NCE=3 contributed essentially zero to"
echo "  the actual update. Re-running with λ_NCE=0 would produce a near-"
echo "  identical adapter at 3.7 H wallclock cost. Reuse instead."
mkdir -p outputs/stage3p8_eval/seed1234
cp -f outputs/stage3p7_eval/seed1234/summary.json outputs/stage3p8_eval/seed1234/
cp -f outputs/stage3p7_eval/seed1234/rows.json    outputs/stage3p8_eval/seed1234/
echo "  copied: outputs/stage3p7_eval/seed1234/* → outputs/stage3p8_eval/seed1234/"
ln -sf "$(pwd)/outputs/stage3p7/A_R1p7_seed1234.pt" outputs/stage3p8/A_R1p8_seed1234.pt
echo "  symlinked: outputs/stage3p7/A_R1p7_seed1234.pt → outputs/stage3p8/A_R1p8_seed1234.pt"
echo

SEEDS=(1235 1236 1237 1238)
MAX_STEPS=400
GRAD_ACCUM=32
WARMUP=300

for SEED in "${SEEDS[@]}"; do
    echo "======================================================================"
    echo "  SEED $SEED — TRAIN (R1.8: λ_cf=5, λ_NCE=0)"
    echo "======================================================================"
    $VENV scripts/stage3_train.py \
        --seed "$SEED" \
        --max_steps $MAX_STEPS \
        --grad_accum $GRAD_ACCUM \
        --warmup_steps $WARMUP \
        --lambda_kl 1.0 \
        --lambda_cf 5.0 \
        --lambda_artifact 1.0 \
        --lambda_cond 0.5 \
        --lambda_nce 0.0 \
        --use_aggressive_aug \
        --log_every 10 \
        --diag_every 50 \
        --ckpt_name "A_R1p8_seed${SEED}.pt" \
        --out_dir outputs/stage3p8 \
        2>&1 | tee "training_logs/stage3p8_train_seed${SEED}.log"

    echo "======================================================================"
    echo "  SEED $SEED — EVAL (R1.8: full Branch-D + B3 + T-only)"
    echo "======================================================================"
    $VENV scripts/stage3_eval.py \
        --seed "$SEED" \
        --checkpoint "outputs/stage3p8/A_R1p8_seed${SEED}.pt" \
        --out_dir outputs/stage3p8_eval \
        2>&1 | tee "training_logs/stage3p8_eval_seed${SEED}.log"
done

echo
echo "All cohort seeds done. Run scripts/stage3p8_finalize.py to aggregate."
