#!/bin/bash
# Run seeds 1235-1238 sequentially with FROZEN hyperparameters from pilot.
# Per kickoff §3.3: "Do NOT retune hyperparameters or loss weights after the
# pilot and pretend the cohort is 'confirmatory' — that's effectively n=1
# masquerading as n=5."
#
# Each seed: 400 steps train + full eval + 4-diagnostic Branch-D shortcut probes.
set -euo pipefail

cd "$(dirname "$0")/.."

VENV=/raid/nurgaly/conda_envs/BTA/bin/python
export CUDA_VISIBLE_DEVICES=6
export HF_HUB_DISABLE_XET=1

SEEDS=(1235 1236 1237 1238)
MAX_STEPS=400
GRAD_ACCUM=32
WARMUP=300

for SEED in "${SEEDS[@]}"; do
    echo "======================================================================"
    echo "  SEED $SEED — TRAIN (R1)"
    echo "======================================================================"
    $VENV scripts/stage3_train.py \
        --seed "$SEED" \
        --max_steps $MAX_STEPS \
        --grad_accum $GRAD_ACCUM \
        --warmup_steps $WARMUP \
        --use_aggressive_aug \
        --n_libri 6000 --n_expr 6000 \
        --log_every 10 \
        2>&1 | tee "training_logs/stage3_train_seed${SEED}.log"

    echo "======================================================================"
    echo "  SEED $SEED — EVAL (R1, with C.1-C.4 + T-only)"
    echo "======================================================================"
    $VENV scripts/stage3_eval.py \
        --seed "$SEED" \
        --checkpoint "outputs/stage3/A_R1_seed${SEED}.pt" \
        2>&1 | tee "training_logs/stage3_eval_seed${SEED}.log"
done

echo "All cohort seeds done. Run scripts/stage3_finalize.py to aggregate."
