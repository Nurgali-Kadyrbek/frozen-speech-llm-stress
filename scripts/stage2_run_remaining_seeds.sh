#!/bin/bash
# Run seeds 1235-1238 sequentially. Each at 300 steps (faster than 600 used
# for seed 1234) — kickoff §2.3 says "1 epoch (~300 steps); 2 epochs only if
# loss has not plateaued". The seed 1234 run at 600 steps establishes that
# baseline; remaining seeds use 300 to fit time budget.
#
# After each seed trains, runs the eval. Output checkpoints + eval summaries
# accumulate in outputs/stage2/ and outputs/stage2_eval/.
set -euo pipefail

cd "$(dirname "$0")/.."

VENV=/raid/nurgaly/conda_envs/BTA/bin/python
export CUDA_VISIBLE_DEVICES=6
export HF_HUB_DISABLE_XET=1

SEEDS=(1235 1236 1237 1238)
MAX_STEPS=300
GRAD_ACCUM=32
WARMUP=250            # scale warmup proportionally with shorter run

for SEED in "${SEEDS[@]}"; do
    echo "======================================================================"
    echo "  SEED $SEED — TRAIN"
    echo "======================================================================"
    $VENV scripts/stage2_train.py \
        --seed "$SEED" \
        --max_steps $MAX_STEPS \
        --grad_accum $GRAD_ACCUM \
        --warmup_steps $WARMUP \
        --n_libri 6000 --n_expr 6000 \
        --log_every 25 \
        2>&1 | tee "training_logs/stage2_train_seed${SEED}.log"

    echo "======================================================================"
    echo "  SEED $SEED — EVAL"
    echo "======================================================================"
    $VENV scripts/stage2_eval.py \
        --seed "$SEED" \
        --checkpoint "outputs/stage2/A_BLSP_seed${SEED}.pt" \
        2>&1 | tee "training_logs/stage2_eval_seed${SEED}.log"
done

echo "All remaining seeds done. Run scripts/stage2_finalize.py to aggregate."
