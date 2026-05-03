#!/bin/bash
# Stage 4 Control B only: shuffled-audio adapter × 3 seeds.
# Used after Control A completed and the cf_pairs reader was patched
# to ignore extra `_shuffle_*` provenance fields.
set -euo pipefail

cd /home/nurgaly/experiment/Beyond_Transcript_Alignment

VENV=/raid/nurgaly/conda_envs/BTA/bin/python
export CUDA_VISIBLE_DEVICES=6
export HF_HUB_DISABLE_XET=1

mkdir -p outputs/stage4 outputs/stage4_eval

SEEDS=(1234 1235 1236)

for SEED in "${SEEDS[@]}"; do
    echo "======================================================================"
    echo "  CONTROL B — SEED $SEED — TRAIN (shuffled-audio A_R1p8_shuffle)"
    echo "======================================================================"
    $VENV scripts/stage3_train.py \
        --seed "$SEED" \
        --max_steps 400 \
        --grad_accum 32 \
        --warmup_steps 300 \
        --lambda_kl 1.0 \
        --lambda_cf 5.0 \
        --lambda_artifact 1.0 \
        --lambda_cond 0.5 \
        --lambda_nce 0.0 \
        --use_aggressive_aug \
        --log_every 10 \
        --diag_every 50 \
        --cf_pairs_path outputs/stage4/cf_pairs_train_shuffled.jsonl \
        --ckpt_name "A_R1p8_shuffle_seed${SEED}.pt" \
        --out_dir outputs/stage4 \
        2>&1 | tee "training_logs/stage4_train_shuffle_seed${SEED}.log"

    echo "======================================================================"
    echo "  CONTROL B — SEED $SEED — EVAL"
    echo "======================================================================"
    $VENV scripts/stage3_eval.py \
        --seed "$SEED" \
        --checkpoint "outputs/stage4/A_R1p8_shuffle_seed${SEED}.pt" \
        --out_dir outputs/stage4_eval \
        2>&1 | tee "training_logs/stage4_eval_shuffle_seed${SEED}.log"
    mv "outputs/stage4_eval/seed${SEED}" "outputs/stage4_eval/shuffle_seed${SEED}"
done

echo "Control B done."
