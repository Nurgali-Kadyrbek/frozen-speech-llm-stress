#!/bin/bash
# Stage 4 controls orchestrator: Control A (text-only) × 3 + Control B (shuffled) × 3.
# Same hyperparameters as Stage 3.8 cohort (λ_cf=5, λ_NCE=0).
set -euo pipefail

cd /home/nurgaly/experiment/Beyond_Transcript_Alignment

VENV=/raid/nurgaly/conda_envs/BTA/bin/python
export CUDA_VISIBLE_DEVICES=6
export HF_HUB_DISABLE_XET=1

mkdir -p outputs/stage4 outputs/stage4_eval

SEEDS=(1234 1235 1236)
MAX_STEPS=400
GRAD_ACCUM=32
WARMUP=300

for SEED in "${SEEDS[@]}"; do
    echo "======================================================================"
    echo "  CONTROL A — SEED $SEED — TRAIN (text-only A_textK)"
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
        --control_mode text_only \
        --proj_P_path outputs/stage4/proj_P.pt \
        --ckpt_name "A_textK_seed${SEED}.pt" \
        --out_dir outputs/stage4 \
        2>&1 | tee "training_logs/stage4_train_textK_seed${SEED}.log"

    echo "======================================================================"
    echo "  CONTROL A — SEED $SEED — EVAL (text-only)"
    echo "======================================================================"
    $VENV scripts/stage3_eval.py \
        --seed "$SEED" \
        --checkpoint "outputs/stage4/A_textK_seed${SEED}.pt" \
        --control_mode text_only \
        --proj_P_path outputs/stage4/proj_P.pt \
        --out_dir outputs/stage4_eval \
        2>&1 | tee "training_logs/stage4_eval_textK_seed${SEED}.log"

    # Move/rename the eval bundle so Control A and Control B don't collide
    # (same seed numbers used for both controls).
    mv "outputs/stage4_eval/seed${SEED}" "outputs/stage4_eval/textK_seed${SEED}"
done

for SEED in "${SEEDS[@]}"; do
    echo "======================================================================"
    echo "  CONTROL B — SEED $SEED — TRAIN (shuffled-audio A_R1p8_shuffle)"
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

echo "All controls done. Run scripts/stage4_finalize.py to aggregate."
