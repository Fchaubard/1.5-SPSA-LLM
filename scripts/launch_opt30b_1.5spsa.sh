#!/bin/bash
# Launch OPT-30B 1.5-SPSA training on WSC and SQuAD (GPUs 6-7)
# Config: 1.5-SPSA with alpha=0.1, lr=eps=5e-4, batch=56, accum=8

cd /workspace/1.5-SPSA-LLM
mkdir -p logs checkpoints

# Common configuration
MODEL="facebook/opt-30b"
SOLVER="spsa"
SATURATING_ALPHA="0.1"
LAMBDA_REG="1.0"
N_PERTS="40"
BATCH_SIZE="56"
ACCUM_STEPS="8"
N_ITERATIONS="1000000"
LR="5e-4"
EPSILON="5e-4"
WEIGHT_DECAY="0.001"
SEQ_LEN="256"
EVAL_INTERVAL="3"
CHECKPOINT_INTERVAL="50"
WANDB_PROJECT="opt30b-1spsa"

echo "=============================================="
echo "Launching OPT-30B 1.5-SPSA on WSC & SQuAD"
echo "=============================================="
echo "Model: $MODEL"
echo "Solver: $SOLVER (1.5-SPSA WITH curvature, alpha=$SATURATING_ALPHA)"
echo "Config: n_perts=$N_PERTS, batch=$BATCH_SIZE, accum=$ACCUM_STEPS (eff=448)"
echo "        lr=$LR, eps=$EPSILON, wd=$WEIGHT_DECAY"
echo "        eval_interval=$EVAL_INTERVAL, checkpoint_interval=$CHECKPOINT_INTERVAL"
echo "W&B Project: $WANDB_PROJECT"
echo "=============================================="

# GPU 6: WSC with 1.5-SPSA
echo "Launching WSC (1.5-SPSA) on GPU 6..."
screen -dmS opt30b_wsc_1.5spsa bash -c "
    cd /workspace/1.5-SPSA-LLM
    CUDA_VISIBLE_DEVICES=6 python scripts/train.py \
        --task wsc \
        --model $MODEL \
        --solver $SOLVER \
        --use_1_5_spsa \
        --saturating_alpha $SATURATING_ALPHA \
        --lambda_reg $LAMBDA_REG \
        --n_perts $N_PERTS \
        --n_iterations $N_ITERATIONS \
        --lr $LR \
        --epsilon $EPSILON \
        --batch_size $BATCH_SIZE \
        --accum_steps $ACCUM_STEPS \
        --seq_len $SEQ_LEN \
        --eval_interval $EVAL_INTERVAL \
        --checkpoint_interval $CHECKPOINT_INTERVAL \
        --search_strategy none \
        --weight_decay $WEIGHT_DECAY \
        --memory_efficient \
        --wandb \
        --wandb_project $WANDB_PROJECT \
        --wandb_run_name opt30b_wsc_1.5spsa_lr5e4 \
        2>&1 | tee logs/opt30b_wsc_1.5spsa.log
"
echo "GPU 6: WSC (1.5-SPSA) launched in screen 'opt30b_wsc_1.5spsa'"

# GPU 7: SQuAD with 1.5-SPSA
echo "Launching SQuAD (1.5-SPSA) on GPU 7..."
screen -dmS opt30b_squad_1.5spsa bash -c "
    cd /workspace/1.5-SPSA-LLM
    CUDA_VISIBLE_DEVICES=7 python scripts/train.py \
        --task squad \
        --model $MODEL \
        --solver $SOLVER \
        --use_1_5_spsa \
        --saturating_alpha $SATURATING_ALPHA \
        --lambda_reg $LAMBDA_REG \
        --n_perts $N_PERTS \
        --n_iterations $N_ITERATIONS \
        --lr $LR \
        --epsilon $EPSILON \
        --batch_size $BATCH_SIZE \
        --accum_steps $ACCUM_STEPS \
        --seq_len 512 \
        --eval_interval $EVAL_INTERVAL \
        --checkpoint_interval $CHECKPOINT_INTERVAL \
        --search_strategy none \
        --weight_decay $WEIGHT_DECAY \
        --memory_efficient \
        --wandb \
        --wandb_project $WANDB_PROJECT \
        --wandb_run_name opt30b_squad_1.5spsa_lr5e4 \
        2>&1 | tee logs/opt30b_squad_1.5spsa.log
"
echo "GPU 7: SQuAD (1.5-SPSA) launched in screen 'opt30b_squad_1.5spsa'"

echo ""
echo "=============================================="
echo "1.5-SPSA jobs launched!"
echo "=============================================="
echo ""
echo "View screens: screen -ls"
echo "Attach: screen -r opt30b_wsc_1.5spsa"
echo "        screen -r opt30b_squad_1.5spsa"
echo ""
echo "Monitor:"
echo "  tail -f logs/opt30b_wsc_1.5spsa.log"
echo "  tail -f logs/opt30b_squad_1.5spsa.log"
echo ""
