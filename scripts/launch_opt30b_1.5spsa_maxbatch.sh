#!/bin/bash
# Launch OPT-30B 1.5-SPSA with MAXED batch sizes (GPUs 6-7)
# WSC: batch=448, accum=1 (no accumulation needed!)
# SQuAD: batch=112, accum=4

cd /workspace/1.5-SPSA-LLM
mkdir -p logs checkpoints

# Common configuration
MODEL="facebook/opt-30b"
SOLVER="spsa"
SATURATING_ALPHA="0.1"
LAMBDA_REG="1.0"
N_PERTS="40"
N_ITERATIONS="1000000"
LR="5e-4"
EPSILON="5e-4"
WEIGHT_DECAY="0.001"
EVAL_INTERVAL="3"
CHECKPOINT_INTERVAL="50"
WANDB_PROJECT="opt30b-1spsa"

echo "=============================================="
echo "Launching OPT-30B 1.5-SPSA with MAX BATCH"
echo "=============================================="
echo "Model: $MODEL"
echo "Solver: 1.5-SPSA (alpha=$SATURATING_ALPHA)"
echo ""
echo "WSC:   batch=448, accum=1  (eff=448) - NO ACCUMULATION!"
echo "SQuAD: batch=112, accum=4  (eff=448)"
echo ""
echo "lr=$LR, eps=$EPSILON, eval_interval=$EVAL_INTERVAL"
echo "=============================================="

# GPU 6: WSC with 1.5-SPSA - MAXED batch=448, accum=1
echo "Launching WSC (1.5-SPSA, batch=448) on GPU 6..."
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
        --batch_size 448 \
        --accum_steps 1 \
        --seq_len 256 \
        --eval_interval $EVAL_INTERVAL \
        --checkpoint_interval $CHECKPOINT_INTERVAL \
        --search_strategy none \
        --weight_decay $WEIGHT_DECAY \
        --memory_efficient \
        --wandb \
        --wandb_project $WANDB_PROJECT \
        --wandb_run_name opt30b_wsc_1.5spsa_maxbatch \
        2>&1 | tee logs/opt30b_wsc_1.5spsa.log
"
echo "GPU 6: WSC launched (batch=448, accum=1)"

# GPU 7: SQuAD with 1.5-SPSA - batch=112, accum=4
echo "Launching SQuAD (1.5-SPSA, batch=112) on GPU 7..."
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
        --batch_size 112 \
        --accum_steps 4 \
        --seq_len 512 \
        --eval_interval $EVAL_INTERVAL \
        --checkpoint_interval $CHECKPOINT_INTERVAL \
        --search_strategy none \
        --weight_decay $WEIGHT_DECAY \
        --memory_efficient \
        --wandb \
        --wandb_project $WANDB_PROJECT \
        --wandb_run_name opt30b_squad_1.5spsa_maxbatch \
        2>&1 | tee logs/opt30b_squad_1.5spsa.log
"
echo "GPU 7: SQuAD launched (batch=112, accum=4)"

echo ""
echo "=============================================="
echo "1.5-SPSA MAX BATCH jobs launched!"
echo "=============================================="
echo ""
echo "Monitor:"
echo "  tail -f logs/opt30b_wsc_1.5spsa.log"
echo "  tail -f logs/opt30b_squad_1.5spsa.log"
echo ""
