#!/bin/bash
# Launch OPT-30B 1-SPSA training on all 6 benchmark tasks with W&B logging
# Config: saturating_alpha=0.1, lambda_reg=1.0, n_perts=40, batch=28, accum=16, lr=1e-4, eps=1e-4, wd=0.001

cd /workspace/1.5-SPSA-LLM
mkdir -p logs checkpoints

# Common configuration
MODEL="facebook/opt-30b"
SOLVER="spsa"
SATURATING_ALPHA="0.1"
LAMBDA_REG="1.0"
N_PERTS="40"
BATCH_SIZE="28"
ACCUM_STEPS="16"
N_ITERATIONS="1000000"
LR="1e-4"
EPSILON="1e-4"
WEIGHT_DECAY="0.001"
SEQ_LEN="256"
EVAL_INTERVAL="50"
WANDB_PROJECT="opt30b-1spsa"

echo "=============================================="
echo "Launching OPT-30B 1-SPSA on 6 benchmark tasks"
echo "=============================================="
echo "Model: $MODEL"
echo "Solver: $SOLVER (1-SPSA without curvature)"
echo "Config: n_perts=$N_PERTS, batch=$BATCH_SIZE, accum=$ACCUM_STEPS"
echo "        lr=$LR, eps=$EPSILON, wd=$WEIGHT_DECAY"
echo "        saturating_alpha=$SATURATING_ALPHA, lambda_reg=$LAMBDA_REG"
echo "W&B Project: $WANDB_PROJECT"
echo "=============================================="

# GPU 0: SST-2
echo "Launching SST-2 on GPU 0..."
screen -dmS opt30b_sst2 bash -c "
    cd /workspace/1.5-SPSA-LLM
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
        --task sst2 \
        --model $MODEL \
        --solver $SOLVER \
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
        --search_strategy none \
        --weight_decay $WEIGHT_DECAY \
        --memory_efficient \
        --wandb \
        --wandb_project $WANDB_PROJECT \
        --wandb_run_name opt30b_sst2_1spsa \
        2>&1 | tee logs/opt30b_sst2_1spsa.log
"
echo "GPU 0: SST-2 launched in screen 'opt30b_sst2'"

# GPU 1: RTE
echo "Launching RTE on GPU 1..."
screen -dmS opt30b_rte bash -c "
    cd /workspace/1.5-SPSA-LLM
    CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
        --task rte \
        --model $MODEL \
        --solver $SOLVER \
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
        --search_strategy none \
        --weight_decay $WEIGHT_DECAY \
        --memory_efficient \
        --wandb \
        --wandb_project $WANDB_PROJECT \
        --wandb_run_name opt30b_rte_1spsa \
        2>&1 | tee logs/opt30b_rte_1spsa.log
"
echo "GPU 1: RTE launched in screen 'opt30b_rte'"

# GPU 2: BoolQ
echo "Launching BoolQ on GPU 2..."
screen -dmS opt30b_boolq bash -c "
    cd /workspace/1.5-SPSA-LLM
    CUDA_VISIBLE_DEVICES=2 python scripts/train.py \
        --task boolq \
        --model $MODEL \
        --solver $SOLVER \
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
        --search_strategy none \
        --weight_decay $WEIGHT_DECAY \
        --memory_efficient \
        --wandb \
        --wandb_project $WANDB_PROJECT \
        --wandb_run_name opt30b_boolq_1spsa \
        2>&1 | tee logs/opt30b_boolq_1spsa.log
"
echo "GPU 2: BoolQ launched in screen 'opt30b_boolq'"

# GPU 3: WSC
echo "Launching WSC on GPU 3..."
screen -dmS opt30b_wsc bash -c "
    cd /workspace/1.5-SPSA-LLM
    CUDA_VISIBLE_DEVICES=3 python scripts/train.py \
        --task wsc \
        --model $MODEL \
        --solver $SOLVER \
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
        --search_strategy none \
        --weight_decay $WEIGHT_DECAY \
        --memory_efficient \
        --wandb \
        --wandb_project $WANDB_PROJECT \
        --wandb_run_name opt30b_wsc_1spsa \
        2>&1 | tee logs/opt30b_wsc_1spsa.log
"
echo "GPU 3: WSC launched in screen 'opt30b_wsc'"

# GPU 4: WiC
echo "Launching WiC on GPU 4..."
screen -dmS opt30b_wic bash -c "
    cd /workspace/1.5-SPSA-LLM
    CUDA_VISIBLE_DEVICES=4 python scripts/train.py \
        --task wic \
        --model $MODEL \
        --solver $SOLVER \
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
        --search_strategy none \
        --weight_decay $WEIGHT_DECAY \
        --memory_efficient \
        --wandb \
        --wandb_project $WANDB_PROJECT \
        --wandb_run_name opt30b_wic_1spsa \
        2>&1 | tee logs/opt30b_wic_1spsa.log
"
echo "GPU 4: WiC launched in screen 'opt30b_wic'"

# GPU 5: SQuAD (longer seq_len for QA)
echo "Launching SQuAD on GPU 5..."
screen -dmS opt30b_squad bash -c "
    cd /workspace/1.5-SPSA-LLM
    CUDA_VISIBLE_DEVICES=5 python scripts/train.py \
        --task squad \
        --model $MODEL \
        --solver $SOLVER \
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
        --search_strategy none \
        --weight_decay $WEIGHT_DECAY \
        --memory_efficient \
        --wandb \
        --wandb_project $WANDB_PROJECT \
        --wandb_run_name opt30b_squad_1spsa \
        2>&1 | tee logs/opt30b_squad_1spsa.log
"
echo "GPU 5: SQuAD launched in screen 'opt30b_squad'"

echo ""
echo "=============================================="
echo "All 6 jobs launched in screen sessions!"
echo "=============================================="
echo ""
echo "View screens: screen -ls"
echo "Attach to a screen: screen -r opt30b_<task>"
echo "  e.g., screen -r opt30b_sst2"
echo ""
echo "Detach from screen: Ctrl+A, then D"
echo ""
echo "W&B Dashboard: https://wandb.ai/fchaubar/$WANDB_PROJECT"
echo ""
