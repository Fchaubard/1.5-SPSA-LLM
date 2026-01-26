#!/bin/bash
# Launch all OPT-30B runs with eval_interval=3
# 1-SPSA: GPUs 0-5 (SST2, RTE, BoolQ, WSC, WiC, SQuAD)
# 1.5-SPSA: GPUs 6-7 (WSC, SQuAD)
# SQuAD runs use lr=5e-5, others use standard LR

cd /workspace/1.5-SPSA-LLM
mkdir -p logs checkpoints

MODEL="facebook/opt-30b"
N_PERTS="40"
N_ITERATIONS="1000000"
WEIGHT_DECAY="0.001"
SEQ_LEN="256"
EVAL_INTERVAL="3"
CHECKPOINT_INTERVAL="50"
WANDB_PROJECT="opt30b-1spsa"

echo "═══════════════════════════════════════════════════════════════════════════"
echo "  Launching ALL OPT-30B Runs (eval_interval=3)"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "  1-SPSA (lr=1e-4, eps=1e-4):"
echo "    GPU 0: SST-2    GPU 1: RTE     GPU 2: BoolQ"
echo "    GPU 3: WSC      GPU 4: WiC    GPU 5: SQuAD (lr=5e-5)"
echo ""
echo "  1.5-SPSA (lr=5e-4, eps=5e-4):"
echo "    GPU 6: WSC (batch=448)    GPU 7: SQuAD (lr=5e-5, batch=112)"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"

# ============================================================================
# 1-SPSA RUNS (GPUs 0-5)
# ============================================================================

# GPU 0: SST-2 1-SPSA
screen -dmS opt30b_sst2 bash -c "
    cd /workspace/1.5-SPSA-LLM
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
        --task sst2 \
        --model $MODEL \
        --solver spsa \
        --saturating_alpha 0.1 \
        --lambda_reg 1.0 \
        --n_perts $N_PERTS \
        --n_iterations $N_ITERATIONS \
        --lr 1e-4 \
        --epsilon 1e-4 \
        --batch_size 28 \
        --accum_steps 16 \
        --seq_len $SEQ_LEN \
        --eval_interval $EVAL_INTERVAL \
        --checkpoint_interval $CHECKPOINT_INTERVAL \
        --search_strategy none \
        --weight_decay $WEIGHT_DECAY \
        --memory_efficient \
        --wandb \
        --wandb_project $WANDB_PROJECT \
        --wandb_run_name sst2_1spsa_v2 \
        2>&1 | tee logs/opt30b_sst2_1spsa.log
"
echo "GPU 0: SST-2 1-SPSA launched"

# GPU 1: RTE 1-SPSA
screen -dmS opt30b_rte bash -c "
    cd /workspace/1.5-SPSA-LLM
    CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
        --task rte \
        --model $MODEL \
        --solver spsa \
        --saturating_alpha 0.1 \
        --lambda_reg 1.0 \
        --n_perts $N_PERTS \
        --n_iterations $N_ITERATIONS \
        --lr 1e-4 \
        --epsilon 1e-4 \
        --batch_size 28 \
        --accum_steps 16 \
        --seq_len $SEQ_LEN \
        --eval_interval $EVAL_INTERVAL \
        --checkpoint_interval $CHECKPOINT_INTERVAL \
        --search_strategy none \
        --weight_decay $WEIGHT_DECAY \
        --memory_efficient \
        --wandb \
        --wandb_project $WANDB_PROJECT \
        --wandb_run_name rte_1spsa_v2 \
        2>&1 | tee logs/opt30b_rte_1spsa.log
"
echo "GPU 1: RTE 1-SPSA launched"

# GPU 2: BoolQ 1-SPSA
screen -dmS opt30b_boolq bash -c "
    cd /workspace/1.5-SPSA-LLM
    CUDA_VISIBLE_DEVICES=2 python scripts/train.py \
        --task boolq \
        --model $MODEL \
        --solver spsa \
        --saturating_alpha 0.1 \
        --lambda_reg 1.0 \
        --n_perts $N_PERTS \
        --n_iterations $N_ITERATIONS \
        --lr 1e-4 \
        --epsilon 1e-4 \
        --batch_size 28 \
        --accum_steps 16 \
        --seq_len $SEQ_LEN \
        --eval_interval $EVAL_INTERVAL \
        --checkpoint_interval $CHECKPOINT_INTERVAL \
        --search_strategy none \
        --weight_decay $WEIGHT_DECAY \
        --memory_efficient \
        --wandb \
        --wandb_project $WANDB_PROJECT \
        --wandb_run_name boolq_1spsa_v2 \
        2>&1 | tee logs/opt30b_boolq_1spsa.log
"
echo "GPU 2: BoolQ 1-SPSA launched"

# GPU 3: WSC 1-SPSA
screen -dmS opt30b_wsc bash -c "
    cd /workspace/1.5-SPSA-LLM
    CUDA_VISIBLE_DEVICES=3 python scripts/train.py \
        --task wsc \
        --model $MODEL \
        --solver spsa \
        --saturating_alpha 0.1 \
        --lambda_reg 1.0 \
        --n_perts $N_PERTS \
        --n_iterations $N_ITERATIONS \
        --lr 1e-4 \
        --epsilon 1e-4 \
        --batch_size 28 \
        --accum_steps 16 \
        --seq_len $SEQ_LEN \
        --eval_interval $EVAL_INTERVAL \
        --checkpoint_interval $CHECKPOINT_INTERVAL \
        --search_strategy none \
        --weight_decay $WEIGHT_DECAY \
        --memory_efficient \
        --wandb \
        --wandb_project $WANDB_PROJECT \
        --wandb_run_name wsc_1spsa_v2 \
        2>&1 | tee logs/opt30b_wsc_1spsa.log
"
echo "GPU 3: WSC 1-SPSA launched"

# GPU 4: WiC 1-SPSA
screen -dmS opt30b_wic bash -c "
    cd /workspace/1.5-SPSA-LLM
    CUDA_VISIBLE_DEVICES=4 python scripts/train.py \
        --task wic \
        --model $MODEL \
        --solver spsa \
        --saturating_alpha 0.1 \
        --lambda_reg 1.0 \
        --n_perts $N_PERTS \
        --n_iterations $N_ITERATIONS \
        --lr 1e-4 \
        --epsilon 1e-4 \
        --batch_size 28 \
        --accum_steps 16 \
        --seq_len $SEQ_LEN \
        --eval_interval $EVAL_INTERVAL \
        --checkpoint_interval $CHECKPOINT_INTERVAL \
        --search_strategy none \
        --weight_decay $WEIGHT_DECAY \
        --memory_efficient \
        --wandb \
        --wandb_project $WANDB_PROJECT \
        --wandb_run_name wic_1spsa_v2 \
        2>&1 | tee logs/opt30b_wic_1spsa.log
"
echo "GPU 4: WiC 1-SPSA launched"

# GPU 5: SQuAD 1-SPSA (lr=5e-5)
screen -dmS opt30b_squad bash -c "
    cd /workspace/1.5-SPSA-LLM
    CUDA_VISIBLE_DEVICES=5 python scripts/train.py \
        --task squad \
        --model $MODEL \
        --solver spsa \
        --saturating_alpha 0.1 \
        --lambda_reg 1.0 \
        --n_perts $N_PERTS \
        --n_iterations $N_ITERATIONS \
        --lr 5e-5 \
        --epsilon 5e-5 \
        --batch_size 28 \
        --accum_steps 16 \
        --seq_len 512 \
        --eval_interval $EVAL_INTERVAL \
        --checkpoint_interval $CHECKPOINT_INTERVAL \
        --search_strategy none \
        --weight_decay $WEIGHT_DECAY \
        --memory_efficient \
        --wandb \
        --wandb_project $WANDB_PROJECT \
        --wandb_run_name squad_1spsa_v2_lr5e5 \
        2>&1 | tee logs/opt30b_squad_1spsa.log
"
echo "GPU 5: SQuAD 1-SPSA (lr=5e-5) launched"

# ============================================================================
# 1.5-SPSA RUNS (GPUs 6-7)
# ============================================================================

# GPU 6: WSC 1.5-SPSA (batch=448, accum=1)
screen -dmS opt30b_wsc_1.5spsa bash -c "
    cd /workspace/1.5-SPSA-LLM
    CUDA_VISIBLE_DEVICES=6 python scripts/train.py \
        --task wsc \
        --model $MODEL \
        --solver spsa \
        --use_1_5_spsa \
        --saturating_alpha 0.1 \
        --lambda_reg 1.0 \
        --n_perts $N_PERTS \
        --n_iterations $N_ITERATIONS \
        --lr 5e-4 \
        --epsilon 5e-4 \
        --batch_size 448 \
        --accum_steps 1 \
        --seq_len $SEQ_LEN \
        --eval_interval $EVAL_INTERVAL \
        --checkpoint_interval $CHECKPOINT_INTERVAL \
        --search_strategy none \
        --weight_decay $WEIGHT_DECAY \
        --memory_efficient \
        --wandb \
        --wandb_project $WANDB_PROJECT \
        --wandb_run_name wsc_1.5spsa_v2_maxbatch \
        2>&1 | tee logs/opt30b_wsc_1.5spsa.log
"
echo "GPU 6: WSC 1.5-SPSA (batch=448) launched"

# GPU 7: SQuAD 1.5-SPSA (lr=5e-5, batch=112, accum=4)
screen -dmS opt30b_squad_1.5spsa bash -c "
    cd /workspace/1.5-SPSA-LLM
    CUDA_VISIBLE_DEVICES=7 python scripts/train.py \
        --task squad \
        --model $MODEL \
        --solver spsa \
        --use_1_5_spsa \
        --saturating_alpha 0.1 \
        --lambda_reg 1.0 \
        --n_perts $N_PERTS \
        --n_iterations $N_ITERATIONS \
        --lr 5e-5 \
        --epsilon 5e-5 \
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
        --wandb_run_name squad_1.5spsa_v2_lr5e5 \
        2>&1 | tee logs/opt30b_squad_1.5spsa.log
"
echo "GPU 7: SQuAD 1.5-SPSA (lr=5e-5, batch=112) launched"

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  All 8 runs launched!"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "  screen -ls                    # List all screens"
echo "  screen -r opt30b_<task>       # Attach to a screen"
echo "  tail -f logs/opt30b_*.log     # Monitor logs"
echo ""
