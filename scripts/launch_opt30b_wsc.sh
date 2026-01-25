#!/bin/bash
# Launch OPT-30B 1.5-SPSA training on WSC task
# Config: LR=EPS=1e-3, n_perts=160, effective_batch=256

cd /workspace/1.5-SPSA-LLM
mkdir -p logs checkpoints

# OPT-30B is ~60GB in fp16, leaving ~20GB for activations on 80GB A100
# Start with batch_size=2, accum_steps=128 for effective batch 256
# If OOM, reduce batch_size to 1 and accum_steps to 256

BATCH_SIZE=2
ACCUM_STEPS=128
# effective batch = BATCH_SIZE * ACCUM_STEPS = 256

echo "=============================================="
echo "Launching OPT-30B 1.5-SPSA on WSC"
echo "LR=1e-3, EPS=1e-3, n_perts=160"
echo "batch_size=$BATCH_SIZE, accum_steps=$ACCUM_STEPS (effective=256)"
echo "Using GPU 0"
echo "=============================================="

screen -dmS opt30b_wsc bash -c '
    cd /workspace/1.5-SPSA-LLM
    source /workspace/1.5-SPSA-LLM/venv/bin/activate

    echo "Starting OPT-30B 1.5-SPSA on WSC..."
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
        --task wsc \
        --model facebook/opt-30b \
        --solver spsa \
        --use_1_5_spsa \
        --saturating_alpha 0.1 \
        --lambda_reg 1.0 \
        --n_perts 160 \
        --n_iterations 1000 \
        --lr 1e-3 \
        --epsilon 1e-3 \
        --batch_size 2 \
        --accum_steps 128 \
        --seq_len 256 \
        --eval_interval 5 \
        --search_strategy none \
        --memory_efficient \
        2>&1 | tee logs/opt30b_wsc_1.5spsa.log

    echo "Training complete!"
'

echo ""
echo "Job launched in screen session: opt30b_wsc"
echo ""
echo "Monitor with: tail -f logs/opt30b_wsc_1.5spsa.log"
echo "View screens: screen -ls"
echo "Attach: screen -r opt30b_wsc"
echo ""
