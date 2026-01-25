#!/bin/bash
# Launch OPT-13B training on WIC and SQuAD with 1-SPSA and 1.5-SPSA
# Config: LR=EPS=1e-3, n_perts=160, effective_batch=256
# Uses GPUs 1-4 (GPU 0 reserved for OPT-30B run)

cd /workspace/1.5-SPSA-LLM
mkdir -p logs checkpoints

echo "=============================================="
echo "Launching 4 training jobs on GPUs 1-4"
echo "OPT-13B | LR=EPS=1e-3 | n_perts=160 | eff_batch=256"
echo "=============================================="

# GPU 1: WIC + 1-SPSA (no curvature scaling)
screen -dmS gpu1_wic_1spsa bash -c '
    cd /workspace/1.5-SPSA-LLM
    source /workspace/1.5-SPSA-LLM/venv/bin/activate

    echo "Starting: WIC + 1-SPSA on GPU 1"
    CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
        --task wic \
        --model facebook/opt-13b \
        --solver spsa \
        --n_perts 160 \
        --n_iterations 1000 \
        --lr 1e-3 \
        --epsilon 1e-3 \
        --batch_size 4 \
        --accum_steps 64 \
        --seq_len 256 \
        --eval_interval 10 \
        --search_strategy none \
        --memory_efficient \
        2>&1 | tee logs/opt13b_wic_1spsa.log

    echo "Training complete!"
'
echo "GPU 1: WIC + 1-SPSA"

# GPU 2: WIC + 1.5-SPSA (with curvature scaling)
screen -dmS gpu2_wic_15spsa bash -c '
    cd /workspace/1.5-SPSA-LLM
    source /workspace/1.5-SPSA-LLM/venv/bin/activate

    echo "Starting: WIC + 1.5-SPSA on GPU 2"
    CUDA_VISIBLE_DEVICES=2 python scripts/train.py \
        --task wic \
        --model facebook/opt-13b \
        --solver spsa \
        --use_1_5_spsa \
        --saturating_alpha 0.1 \
        --lambda_reg 1.0 \
        --n_perts 160 \
        --n_iterations 1000 \
        --lr 1e-3 \
        --epsilon 1e-3 \
        --batch_size 4 \
        --accum_steps 64 \
        --seq_len 256 \
        --eval_interval 10 \
        --search_strategy none \
        --memory_efficient \
        2>&1 | tee logs/opt13b_wic_15spsa.log

    echo "Training complete!"
'
echo "GPU 2: WIC + 1.5-SPSA"

# GPU 3: SQuAD + 1-SPSA (no curvature scaling)
# Note: seq_len=2048 for SQuAD as per MeZO paper
screen -dmS gpu3_squad_1spsa bash -c '
    cd /workspace/1.5-SPSA-LLM
    source /workspace/1.5-SPSA-LLM/venv/bin/activate

    echo "Starting: SQuAD + 1-SPSA on GPU 3"
    CUDA_VISIBLE_DEVICES=3 python scripts/train.py \
        --task squad \
        --model facebook/opt-13b \
        --solver spsa \
        --n_perts 160 \
        --n_iterations 1000 \
        --lr 1e-3 \
        --epsilon 1e-3 \
        --batch_size 2 \
        --accum_steps 128 \
        --seq_len 2048 \
        --eval_interval 10 \
        --search_strategy none \
        --memory_efficient \
        2>&1 | tee logs/opt13b_squad_1spsa.log

    echo "Training complete!"
'
echo "GPU 3: SQuAD + 1-SPSA (seq_len=2048)"

# GPU 4: SQuAD + 1.5-SPSA (with curvature scaling)
# Note: seq_len=2048 for SQuAD as per MeZO paper
screen -dmS gpu4_squad_15spsa bash -c '
    cd /workspace/1.5-SPSA-LLM
    source /workspace/1.5-SPSA-LLM/venv/bin/activate

    echo "Starting: SQuAD + 1.5-SPSA on GPU 4"
    CUDA_VISIBLE_DEVICES=4 python scripts/train.py \
        --task squad \
        --model facebook/opt-13b \
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
        --seq_len 2048 \
        --eval_interval 10 \
        --search_strategy none \
        --memory_efficient \
        2>&1 | tee logs/opt13b_squad_15spsa.log

    echo "Training complete!"
'
echo "GPU 4: SQuAD + 1.5-SPSA (seq_len=2048)"

echo ""
echo "=============================================="
echo "All 4 jobs launched!"
echo "=============================================="
echo ""
echo "Monitor with:"
echo "  tail -f logs/opt13b_wic_1spsa.log"
echo "  tail -f logs/opt13b_wic_15spsa.log"
echo "  tail -f logs/opt13b_squad_1spsa.log"
echo "  tail -f logs/opt13b_squad_15spsa.log"
echo ""
echo "View screens: screen -ls"
echo "Attach: screen -r <name>"
echo ""
