#!/bin/bash
# ==============================================================================
# Round 2: Beat MeZO on SQuAD and WIC with 1.5-SPSA
# 4 SQuAD runs (GPUs 0-3) + 3 WIC runs (GPUs 4-6), leaving GPU 7 free
# ==============================================================================

# Kill existing round2 screen sessions
screen -ls | grep 'round2' | awk '{print $1}' | xargs -I{} screen -S {} -X quit 2>/dev/null

cd /home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
mkdir -p logs checkpoints

# ==============================================================================
# CONFIGURATION
# ==============================================================================

MODEL_PATH="facebook/opt-30b"
N_PERTS=40
N_ITERATIONS=100
EVAL_INTERVAL=5
CHECKPOINT_INTERVAL=50
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# W&B config
WANDB_PROJECT="spsa-mezo-benchmarks"
WANDB_API_KEY=""

echo "=============================================="
echo "Round 2: SQuAD + WIC Hyperparameter Sweep"
echo "=============================================="
echo ""
echo "SQuAD runs (GPUs 0-3):"
echo "  GPU 0: lr=5e-5 (lower LR)"
echo "  GPU 1: batch=48 (larger batch)"
echo "  GPU 2: alpha=0.3 (less dampening)"
echo "  GPU 3: alpha=0.3, batch=48, lr=5e-5 (combined)"
echo ""
echo "WIC runs (GPUs 4-6):"
echo "  GPU 4: alpha=0.5, lambda_reg=0.1, wd=0 (aggressive)"
echo "  GPU 5: lr=2e-4, wd=0 (higher LR)"
echo "  GPU 6: alpha=0.3, lr=5e-5 (moderate)"
echo ""
echo "=============================================="

# ==============================================================================
# SQuAD runs (GPUs 0-3) - targeting 85.2 F1 (currently 60.2)
# ==============================================================================

# GPU 0: Lower LR for multi-token generative loss
SESSION_NAME="round2_squad_lr5e-5"
echo "Launching GPU 0: squad - lr=5e-5"
screen -dmS "$SESSION_NAME" bash -c "
    cd /home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
    source /mnt/data/venv/bin/activate
    export WANDB_API_KEY='$WANDB_API_KEY'
    export PYTHONPATH=/mnt/data/python_packages:/home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
    export HF_HOME=/mnt/data/huggingface
    export HF_DATASETS_CACHE=/mnt/data/huggingface/datasets
    export TRANSFORMERS_CACHE=/mnt/data/huggingface/transformers
    export TMPDIR=/mnt/data/tmp
    export TRITON_CACHE_DIR=/mnt/data/triton_cache
    mkdir -p /mnt/data/tmp /mnt/data/triton_cache

    echo '=============================================='
    echo 'Round 2: squad - lr=5e-5 on GPU 0'
    echo '=============================================='

    CUDA_VISIBLE_DEVICES=0 python scripts/train.py \\
        --task squad \\
        --model $MODEL_PATH \\
        --solver spsa --use_1_5_spsa \\
        --saturating_alpha 0.1 \\
        --lambda_reg 1.0 \\
        --n_perts $N_PERTS \\
        --batch_size 28 \\
        --n_iterations $N_ITERATIONS \\
        --lr 5e-5 \\
        --epsilon 5e-5 \\
        --weight_decay 0.001 \\
        --seq_len 512 \\
        --eval_interval $EVAL_INTERVAL \\
        --checkpoint_interval $CHECKPOINT_INTERVAL \\
        --memory_efficient \\
        --wandb --wandb_project $WANDB_PROJECT \\
        --wandb_run_name 'round2_squad_lr5e-5' \\
        2>&1 | tee logs/round2_squad_lr5e-5_${TIMESTAMP}.log

    echo 'squad - lr5e-5 completed!'
"

# GPU 1: Larger batch for more stable loss estimates
SESSION_NAME="round2_squad_bs48"
echo "Launching GPU 1: squad - batch=48"
screen -dmS "$SESSION_NAME" bash -c "
    cd /home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
    source /mnt/data/venv/bin/activate
    export WANDB_API_KEY='$WANDB_API_KEY'
    export PYTHONPATH=/mnt/data/python_packages:/home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
    export HF_HOME=/mnt/data/huggingface
    export HF_DATASETS_CACHE=/mnt/data/huggingface/datasets
    export TRANSFORMERS_CACHE=/mnt/data/huggingface/transformers
    export TMPDIR=/mnt/data/tmp
    export TRITON_CACHE_DIR=/mnt/data/triton_cache
    mkdir -p /mnt/data/tmp /mnt/data/triton_cache

    echo '=============================================='
    echo 'Round 2: squad - batch=48 on GPU 1'
    echo '=============================================='

    CUDA_VISIBLE_DEVICES=1 python scripts/train.py \\
        --task squad \\
        --model $MODEL_PATH \\
        --solver spsa --use_1_5_spsa \\
        --saturating_alpha 0.1 \\
        --lambda_reg 1.0 \\
        --n_perts $N_PERTS \\
        --batch_size 48 \\
        --n_iterations $N_ITERATIONS \\
        --lr 1e-4 \\
        --epsilon 1e-4 \\
        --weight_decay 0.001 \\
        --seq_len 512 \\
        --eval_interval $EVAL_INTERVAL \\
        --checkpoint_interval $CHECKPOINT_INTERVAL \\
        --memory_efficient \\
        --wandb --wandb_project $WANDB_PROJECT \\
        --wandb_run_name 'round2_squad_bs48' \\
        2>&1 | tee logs/round2_squad_bs48_${TIMESTAMP}.log

    echo 'squad - bs48 completed!'
"

# GPU 2: Less curvature dampening
SESSION_NAME="round2_squad_alpha0.3"
echo "Launching GPU 2: squad - alpha=0.3"
screen -dmS "$SESSION_NAME" bash -c "
    cd /home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
    source /mnt/data/venv/bin/activate
    export WANDB_API_KEY='$WANDB_API_KEY'
    export PYTHONPATH=/mnt/data/python_packages:/home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
    export HF_HOME=/mnt/data/huggingface
    export HF_DATASETS_CACHE=/mnt/data/huggingface/datasets
    export TRANSFORMERS_CACHE=/mnt/data/huggingface/transformers
    export TMPDIR=/mnt/data/tmp
    export TRITON_CACHE_DIR=/mnt/data/triton_cache
    mkdir -p /mnt/data/tmp /mnt/data/triton_cache

    echo '=============================================='
    echo 'Round 2: squad - alpha=0.3 on GPU 2'
    echo '=============================================='

    CUDA_VISIBLE_DEVICES=2 python scripts/train.py \\
        --task squad \\
        --model $MODEL_PATH \\
        --solver spsa --use_1_5_spsa \\
        --saturating_alpha 0.3 \\
        --lambda_reg 1.0 \\
        --n_perts $N_PERTS \\
        --batch_size 28 \\
        --n_iterations $N_ITERATIONS \\
        --lr 1e-4 \\
        --epsilon 1e-4 \\
        --weight_decay 0.001 \\
        --seq_len 512 \\
        --eval_interval $EVAL_INTERVAL \\
        --checkpoint_interval $CHECKPOINT_INTERVAL \\
        --memory_efficient \\
        --wandb --wandb_project $WANDB_PROJECT \\
        --wandb_run_name 'round2_squad_alpha0.3' \\
        2>&1 | tee logs/round2_squad_alpha0.3_${TIMESTAMP}.log

    echo 'squad - alpha0.3 completed!'
"

# GPU 3: Combined best guesses
SESSION_NAME="round2_squad_combined"
echo "Launching GPU 3: squad - combined (lr=5e-5, alpha=0.3, batch=48)"
screen -dmS "$SESSION_NAME" bash -c "
    cd /home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
    source /mnt/data/venv/bin/activate
    export WANDB_API_KEY='$WANDB_API_KEY'
    export PYTHONPATH=/mnt/data/python_packages:/home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
    export HF_HOME=/mnt/data/huggingface
    export HF_DATASETS_CACHE=/mnt/data/huggingface/datasets
    export TRANSFORMERS_CACHE=/mnt/data/huggingface/transformers
    export TMPDIR=/mnt/data/tmp
    export TRITON_CACHE_DIR=/mnt/data/triton_cache
    mkdir -p /mnt/data/tmp /mnt/data/triton_cache

    echo '=============================================='
    echo 'Round 2: squad - combined on GPU 3'
    echo '=============================================='

    CUDA_VISIBLE_DEVICES=3 python scripts/train.py \\
        --task squad \\
        --model $MODEL_PATH \\
        --solver spsa --use_1_5_spsa \\
        --saturating_alpha 0.3 \\
        --lambda_reg 1.0 \\
        --n_perts $N_PERTS \\
        --batch_size 48 \\
        --n_iterations $N_ITERATIONS \\
        --lr 5e-5 \\
        --epsilon 5e-5 \\
        --weight_decay 0.001 \\
        --seq_len 512 \\
        --eval_interval $EVAL_INTERVAL \\
        --checkpoint_interval $CHECKPOINT_INTERVAL \\
        --memory_efficient \\
        --wandb --wandb_project $WANDB_PROJECT \\
        --wandb_run_name 'round2_squad_combined' \\
        2>&1 | tee logs/round2_squad_combined_${TIMESTAMP}.log

    echo 'squad - combined completed!'
"

# ==============================================================================
# WIC runs (GPUs 4-6) - targeting 59.1% (currently 55.4%, barely above random)
# ==============================================================================

# GPU 4: Aggressive underfitting fix - minimal dampening, no regularization
SESSION_NAME="round2_wic_aggressive"
echo "Launching GPU 4: wic - aggressive (alpha=0.5, lambda=0.1, wd=0)"
screen -dmS "$SESSION_NAME" bash -c "
    cd /home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
    source /mnt/data/venv/bin/activate
    export WANDB_API_KEY='$WANDB_API_KEY'
    export PYTHONPATH=/mnt/data/python_packages:/home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
    export HF_HOME=/mnt/data/huggingface
    export HF_DATASETS_CACHE=/mnt/data/huggingface/datasets
    export TRANSFORMERS_CACHE=/mnt/data/huggingface/transformers
    export TMPDIR=/mnt/data/tmp
    export TRITON_CACHE_DIR=/mnt/data/triton_cache
    mkdir -p /mnt/data/tmp /mnt/data/triton_cache

    echo '=============================================='
    echo 'Round 2: wic - aggressive on GPU 4'
    echo '=============================================='

    CUDA_VISIBLE_DEVICES=4 python scripts/train.py \\
        --task wic \\
        --model $MODEL_PATH \\
        --solver spsa --use_1_5_spsa \\
        --saturating_alpha 0.5 \\
        --lambda_reg 0.1 \\
        --n_perts $N_PERTS \\
        --batch_size 32 \\
        --n_iterations $N_ITERATIONS \\
        --lr 1e-4 \\
        --epsilon 1e-4 \\
        --weight_decay 0.0 \\
        --seq_len 256 \\
        --eval_interval $EVAL_INTERVAL \\
        --checkpoint_interval $CHECKPOINT_INTERVAL \\
        --memory_efficient \\
        --wandb --wandb_project $WANDB_PROJECT \\
        --wandb_run_name 'round2_wic_aggressive' \\
        2>&1 | tee logs/round2_wic_aggressive_${TIMESTAMP}.log

    echo 'wic - aggressive completed!'
"

# GPU 5: Higher LR, no weight decay
SESSION_NAME="round2_wic_lr2e-4"
echo "Launching GPU 5: wic - lr=2e-4, wd=0"
screen -dmS "$SESSION_NAME" bash -c "
    cd /home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
    source /mnt/data/venv/bin/activate
    export WANDB_API_KEY='$WANDB_API_KEY'
    export PYTHONPATH=/mnt/data/python_packages:/home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
    export HF_HOME=/mnt/data/huggingface
    export HF_DATASETS_CACHE=/mnt/data/huggingface/datasets
    export TRANSFORMERS_CACHE=/mnt/data/huggingface/transformers
    export TMPDIR=/mnt/data/tmp
    export TRITON_CACHE_DIR=/mnt/data/triton_cache
    mkdir -p /mnt/data/tmp /mnt/data/triton_cache

    echo '=============================================='
    echo 'Round 2: wic - lr=2e-4 on GPU 5'
    echo '=============================================='

    CUDA_VISIBLE_DEVICES=5 python scripts/train.py \\
        --task wic \\
        --model $MODEL_PATH \\
        --solver spsa --use_1_5_spsa \\
        --saturating_alpha 0.1 \\
        --lambda_reg 1.0 \\
        --n_perts $N_PERTS \\
        --batch_size 32 \\
        --n_iterations $N_ITERATIONS \\
        --lr 2e-4 \\
        --epsilon 2e-4 \\
        --weight_decay 0.0 \\
        --seq_len 256 \\
        --eval_interval $EVAL_INTERVAL \\
        --checkpoint_interval $CHECKPOINT_INTERVAL \\
        --memory_efficient \\
        --wandb --wandb_project $WANDB_PROJECT \\
        --wandb_run_name 'round2_wic_lr2e-4' \\
        2>&1 | tee logs/round2_wic_lr2e-4_${TIMESTAMP}.log

    echo 'wic - lr2e-4 completed!'
"

# GPU 6: Moderate dampening with lower LR
SESSION_NAME="round2_wic_moderate"
echo "Launching GPU 6: wic - moderate (alpha=0.3, lr=5e-5)"
screen -dmS "$SESSION_NAME" bash -c "
    cd /home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
    source /mnt/data/venv/bin/activate
    export WANDB_API_KEY='$WANDB_API_KEY'
    export PYTHONPATH=/mnt/data/python_packages:/home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
    export HF_HOME=/mnt/data/huggingface
    export HF_DATASETS_CACHE=/mnt/data/huggingface/datasets
    export TRANSFORMERS_CACHE=/mnt/data/huggingface/transformers
    export TMPDIR=/mnt/data/tmp
    export TRITON_CACHE_DIR=/mnt/data/triton_cache
    mkdir -p /mnt/data/tmp /mnt/data/triton_cache

    echo '=============================================='
    echo 'Round 2: wic - moderate on GPU 6'
    echo '=============================================='

    CUDA_VISIBLE_DEVICES=6 python scripts/train.py \\
        --task wic \\
        --model $MODEL_PATH \\
        --solver spsa --use_1_5_spsa \\
        --saturating_alpha 0.3 \\
        --lambda_reg 1.0 \\
        --n_perts $N_PERTS \\
        --batch_size 32 \\
        --n_iterations $N_ITERATIONS \\
        --lr 5e-5 \\
        --epsilon 5e-5 \\
        --weight_decay 0.001 \\
        --seq_len 256 \\
        --eval_interval $EVAL_INTERVAL \\
        --checkpoint_interval $CHECKPOINT_INTERVAL \\
        --memory_efficient \\
        --wandb --wandb_project $WANDB_PROJECT \\
        --wandb_run_name 'round2_wic_moderate' \\
        2>&1 | tee logs/round2_wic_moderate_${TIMESTAMP}.log

    echo 'wic - moderate completed!'
"

echo ""
echo "=============================================="
echo "All 7 runs launched!"
echo "=============================================="
echo ""
echo "Monitor with:"
echo "  screen -ls                        # List all sessions"
echo "  screen -r round2_squad_lr5e-5     # Attach to a session"
echo "  tail -f logs/round2_*.log         # Watch log files"
echo ""
echo "Sessions:"
screen -ls | grep round2
