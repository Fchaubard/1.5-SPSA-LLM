#!/bin/bash
# ==============================================================================
# Round 3: Weight Decay Sweep for WIC & SQuAD
# SQuAD on GPUs 0-3, WIC on GPUs 4-6
# ==============================================================================

cd /home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
mkdir -p logs checkpoints

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "Launching Round 3: Weight Decay Sweep"
echo "=============================================="
echo ""
echo "SQuAD (GPUs 0-3): wd=0.1, 0.01, 0.001, 0"
echo "WIC (GPUs 4-6): wd=0.001, 0.01, 0"
echo "All: lr=eps=1e-4, n_perts=40, bs=32, accum=32 (eff=1024)"
echo ""

# GPU 0: SQuAD wd=0.1
/usr/bin/screen -dmS "round3_squad_wd0.1" bash -c "
    cd /home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
    source /mnt/data/venv/bin/activate
    export PYTHONPATH=/home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
    export HF_HOME=/mnt/data/huggingface
    export HF_DATASETS_CACHE=/mnt/data/huggingface/datasets
    export TRANSFORMERS_CACHE=/mnt/data/huggingface/transformers
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py --task squad --model facebook/opt-30b --solver spsa --use_1_5_spsa --saturating_alpha 0.1 --lambda_reg 1.0 --n_perts 40 --batch_size 32 --accum_steps 32 --lr 1e-4 --epsilon 1e-4 --weight_decay 0.1 --seq_len 512 --eval_interval 10 --n_iterations 1000000 --memory_efficient 2>&1 | tee logs/round3_squad_wd0.1_${TIMESTAMP}.log
"
echo "GPU 0: SQuAD wd=0.1"

# GPU 1: SQuAD wd=0.01
/usr/bin/screen -dmS "round3_squad_wd0.01" bash -c "
    cd /home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
    source /mnt/data/venv/bin/activate
    export PYTHONPATH=/home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
    export HF_HOME=/mnt/data/huggingface
    export HF_DATASETS_CACHE=/mnt/data/huggingface/datasets
    export TRANSFORMERS_CACHE=/mnt/data/huggingface/transformers
    CUDA_VISIBLE_DEVICES=1 python scripts/train.py --task squad --model facebook/opt-30b --solver spsa --use_1_5_spsa --saturating_alpha 0.1 --lambda_reg 1.0 --n_perts 40 --batch_size 32 --accum_steps 32 --lr 1e-4 --epsilon 1e-4 --weight_decay 0.01 --seq_len 512 --eval_interval 10 --n_iterations 1000000 --memory_efficient 2>&1 | tee logs/round3_squad_wd0.01_${TIMESTAMP}.log
"
echo "GPU 1: SQuAD wd=0.01"

# GPU 2: SQuAD wd=0.001
/usr/bin/screen -dmS "round3_squad_wd0.001" bash -c "
    cd /home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
    source /mnt/data/venv/bin/activate
    export PYTHONPATH=/home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
    export HF_HOME=/mnt/data/huggingface
    export HF_DATASETS_CACHE=/mnt/data/huggingface/datasets
    export TRANSFORMERS_CACHE=/mnt/data/huggingface/transformers
    CUDA_VISIBLE_DEVICES=2 python scripts/train.py --task squad --model facebook/opt-30b --solver spsa --use_1_5_spsa --saturating_alpha 0.1 --lambda_reg 1.0 --n_perts 40 --batch_size 32 --accum_steps 32 --lr 1e-4 --epsilon 1e-4 --weight_decay 0.001 --seq_len 512 --eval_interval 10 --n_iterations 1000000 --memory_efficient 2>&1 | tee logs/round3_squad_wd0.001_${TIMESTAMP}.log
"
echo "GPU 2: SQuAD wd=0.001"

# GPU 3: SQuAD wd=0
/usr/bin/screen -dmS "round3_squad_wd0" bash -c "
    cd /home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
    source /mnt/data/venv/bin/activate
    export PYTHONPATH=/home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
    export HF_HOME=/mnt/data/huggingface
    export HF_DATASETS_CACHE=/mnt/data/huggingface/datasets
    export TRANSFORMERS_CACHE=/mnt/data/huggingface/transformers
    CUDA_VISIBLE_DEVICES=3 python scripts/train.py --task squad --model facebook/opt-30b --solver spsa --use_1_5_spsa --saturating_alpha 0.1 --lambda_reg 1.0 --n_perts 40 --batch_size 32 --accum_steps 32 --lr 1e-4 --epsilon 1e-4 --weight_decay 0 --seq_len 512 --eval_interval 10 --n_iterations 1000000 --memory_efficient 2>&1 | tee logs/round3_squad_wd0_${TIMESTAMP}.log
"
echo "GPU 3: SQuAD wd=0"

# GPU 4: WIC wd=0.001
/usr/bin/screen -dmS "round3_wic_wd0.001" bash -c "
    cd /home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
    source /mnt/data/venv/bin/activate
    export PYTHONPATH=/home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
    export HF_HOME=/mnt/data/huggingface
    export HF_DATASETS_CACHE=/mnt/data/huggingface/datasets
    export TRANSFORMERS_CACHE=/mnt/data/huggingface/transformers
    CUDA_VISIBLE_DEVICES=4 python scripts/train.py --task wic --model facebook/opt-30b --solver spsa --use_1_5_spsa --saturating_alpha 0.1 --lambda_reg 1.0 --n_perts 40 --batch_size 32 --accum_steps 32 --lr 1e-4 --epsilon 1e-4 --weight_decay 0.001 --seq_len 256 --eval_interval 10 --n_iterations 1000000 --memory_efficient 2>&1 | tee logs/round3_wic_wd0.001_${TIMESTAMP}.log
"
echo "GPU 4: WIC wd=0.001"

# GPU 5: WIC wd=0.01
/usr/bin/screen -dmS "round3_wic_wd0.01" bash -c "
    cd /home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
    source /mnt/data/venv/bin/activate
    export PYTHONPATH=/home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
    export HF_HOME=/mnt/data/huggingface
    export HF_DATASETS_CACHE=/mnt/data/huggingface/datasets
    export TRANSFORMERS_CACHE=/mnt/data/huggingface/transformers
    CUDA_VISIBLE_DEVICES=5 python scripts/train.py --task wic --model facebook/opt-30b --solver spsa --use_1_5_spsa --saturating_alpha 0.1 --lambda_reg 1.0 --n_perts 40 --batch_size 32 --accum_steps 32 --lr 1e-4 --epsilon 1e-4 --weight_decay 0.01 --seq_len 256 --eval_interval 10 --n_iterations 1000000 --memory_efficient 2>&1 | tee logs/round3_wic_wd0.01_${TIMESTAMP}.log
"
echo "GPU 5: WIC wd=0.01"

# GPU 6: WIC wd=0
/usr/bin/screen -dmS "round3_wic_wd0" bash -c "
    cd /home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
    source /mnt/data/venv/bin/activate
    export PYTHONPATH=/home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
    export HF_HOME=/mnt/data/huggingface
    export HF_DATASETS_CACHE=/mnt/data/huggingface/datasets
    export TRANSFORMERS_CACHE=/mnt/data/huggingface/transformers
    CUDA_VISIBLE_DEVICES=6 python scripts/train.py --task wic --model facebook/opt-30b --solver spsa --use_1_5_spsa --saturating_alpha 0.1 --lambda_reg 1.0 --n_perts 40 --batch_size 32 --accum_steps 32 --lr 1e-4 --epsilon 1e-4 --weight_decay 0 --seq_len 256 --eval_interval 10 --n_iterations 1000000 --memory_efficient 2>&1 | tee logs/round3_wic_wd0_${TIMESTAMP}.log
"
echo "GPU 6: WIC wd=0"

echo ""
echo "=============================================="
echo "All 7 jobs launched!"
echo "=============================================="
echo ""
echo "Monitor with:"
echo "  screen -ls"
echo "  screen -r <session_name>"
echo "  tail -f logs/round3_*.log"
