#!/bin/bash
# Launch all 1.5-SPSA training jobs
# Now with test accuracy tracking and checkpointing on best test

cd /workspace/1.5-SPSA-LLM
mkdir -p logs checkpoints

echo "=============================================="
echo "Launching 8 training jobs on 8 GPUs"
echo "=============================================="

# GPU 0: np40_bs16 (lr=1e-7) then np160_bs16 (lr=3e-5) - sequential
screen -dmS gpu0_jobs bash -c '
    cd /workspace/1.5-SPSA-LLM
    echo "Starting: n_perts=40, batch=16, lr=1e-7 on GPU 0"
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --task sst2 --model facebook/opt-13b --solver spsa --use_1_5_spsa \
        --saturating_alpha 0.1 --lambda_reg 1.0 \
        --n_perts 40 --n_iterations 10000 --lr 1e-7 --batch_size 16 \
        --seq_len 256 --eval_interval 20 --search_strategy none \
        2>&1 | tee logs/job_np40_bs16.log

    echo "Starting: n_perts=160, batch=16, lr=3e-5 on GPU 0"
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --task sst2 --model facebook/opt-13b --solver spsa --use_1_5_spsa \
        --saturating_alpha 0.1 --lambda_reg 1.0 \
        --n_perts 160 --n_iterations 2500 --lr 3e-5 --batch_size 16 \
        --seq_len 256 --eval_interval 5 --search_strategy none \
        2>&1 | tee logs/job_np160_bs16.log
'
echo "GPU 0: np40_bs16 -> np160_bs16 (sequential)"

# GPU 1: np40_bs128
screen -dmS gpu1_job bash -c '
    cd /workspace/1.5-SPSA-LLM
    CUDA_VISIBLE_DEVICES=1 python train.py \
        --task sst2 --model facebook/opt-13b --solver spsa --use_1_5_spsa \
        --saturating_alpha 0.1 --lambda_reg 1.0 \
        --n_perts 40 --n_iterations 1250 --lr 5e-5 --batch_size 128 \
        --seq_len 256 --eval_interval 3 --search_strategy none \
        2>&1 | tee logs/job_np40_bs128.log
'
echo "GPU 1: np40_bs128"

# GPU 2: np40_bs512
screen -dmS gpu2_job bash -c '
    cd /workspace/1.5-SPSA-LLM
    CUDA_VISIBLE_DEVICES=2 python train.py \
        --task sst2 --model facebook/opt-13b --solver spsa --use_1_5_spsa \
        --saturating_alpha 0.1 --lambda_reg 1.0 \
        --n_perts 40 --n_iterations 314 --lr 1e-4 --batch_size 512 \
        --seq_len 256 --eval_interval 1 --search_strategy none \
        2>&1 | tee logs/job_np40_bs512.log
'
echo "GPU 2: np40_bs512"

# GPU 3: np160_bs128
screen -dmS gpu3_job bash -c '
    cd /workspace/1.5-SPSA-LLM
    CUDA_VISIBLE_DEVICES=3 python train.py \
        --task sst2 --model facebook/opt-13b --solver spsa --use_1_5_spsa \
        --saturating_alpha 0.1 --lambda_reg 1.0 \
        --n_perts 160 --n_iterations 313 --lr 1e-4 --batch_size 128 \
        --seq_len 256 --eval_interval 1 --search_strategy none \
        2>&1 | tee logs/job_np160_bs128.log
'
echo "GPU 3: np160_bs128"

# GPU 4: np160_bs512
screen -dmS gpu4_job bash -c '
    cd /workspace/1.5-SPSA-LLM
    CUDA_VISIBLE_DEVICES=4 python train.py \
        --task sst2 --model facebook/opt-13b --solver spsa --use_1_5_spsa \
        --saturating_alpha 0.1 --lambda_reg 1.0 \
        --n_perts 160 --n_iterations 78 --lr 3e-4 --batch_size 512 \
        --seq_len 256 --eval_interval 1 --search_strategy none \
        2>&1 | tee logs/job_np160_bs512.log
'
echo "GPU 4: np160_bs512"

# GPU 5: np640_bs16 (lr=1e-6)
screen -dmS gpu5_job bash -c '
    cd /workspace/1.5-SPSA-LLM
    CUDA_VISIBLE_DEVICES=5 python train.py \
        --task sst2 --model facebook/opt-13b --solver spsa --use_1_5_spsa \
        --saturating_alpha 0.1 --lambda_reg 1.0 \
        --n_perts 640 --n_iterations 625 --lr 1e-6 --batch_size 16 \
        --seq_len 256 --eval_interval 2 --search_strategy none \
        2>&1 | tee logs/job_np640_bs16.log
'
echo "GPU 5: np640_bs16"

# GPU 6: np640_bs128
screen -dmS gpu6_job bash -c '
    cd /workspace/1.5-SPSA-LLM
    CUDA_VISIBLE_DEVICES=6 python train.py \
        --task sst2 --model facebook/opt-13b --solver spsa --use_1_5_spsa \
        --saturating_alpha 0.1 --lambda_reg 1.0 \
        --n_perts 640 --n_iterations 79 --lr 2e-4 --batch_size 128 \
        --seq_len 256 --eval_interval 1 --search_strategy none \
        2>&1 | tee logs/job_np640_bs128.log
'
echo "GPU 6: np640_bs128"

# GPU 7: np640_bs512
screen -dmS gpu7_job bash -c '
    cd /workspace/1.5-SPSA-LLM
    CUDA_VISIBLE_DEVICES=7 python train.py \
        --task sst2 --model facebook/opt-13b --solver spsa --use_1_5_spsa \
        --saturating_alpha 0.1 --lambda_reg 1.0 \
        --n_perts 640 --n_iterations 20 --lr 5e-4 --batch_size 512 \
        --seq_len 256 --eval_interval 1 --search_strategy none \
        2>&1 | tee logs/job_np640_bs512.log
'
echo "GPU 7: np640_bs512"

echo ""
echo "=============================================="
echo "All jobs launched!"
echo "=============================================="
echo ""
echo "Monitor with: ./summarize.sh"
echo "View screens: screen -ls"
echo "Attach: screen -r <name>"
echo ""
