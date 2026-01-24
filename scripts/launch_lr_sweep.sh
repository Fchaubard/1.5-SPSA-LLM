#!/bin/bash
# Launch LR sweep for SQuAD and WIC
# 8 runs: 4 LRs x 2 tasks
# Settings: eb=512, wd=0, n_perts=20, lr=eps

cd /home/romeo/1.5-SPSA-LLM
mkdir -p logs checkpoints

WANDB_API_KEY='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'

# LR values to sweep
LRS=("1e-4" "5e-5" "1e-5" "5e-6")
TASKS=("squad" "wic")

# Task-specific settings
declare -A SEQ_LENS
SEQ_LENS[squad]=512
SEQ_LENS[wic]=256

GPU=0
for task in "${TASKS[@]}"; do
    for lr in "${LRS[@]}"; do
        SEQ_LEN=${SEQ_LENS[$task]}
        SESSION="${task}_lr${lr}_np20_eb512"

        echo "Launching $SESSION on GPU $GPU"

        screen -dmS "$SESSION" bash -c "
            cd /home/romeo/1.5-SPSA-LLM
            export WANDB_API_KEY='$WANDB_API_KEY'
            export PYTHONPATH=/home/romeo/1.5-SPSA-LLM

            CUDA_VISIBLE_DEVICES=$GPU python scripts/train.py \
                --task $task \
                --model facebook/opt-13b \
                --solver spsa --use_1_5_spsa --saturating_alpha 0.1 --lambda_reg 1.0 \
                --n_perts 20 \
                --batch_size 32 \
                --accum_steps 16 \
                --n_iterations 1000000 \
                --lr $lr \
                --epsilon $lr \
                --weight_decay 0 \
                --seq_len $SEQ_LEN \
                --eval_interval 10 \
                --checkpoint_interval 100 \
                --search_strategy none \
                --memory_efficient \
                --wandb --wandb_project spsa-mezo-benchmarks \
                --wandb_run_name '${task}_lr${lr}_np20_eb512' \
                2>&1 | tee logs/${task}_lr${lr}_np20_eb512.log
        "

        GPU=$((GPU + 1))
    done
done

echo ""
echo "Launched 8 runs:"
screen -ls | grep -E "squad|wic"
