#!/bin/bash
# Safe ablation study for Atari Breakout with 1.5-SPSA
#
# Runs experiments in batches of 2 to avoid system overload
# Uses 2 workers per experiment instead of 4

set -e

cd /home/romeo/1.5-SPSA-LLM
mkdir -p logs

# Common parameters
N_ITERATIONS=500
NUM_WORKERS=2  # Reduced from 4
EVAL_INTERVAL=10
CHECKPOINT_INTERVAL=50

run_batch() {
    local BATCH_NAME=$1
    shift
    local CONFIGS=("$@")

    echo "Starting batch: $BATCH_NAME"

    for config in "${CONFIGS[@]}"; do
        IFS=':' read -r GPU N_PERTS LR K <<< "$config"
        NAME="atari_np${N_PERTS}_lr${LR}_k${K}"
        LOG="logs/${NAME}_$(date +%Y%m%d_%H%M%S).log"

        echo "  Launching ${NAME} on GPU ${GPU}..."

        screen -dmS "${NAME}" bash -c "
            export CUDA_VISIBLE_DEVICES=${GPU}
            python scripts/train.py \
                --task atari_breakout \
                --solver spsa \
                --use_1_5_spsa \
                --n_perts ${N_PERTS} \
                --lr ${LR} \
                --epsilon ${LR} \
                --rollouts_per_pert ${K} \
                --num_workers ${NUM_WORKERS} \
                --n_iterations ${N_ITERATIONS} \
                --eval_interval ${EVAL_INTERVAL} \
                --checkpoint_interval ${CHECKPOINT_INTERVAL} \
                --wandb \
                --wandb_project '1.5-spsa-atari-ablation' \
                --wandb_run_name '${NAME}' \
                2>&1 | tee ${LOG}
        "
    done

    echo "Batch ${BATCH_NAME} launched. Waiting for completion..."

    # Wait for all screens in this batch to finish
    while true; do
        active=$(screen -ls 2>/dev/null | grep -c "atari_" || echo 0)
        if [ "$active" -eq 0 ]; then
            echo "Batch ${BATCH_NAME} complete!"
            break
        fi
        sleep 60
        # Show progress
        for config in "${CONFIGS[@]}"; do
            IFS=':' read -r GPU N_PERTS LR K <<< "$config"
            NAME="atari_np${N_PERTS}_lr${LR}_k${K}"
            LOG=$(ls logs/${NAME}_*.log 2>/dev/null | tail -1)
            if [ -n "$LOG" ]; then
                iter=$(grep -c "Iter" "$LOG" 2>/dev/null || echo 0)
                echo "    ${NAME}: ${iter} iterations"
            fi
        done
    done
}

echo "========================================"
echo "Safe Ablation Study (2 workers, 2 concurrent)"
echo "========================================"

# Batch 1: GPU 0-1 (2 experiments)
run_batch "1" \
    "0:40:1e-3:10" \
    "1:40:1e-4:10"

# Batch 2: GPU 0-1
run_batch "2" \
    "0:40:1e-3:50" \
    "1:40:1e-4:50"

# Batch 3: GPU 0-1
run_batch "3" \
    "0:100:1e-3:10" \
    "1:100:1e-4:10"

# Batch 4: GPU 0-1
run_batch "4" \
    "0:100:1e-3:50" \
    "1:100:1e-4:50"

echo "========================================"
echo "All experiments complete!"
echo "========================================"
