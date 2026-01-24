#!/bin/bash
# Ablation study for Atari Breakout with 1.5-SPSA
#
# Parameters:
#   n_perts: [40, 100]
#   lr=eps: [1e-3, 1e-4]
#   K (rollouts_per_pert): [10, 50]
#
# Total: 2 x 2 x 2 = 8 experiments, one per GPU

set -e

cd /home/romeo/1.5-SPSA-LLM

# Create logs directory
mkdir -p logs

# Common parameters
N_ITERATIONS=500
NUM_WORKERS=0  # Sequential evaluation (spawn is too slow)
EVAL_INTERVAL=10
CHECKPOINT_INTERVAL=50

# Launch experiments
GPU=0
for N_PERTS in 40 100; do
    for LR in 1e-3 1e-4; do
        for K in 10 50; do
            NAME="atari_np${N_PERTS}_lr${LR}_k${K}"
            LOG="logs/${NAME}_$(date +%Y%m%d_%H%M%S).log"

            echo "Launching ${NAME} on GPU ${GPU}..."

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

            GPU=$((GPU + 1))
        done
    done
done

echo ""
echo "========================================"
echo "Launched 8 experiments on GPUs 0-7"
echo "========================================"
echo ""
echo "Ablation grid:"
echo "  n_perts: [40, 100]"
echo "  lr=eps: [1e-3, 1e-4]"
echo "  K: [10, 50]"
echo ""
echo "Monitor with: screen -ls"
echo "Attach to a session: screen -r <name>"
echo "View logs: tail -f logs/atari_*.log"
