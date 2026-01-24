#!/bin/bash
# Launch script for Atari Breakout training with 1.5-SPSA
#
# This trains a CNN policy on Atari Breakout using:
# - 1.5-SPSA optimizer with curvature scaling
# - Multiprocessing for parallel rollout collection
# - ~1.7M parameter CNN (DQN-style architecture)

set -e

# Create logs directory
mkdir -p logs

# Default settings (can be overridden via environment variables or command line)
N_PERTS=${N_PERTS:-40}
ROLLOUTS_PER_PERT=${ROLLOUTS_PER_PERT:-10}
NUM_WORKERS=${NUM_WORKERS:-8}
N_ITERATIONS=${N_ITERATIONS:-1000}
LR=${LR:-1e-3}
EPSILON=${EPSILON:-1e-3}
EVAL_INTERVAL=${EVAL_INTERVAL:-10}
WANDB_PROJECT=${WANDB_PROJECT:-"1.5-spsa-atari"}

echo "========================================"
echo "Atari Breakout Training with 1.5-SPSA"
echo "========================================"
echo "n_perts: ${N_PERTS}"
echo "rollouts_per_pert: ${ROLLOUTS_PER_PERT}"
echo "num_workers: ${NUM_WORKERS}"
echo "n_iterations: ${N_ITERATIONS}"
echo "lr: ${LR}"
echo "epsilon: ${EPSILON}"
echo "========================================"

python scripts/train.py \
    --task atari_breakout \
    --solver spsa \
    --use_1_5_spsa \
    --n_perts ${N_PERTS} \
    --rollouts_per_pert ${ROLLOUTS_PER_PERT} \
    --num_workers ${NUM_WORKERS} \
    --n_iterations ${N_ITERATIONS} \
    --lr ${LR} \
    --epsilon ${EPSILON} \
    --eval_interval ${EVAL_INTERVAL} \
    --checkpoint_interval 100 \
    --wandb \
    --wandb_project "${WANDB_PROJECT}" \
    "$@" \
    2>&1 | tee logs/atari_breakout_$(date +%Y%m%d_%H%M%S).log
