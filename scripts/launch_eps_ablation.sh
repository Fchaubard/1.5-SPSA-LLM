#!/bin/bash
# ==============================================================================
# Launch Epsilon Ablation Study
# Tests different epsilon values with fixed lr=1e-4 on SST-2 using all 8 GPUs
# ==============================================================================
screen -ls | grep '\.' | awk '{print $1}' | xargs -I{} screen -S {} -X quit 2>/dev/null
# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Model and task
MODEL="OPT13B"
TASK="sst2"

# Fixed hyperparameters
N_PERTS=40
BATCH_SIZE=32
ACCUM_STEPS=16                # effective_batch = 32 * 16 = 512
SEQ_LEN=256
LEARNING_RATE=1e-4            # Fixed LR for ablation
N_ITERATIONS=100              # Short runs for ablation
EVAL_INTERVAL=5
CHECKPOINT_INTERVAL=50
SEARCH_STRATEGY="none"
MEMORY_EFFICIENT=true

# Epsilon values to test (one per GPU)
declare -a EPSILONS
EPSILONS[0]="1e-2"
EPSILONS[1]="5e-3"
EPSILONS[2]="1e-3"
EPSILONS[3]="5e-4"
EPSILONS[4]="1e-4"
EPSILONS[5]="5e-5"
EPSILONS[6]="1e-5"
EPSILONS[7]="5e-6"

# W&B configuration
WANDB_ENABLED=true
WANDB_PROJECT="spsa-mezo-benchmarks"
WANDB_API_KEY="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"

# ==============================================================================
# DO NOT EDIT BELOW THIS LINE
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"
mkdir -p logs checkpoints

# Map model name to HuggingFace model path
case $MODEL in
    "OPT125M")
        MODEL_PATH="facebook/opt-125m"
        ;;
    "OPT13B")
        MODEL_PATH="facebook/opt-13b"
        ;;
    "OPT30B")
        MODEL_PATH="facebook/opt-30b"
        ;;
    "OPT66B")
        MODEL_PATH="facebook/opt-66b"
        ;;
    *)
        echo "Unknown model: $MODEL"
        exit 1
        ;;
esac

# Build W&B arguments
WANDB_ARGS=""
if [ "$WANDB_ENABLED" = true ]; then
    WANDB_ARGS="--wandb --wandb_project $WANDB_PROJECT"
fi

# Build memory efficient argument
MEM_EFF_ARG=""
if [ "$MEMORY_EFFICIENT" = true ]; then
    MEM_EFF_ARG="--memory_efficient"
fi

# Export W&B API key if set
if [ -n "$WANDB_API_KEY" ]; then
    export WANDB_API_KEY="$WANDB_API_KEY"
fi

echo "=============================================="
echo "Launching Epsilon Ablation Study"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Task:         $TASK"
echo "  Model:        $MODEL ($MODEL_PATH)"
echo "  N_Perts:      $N_PERTS"
echo "  Batch:        ${BATCH_SIZE}x${ACCUM_STEPS}=$((BATCH_SIZE*ACCUM_STEPS))"
echo "  Seq_len:      $SEQ_LEN"
echo "  LR (fixed):   $LEARNING_RATE"
echo "  Iterations:   $N_ITERATIONS"
echo "  Eval Int.:    $EVAL_INTERVAL"
echo "  Mem Eff:      $MEMORY_EFFICIENT"
echo "  W&B:          $WANDB_ENABLED"
echo ""
echo "Epsilon values to test:"
for i in {0..7}; do
    echo "  GPU $i: eps=${EPSILONS[$i]}"
done
echo ""
echo "=============================================="

# Launch each epsilon on its corresponding GPU
for GPU_ID in {0..7}; do
    EPS=${EPSILONS[$GPU_ID]}
    SESSION_NAME="eps_ablation_${EPS}"

    echo "Launching eps=$EPS on GPU $GPU_ID (screen: $SESSION_NAME)"

    screen -dmS "$SESSION_NAME" bash -c "
        cd $REPO_ROOT

        # Export W&B API key and Python path in screen session
        export WANDB_API_KEY='$WANDB_API_KEY'
        export PYTHONPATH=$REPO_ROOT

        echo '=============================================='
        echo 'Epsilon Ablation: eps=$EPS on GPU $GPU_ID'
        echo 'LR (fixed): $LEARNING_RATE'
        echo 'Screen session: $SESSION_NAME'
        echo '=============================================='

        CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/train.py \\
            --task $TASK \\
            --model $MODEL_PATH \\
            --solver spsa \\
            --n_perts $N_PERTS \\
            --batch_size $BATCH_SIZE \\
            --accum_steps $ACCUM_STEPS \\
            --n_iterations $N_ITERATIONS \\
            --lr $LEARNING_RATE \\
            --epsilon $EPS \\
            --seq_len $SEQ_LEN \\
            --eval_interval $EVAL_INTERVAL \\
            --checkpoint_interval $CHECKPOINT_INTERVAL \\
            --search_strategy $SEARCH_STRATEGY \\
            $MEM_EFF_ARG \\
            $WANDB_ARGS \\
            --wandb_run_name 'eps_ablation_eps${EPS}_lr${LEARNING_RATE}' \\
            2>&1 | tee logs/eps_ablation_${EPS}.log

        echo ''
        echo '=============================================='
        echo 'eps=$EPS ablation completed!'
        echo '=============================================='
    "
done

echo ""
echo "=============================================="
echo "All 8 epsilon ablation jobs launched!"
echo "=============================================="
echo ""
echo "Monitor with:"
echo "  screen -ls                       # List all sessions"
echo "  screen -r <session_name>         # Attach to a session"
echo "  tail -f logs/eps_ablation_*.log  # Watch log files"
echo ""
echo "Session names:"
for GPU_ID in {0..7}; do
    echo "  GPU $GPU_ID: eps_ablation_${EPSILONS[$GPU_ID]}"
done
echo ""
