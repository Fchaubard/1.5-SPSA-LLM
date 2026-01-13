#!/bin/bash
# ==============================================================================
# Launch MeZO Benchmark Training Jobs
# Runs all 6 benchmarks (SST-2, RTE, BoolQ, WSC, WiC, SQuAD) on GPUs 0-5
# ==============================================================================

# ==============================================================================
# CONFIGURATION - Edit these variables before running
# ==============================================================================

# Solver configuration
# Options: "1SPSA", "1.5SPSA", "MeZO", "Backprop"
SOLVER="1.5SPSA"

# Model configuration
# Options: "OPT125M" (for testing), "OPT13B", "OPT30B", "OPT66B"
MODEL="OPT13B"

# Training hyperparameters
N_PERTS=40                    # Number of perturbations per iteration
MICROBATCH_SIZE=16            # Batch size per forward pass
ACCUM_STEPS=1                 # Gradient accumulation steps (mainly for backprop)
SATURATING_ALPHA=0.1          # Exponent for 1.5-SPSA curvature scaling
LAMBDA_REG=1.0                # Minimum curvature regularization
N_ITERATIONS=1000000          # Total training iterations (1M)
EVAL_INTERVAL=100             # Evaluate every N iterations
CHECKPOINT_INTERVAL=1000      # Save checkpoint every N iterations (set high to save storage)
SEARCH_STRATEGY="none"        # LR search: "none", "line", "local", "binary", "quadratic"
LEARNING_RATE=1e-4            # Initial learning rate
EPSILON=1e-4                  # Perturbation size (defaults to LR if not set)

# Per-task sequence lengths (based on MeZO paper / typical input lengths)
# MeZO uses max_length=2048 but most tasks don't need that much
declare -A SEQ_LENS
SEQ_LENS[sst2]=128            # Short sentences
SEQ_LENS[rte]=256             # Sentence pairs
SEQ_LENS[boolq]=512           # Passage + question
SEQ_LENS[wsc]=128             # Short sentences
SEQ_LENS[wic]=256             # Two sentences + word
SEQ_LENS[squad]=512           # Context + question (can be long)

# W&B configuration
WANDB_ENABLED=false           # Set to true to enable W&B logging
WANDB_PROJECT="spsa-mezo-benchmarks"

# W&B API Key - SET THIS BEFORE RUNNING
# You can also set it in your environment: export WANDB_API_KEY=your_key
WANDB_API_KEY=""

# ==============================================================================
# DO NOT EDIT BELOW THIS LINE (unless you know what you're doing)
# ==============================================================================

cd /workspace/1.5-SPSA-LLM
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

# Map solver name to train.py arguments
case $SOLVER in
    "1SPSA")
        SOLVER_ARGS="--solver spsa"
        ;;
    "1.5SPSA")
        SOLVER_ARGS="--solver spsa --use_1_5_spsa --saturating_alpha $SATURATING_ALPHA --lambda_reg $LAMBDA_REG"
        ;;
    "MeZO")
        SOLVER_ARGS="--solver mezo"
        ;;
    "Backprop")
        SOLVER_ARGS="--solver backprop --accum_steps $ACCUM_STEPS"
        ;;
    *)
        echo "Unknown solver: $SOLVER"
        exit 1
        ;;
esac

# Build W&B arguments
WANDB_ARGS=""
if [ "$WANDB_ENABLED" = true ]; then
    WANDB_ARGS="--wandb --wandb_project $WANDB_PROJECT"
fi

# Export W&B API key if set
if [ -n "$WANDB_API_KEY" ]; then
    export WANDB_API_KEY="$WANDB_API_KEY"
fi

# Define benchmarks and their GPUs
declare -A BENCHMARKS
BENCHMARKS[0]="sst2"
BENCHMARKS[1]="rte"
BENCHMARKS[2]="boolq"
BENCHMARKS[3]="wsc"
BENCHMARKS[4]="wic"
BENCHMARKS[5]="squad"

echo "=============================================="
echo "Launching MeZO Benchmark Training Jobs"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Solver:       $SOLVER"
echo "  Model:        $MODEL ($MODEL_PATH)"
echo "  N_Perts:      $N_PERTS"
echo "  Batch Size:   $MICROBATCH_SIZE"
echo "  Accum Steps:  $ACCUM_STEPS"
echo "  Sat. Alpha:   $SATURATING_ALPHA"
echo "  Lambda Reg:   $LAMBDA_REG"
echo "  Iterations:   $N_ITERATIONS"
echo "  Eval Int.:    $EVAL_INTERVAL"
echo "  Ckpt Int.:    $CHECKPOINT_INTERVAL"
echo "  Search:       $SEARCH_STRATEGY"
echo "  LR:           $LEARNING_RATE"
echo "  W&B:          $WANDB_ENABLED"
echo ""
echo "Per-task sequence lengths:"
echo "  sst2:  ${SEQ_LENS[sst2]}"
echo "  rte:   ${SEQ_LENS[rte]}"
echo "  boolq: ${SEQ_LENS[boolq]}"
echo "  wsc:   ${SEQ_LENS[wsc]}"
echo "  wic:   ${SEQ_LENS[wic]}"
echo "  squad: ${SEQ_LENS[squad]}"
echo ""
echo "=============================================="

# Launch each benchmark on its corresponding GPU
for GPU_ID in {0..5}; do
    TASK=${BENCHMARKS[$GPU_ID]}
    TASK_SEQ_LEN=${SEQ_LENS[$TASK]}
    SESSION_NAME="${SOLVER}_${TASK}_${MODEL}_np${N_PERTS}_bs${MICROBATCH_SIZE}"

    echo "Launching $TASK on GPU $GPU_ID (screen: $SESSION_NAME, seq_len: $TASK_SEQ_LEN)"

    screen -dmS "$SESSION_NAME" bash -c "
        cd /workspace/1.5-SPSA-LLM

        # Export W&B API key and Python path in screen session
        export WANDB_API_KEY='$WANDB_API_KEY'
        export PYTHONPATH=/workspace/1.5-SPSA-LLM

        echo '=============================================='
        echo 'Starting $TASK training on GPU $GPU_ID'
        echo 'Solver: $SOLVER | Model: $MODEL'
        echo 'Seq Length: $TASK_SEQ_LEN'
        echo 'Screen session: $SESSION_NAME'
        echo '=============================================='

        CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/train.py \\
            --task $TASK \\
            --model $MODEL_PATH \\
            $SOLVER_ARGS \\
            --n_perts $N_PERTS \\
            --batch_size $MICROBATCH_SIZE \\
            --n_iterations $N_ITERATIONS \\
            --lr $LEARNING_RATE \\
            --epsilon $EPSILON \\
            --seq_len $TASK_SEQ_LEN \\
            --eval_interval $EVAL_INTERVAL \\
            --checkpoint_interval $CHECKPOINT_INTERVAL \\
            --search_strategy $SEARCH_STRATEGY \\
            $WANDB_ARGS \\
            --wandb_run_name '${SOLVER}_${TASK}_${MODEL}' \\
            2>&1 | tee logs/benchmark_${SOLVER}_${TASK}_${MODEL}.log

        echo ''
        echo '=============================================='
        echo '$TASK training completed!'
        echo '=============================================='
    "
done

echo ""
echo "=============================================="
echo "All 6 benchmark jobs launched!"
echo "=============================================="
echo ""
echo "Monitor with:"
echo "  screen -ls                    # List all sessions"
echo "  screen -r <session_name>      # Attach to a session"
echo "  tail -f logs/benchmark_*.log  # Watch log files"
echo ""
echo "Session names:"
for GPU_ID in {0..5}; do
    TASK=${BENCHMARKS[$GPU_ID]}
    echo "  GPU $GPU_ID: ${SOLVER}_${TASK}_${MODEL}_np${N_PERTS}_bs${MICROBATCH_SIZE}"
done
echo ""
