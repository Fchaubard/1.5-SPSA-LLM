#!/bin/bash
# ==============================================================================
# Launch MeZO Benchmark Training Jobs
# Runs all 6 benchmarks (SST-2, RTE, BoolQ, WSC, WiC, SQuAD) on GPUs 0-5
# ==============================================================================
screen -ls | grep '\.' | awk '{print $1}' | xargs -I{} screen -S {} -X quit 2>/dev/null

# ==============================================================================
# CONFIGURATION - Edit these variables before running
# ==============================================================================

# Solver configuration
# Options: "1SPSA", "1.5SPSA", "MeZO", "Backprop"
SOLVER="1.5SPSA"

# Model configuration
# Options: "OPT125M" (for testing), "OPT13B", "OPT30B", "OPT66B"
MODEL="OPT30B"

# Training hyperparameters
N_PERTS=40                    # Number of perturbations per iteration
MICROBATCH_SIZE=32            # Batch size per forward pass (default for most tasks)
ACCUM_STEPS=16                # Gradient accumulation steps (effective batch = 32*16 = 512)

# Per-task batch sizes
# Note: accum_steps only works for backprop, not SPSA
# For SPSA, batch_size IS the effective batch size per perturbation
# Shorter sequences allow larger batches
declare -A TASK_BATCH
TASK_BATCH[sst2]=32           # seq=128, short sentences
TASK_BATCH[rte]=32            # seq=256
TASK_BATCH[boolq]=32          # seq=512, longer passages
TASK_BATCH[wsc]=32            # seq=128, short sentences
TASK_BATCH[wic]=32            # seq=256
TASK_BATCH[squad]=28          # seq=512, reduced for generation OOM
SATURATING_ALPHA=0.1          # Exponent for 1.5-SPSA curvature scaling
LAMBDA_REG=1.0                # Minimum curvature regularization
N_ITERATIONS=1000000          # Total training iterations (1M)
EVAL_INTERVAL=10              # Evaluate every N iterations
CHECKPOINT_INTERVAL=100       # Save checkpoint every N iterations
SEARCH_STRATEGY="none"        # LR search: "none", "line", "local", "binary", "quadratic"
LEARNING_RATE=1e-4            # Initial learning rate
EPSILON=1e-4                  # Perturbation size (defaults to LR if not set)
WEIGHT_DECAY=0.001            # Weight decay (L2 regularization)
MEMORY_EFFICIENT=true         # Memory-efficient mode: regenerate RNG instead of caching gradients (required for OPT-13B+ on 40GB GPUs)

# Per-task sequence lengths (based on MeZO paper / typical input lengths)
# MeZO uses max_length=2048 but most tasks don't need that much
declare -A SEQ_LENS
SEQ_LENS[sst2]=256            # Short sentences
SEQ_LENS[rte]=256             # Sentence pairs
SEQ_LENS[boolq]=512           # Passage + question
SEQ_LENS[wsc]=256             # Short sentences
SEQ_LENS[wic]=256             # Two sentences + word
SEQ_LENS[squad]=512           # Context + question (can be long)

# Timestamp for unique log names (NEVER overwrite logs!)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# W&B configuration
WANDB_ENABLED=false           # Set to true to enable W&B logging
WANDB_PROJECT="spsa-mezo-benchmarks"

# W&B API Key - SET THIS BEFORE RUNNING
# You can also set it in your environment: export WANDB_API_KEY=your_key
WANDB_API_KEY=""

# ==============================================================================
# DO NOT EDIT BELOW THIS LINE (unless you know what you're doing)
# ==============================================================================

cd /home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
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

# Build memory efficient argument
MEM_EFF_ARG=""
if [ "$MEMORY_EFFICIENT" = true ]; then
    MEM_EFF_ARG="--memory_efficient"
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
echo "  Mem Eff:      $MEMORY_EFFICIENT"
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
    TASK_BS=${TASK_BATCH[$TASK]}
    SESSION_NAME="${SOLVER}_${TASK}_${MODEL}_np${N_PERTS}_bs${TASK_BS}"

    echo "Launching $TASK on GPU $GPU_ID (screen: $SESSION_NAME, seq_len: $TASK_SEQ_LEN, batch: ${TASK_BS})"

    screen -dmS "$SESSION_NAME" bash -c "
        cd /home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM

        # Export W&B API key and Python path in screen session
        export WANDB_API_KEY='$WANDB_API_KEY'
        export PYTHONPATH=/mnt/data/python_packages:/home/ec2-user/functor/garbage/zoo/1.5-SPSA-LLM
        export HF_HOME=/mnt/data/huggingface
        export HF_DATASETS_CACHE=/mnt/data/huggingface/datasets
        export TRANSFORMERS_CACHE=/mnt/data/huggingface/transformers
        export TMPDIR=/mnt/data/tmp
        export TRITON_CACHE_DIR=/mnt/data/triton_cache
        mkdir -p /mnt/data/tmp /mnt/data/triton_cache

        echo '=============================================='
        echo 'Starting $TASK training on GPU $GPU_ID'
        echo 'Solver: $SOLVER | Model: $MODEL'
        echo 'Seq Length: $TASK_SEQ_LEN'
        echo 'Batch Size: $TASK_BS'
        echo 'Screen session: $SESSION_NAME'
        echo '=============================================='

        CUDA_VISIBLE_DEVICES=$GPU_ID /usr/bin/python3.9 scripts/train.py \\
            --task $TASK \\
            --model $MODEL_PATH \\
            $SOLVER_ARGS \\
            --n_perts $N_PERTS \\
            --batch_size $TASK_BS \\
            --accum_steps $ACCUM_STEPS \\
            --n_iterations $N_ITERATIONS \\
            --lr $LEARNING_RATE \\
            --epsilon $EPSILON \\
            --weight_decay $WEIGHT_DECAY \\
            --seq_len $TASK_SEQ_LEN \\
            --eval_interval $EVAL_INTERVAL \\
            --checkpoint_interval $CHECKPOINT_INTERVAL \\
            --search_strategy $SEARCH_STRATEGY \\
            $MEM_EFF_ARG \\
            $WANDB_ARGS \\
            --wandb_run_name '${SOLVER}_${TASK}_${MODEL}' \\
            2>&1 | tee logs/benchmark_${SOLVER}_${TASK}_${MODEL}_${TIMESTAMP}.log

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
    TASK_BS=${TASK_BATCH[$TASK]}
    echo "  GPU $GPU_ID: ${SOLVER}_${TASK}_${MODEL}_np${N_PERTS}_bs${TASK_BS}"
done
echo ""
