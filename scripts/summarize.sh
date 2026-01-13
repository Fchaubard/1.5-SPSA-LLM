#!/bin/bash

# Summarize 1.5-SPSA training runs from log files
# Parses output and creates a table sorted by validation accuracy

LOGDIR="/workspace/1.5-SPSA-LLM/logs"
TMPFILE=$(mktemp)

# Use python for reliable parsing
python3 - "$LOGDIR" "$TMPFILE" << 'PYEOF'
import sys
import os
import re
from pathlib import Path

logdir = sys.argv[1]
outfile = sys.argv[2]

results = []

for logfile in Path(logdir).glob("*.log"):
    try:
        with open(logfile, 'r', errors='ignore') as f:
            content = f.read()
    except:
        continue

    # Skip old failed logs (bs1024)
    if 'bs1024' in str(logfile):
        continue

    name = logfile.stem  # e.g., job_np40_bs128

    # Extract hyperparameters
    n_perts = None
    batch_size = None
    lr = None
    max_iters = None
    sat_alpha = None

    match = re.search(r'n_perts:\s*(\d+)', content)
    if match:
        n_perts = int(match.group(1))

    match = re.search(r'batch_size:\s*(\d+)', content)
    if match:
        batch_size = int(match.group(1))

    match = re.search(r'\]\s+lr:\s*([0-9.e+-]+)', content)
    if match:
        lr = match.group(1)

    match = re.search(r'n_iterations:\s*(\d+)', content)
    if match:
        max_iters = int(match.group(1))

    match = re.search(r'saturating_alpha=([0-9.]+)', content)
    if match:
        sat_alpha = match.group(1)

    # Extract initial values
    initial_loss = None
    initial_val_acc = None

    match = re.search(r'Initial loss:\s*([0-9.]+)', content)
    if match:
        initial_loss = float(match.group(1))

    match = re.search(r'Initial val accuracy:\s*([0-9.]+)', content)
    if match:
        initial_val_acc = float(match.group(1))

    # Extract training progress
    # Format: [HH:MM:SS] Iter      X | Loss: Y.YYYY | Acc: ZZ.Z% | Best: W.WWWW | ...
    iter_matches = re.findall(
        r'Iter\s+(\d+)\s*\|\s*Loss:\s*([0-9.]+)\s*\|\s*Acc:\s*([0-9.]+)%',
        content
    )

    current_iter = 0
    train_loss = initial_loss
    train_acc = None

    if iter_matches:
        last = iter_matches[-1]
        current_iter = int(last[0])
        train_loss = float(last[1])
        train_acc = float(last[2])

    # Extract validation accuracy (from eval intervals)
    # Old format: >>> [Eval] Val Accuracy: 0.9240 (92.4%)
    # New format: >>> [Eval] Val: 0.9240 (92.4%) | Test: 0.9100 (91.0%)
    val_matches = re.findall(r'\[Eval\] Val Accuracy:\s*([0-9.]+)', content)
    val_matches_new = re.findall(r'\[Eval\] Val:\s*([0-9.]+)', content)
    val_acc = initial_val_acc
    if val_matches_new:
        val_acc = float(val_matches_new[-1])
    elif val_matches:
        val_acc = float(val_matches[-1])

    # Extract test accuracy (new format only)
    test_matches = re.findall(r'\| Test:\s*([0-9.]+)', content)
    test_acc = None
    if test_matches:
        test_acc = float(test_matches[-1])

    # Determine status
    status = "running"
    if 'OutOfMemoryError' in content or 'CUDA out of memory' in content:
        status = "OOM"
    elif current_iter >= (max_iters or 9999) - 1:
        status = "completed"
    elif 'TRAINING' not in content:
        status = "loading"
    elif current_iter == 0 and train_acc is None:
        status = "warmup"

    # Only add if we have basic info
    if n_perts and batch_size:
        results.append({
            'name': name,
            'n_perts': n_perts,
            'batch_size': batch_size,
            'lr': lr or '-',
            'sat_alpha': sat_alpha or '-',
            'max_iters': max_iters or 0,
            'current_iter': current_iter,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'status': status
        })

# Sort by val_acc (highest first)
def sort_key(x):
    va = x['val_acc']
    if va is None:
        return -1
    return va

results.sort(key=sort_key, reverse=True)

# Write to output file
with open(outfile, 'w') as f:
    for r in results:
        train_loss_str = f"{r['train_loss']:.4f}" if r['train_loss'] else '-'
        train_acc_str = f"{r['train_acc']:.1f}%" if r['train_acc'] else '-'
        val_acc_str = f"{r['val_acc']*100:.1f}%" if r['val_acc'] else '-'
        test_acc_str = f"{r['test_acc']*100:.1f}%" if r['test_acc'] else '-'
        progress = f"{r['current_iter']}/{r['max_iters']}"
        f.write(f"{r['n_perts']}|{r['batch_size']}|{r['lr']}|{progress}|{train_loss_str}|{train_acc_str}|{val_acc_str}|{test_acc_str}|{r['status']}|{r['name']}\n")
PYEOF

echo ""
echo "========================================================================================================================"
echo "                              1.5-SPSA TRAINING SUMMARY (sorted by Val Acc)"
echo "========================================================================================================================"
printf "%-7s | %-5s | %-8s | %-11s | %-10s | %-9s | %-8s | %-8s | %-10s | %s\n" \
    "n_perts" "batch" "lr" "progress" "train_loss" "train_acc" "val_acc" "test_acc" "status" "log"
echo "------------------------------------------------------------------------------------------------------------------------"

# Display sorted results
if [ -s "$TMPFILE" ]; then
    while IFS='|' read -r n_perts batch lr progress train_loss train_acc val_acc test_acc status name; do
        printf "%-7s | %-5s | %-8s | %-11s | %-10s | %-9s | %-8s | %-8s | %-10s | %s\n" \
            "$n_perts" "$batch" "$lr" "$progress" "$train_loss" "$train_acc" "$val_acc" "$test_acc" "$status" "$name"
    done < "$TMPFILE"
else
    echo "No training logs found in $LOGDIR"
fi

echo "------------------------------------------------------------------------------------------------------------------------"

# Summary counts
if [ -s "$TMPFILE" ]; then
    total=$(wc -l < "$TMPFILE" | tr -d ' ')
    completed=$(grep -c "completed" "$TMPFILE" 2>/dev/null) || completed=0
    running=$(grep -c "running" "$TMPFILE" 2>/dev/null) || running=0
    oom=$(grep -c "OOM" "$TMPFILE" 2>/dev/null) || oom=0
    loading=$(grep -cE "loading|warmup" "$TMPFILE" 2>/dev/null) || loading=0

    echo ""
    echo "Summary: Total=$total | Running=$running | Completed=$completed | Loading/Warmup=$loading | OOM=$oom"
fi

# GPU status
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | \
    while read line; do echo "  $line"; done

echo ""

rm -f "$TMPFILE" 2>/dev/null
