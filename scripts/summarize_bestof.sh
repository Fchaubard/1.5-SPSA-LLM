#!/bin/bash

# Summarize BEST val and test accuracy over all training iterations
# Parses entire log history to find peak performance

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

    # Extract current iteration
    iter_matches = re.findall(
        r'Iter\s+(\d+)\s*\|\s*Loss:\s*([0-9.]+)\s*\|\s*Acc:\s*([0-9.]+)%',
        content
    )
    current_iter = 0
    current_train_acc = None
    if iter_matches:
        last = iter_matches[-1]
        current_iter = int(last[0])
        current_train_acc = float(last[2])

    # Find BEST train accuracy (from all iterations)
    best_train_acc = 0.0
    for match in iter_matches:
        acc = float(match[2])
        if acc > best_train_acc:
            best_train_acc = acc

    # Find BEST val and test accuracy from all eval outputs
    # Format: >>> [Eval] Val: 0.9240 (92.4%) | Test: 0.9100 (91.0%)
    eval_matches = re.findall(
        r'\[Eval\] Val:\s*([0-9.]+).*?\| Test:\s*([0-9.]+)',
        content
    )

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_val_iter = 0
    best_test_iter = 0

    # Also track which iteration each best occurred at
    eval_with_iter = re.findall(
        r'Iter\s+(\d+).*?(?:\n.*?)*?\[Eval\] Val:\s*([0-9.]+).*?\| Test:\s*([0-9.]+)',
        content
    )

    # Simple approach: just find max values
    for match in eval_matches:
        val_acc = float(match[0])
        test_acc = float(match[1])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if test_acc > best_test_acc:
            best_test_acc = test_acc

    # Determine status
    status = "running"
    if 'OutOfMemoryError' in content or 'CUDA out of memory' in content:
        status = "OOM"
    elif current_iter >= (max_iters or 9999) - 1:
        status = "completed"
    elif 'TRAINING' not in content:
        status = "loading"

    # Only add if we have basic info
    if n_perts and batch_size:
        results.append({
            'name': name,
            'n_perts': n_perts,
            'batch_size': batch_size,
            'lr': lr or '-',
            'max_iters': max_iters or 0,
            'current_iter': current_iter,
            'best_train_acc': best_train_acc,
            'best_val_acc': best_val_acc,
            'best_test_acc': best_test_acc,
            'status': status
        })

# Sort by best_test_acc (highest first)
def sort_key(x):
    return x['best_test_acc']

results.sort(key=sort_key, reverse=True)

# Write to output file
with open(outfile, 'w') as f:
    for r in results:
        best_train_str = f"{r['best_train_acc']:.1f}%" if r['best_train_acc'] > 0 else '-'
        best_val_str = f"{r['best_val_acc']*100:.1f}%" if r['best_val_acc'] > 0 else '-'
        best_test_str = f"{r['best_test_acc']*100:.1f}%" if r['best_test_acc'] > 0 else '-'
        progress = f"{r['current_iter']}/{r['max_iters']}"
        f.write(f"{r['n_perts']}|{r['batch_size']}|{r['lr']}|{progress}|{best_train_str}|{best_val_str}|{best_test_str}|{r['status']}|{r['name']}\n")
PYEOF

echo ""
echo "========================================================================================================================"
echo "                         1.5-SPSA BEST OF TRAINING (sorted by Best Test Acc)"
echo "========================================================================================================================"
printf "%-7s | %-5s | %-8s | %-11s | %-10s | %-10s | %-10s | %-10s | %s\n" \
    "n_perts" "batch" "lr" "progress" "best_train" "best_val" "best_test" "status" "log"
echo "------------------------------------------------------------------------------------------------------------------------"

# Display sorted results
if [ -s "$TMPFILE" ]; then
    while IFS='|' read -r n_perts batch lr progress best_train best_val best_test status name; do
        printf "%-7s | %-5s | %-8s | %-11s | %-10s | %-10s | %-10s | %-10s | %s\n" \
            "$n_perts" "$batch" "$lr" "$progress" "$best_train" "$best_val" "$best_test" "$status" "$name"
    done < "$TMPFILE"
else
    echo "No training logs found in $LOGDIR"
fi

echo "------------------------------------------------------------------------------------------------------------------------"

# Summary
if [ -s "$TMPFILE" ]; then
    # Find overall best
    best_test=$(head -1 "$TMPFILE" | cut -d'|' -f7)
    best_job=$(head -1 "$TMPFILE" | cut -d'|' -f9)
    echo ""
    echo "OVERALL BEST TEST: $best_test ($best_job)"
    echo ""
    echo "MeZO baseline: 91.4% test accuracy on SST-2"
fi

rm -f "$TMPFILE" 2>/dev/null
