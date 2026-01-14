#!/usr/bin/env python3
"""
Train OPT-13B with SPSA - Adaptive LR/Epsilon search.

Implements two search strategies:
1. Binary search: Test 3 points, halve search space, repeat
2. Quadratic fit: Use quadratic interpolation to find optimal point
"""

import argparse
import torch
import triton
import triton.language as tl
import time
import math
from datetime import datetime


def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


@triton.jit
def _unpack_and_apply(w_ptr, packed_ptr, n_elements, alpha, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    w = tl.load(w_ptr + offsets, mask=mask)
    byte_idx = offsets // 8
    bit_idx = offsets % 8
    packed_byte = tl.load(packed_ptr + byte_idx, mask=mask)
    bit = (packed_byte >> bit_idx) & 1
    sign = tl.where(bit == 1, 1.0, -1.0)
    tl.store(w_ptr + offsets, w + alpha * sign, mask=mask)


@triton.jit
def _unpack_and_accumulate(grad_ptr, packed_ptr, n_elements, coeff, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    byte_idx = offsets // 8
    bit_idx = offsets % 8
    packed_byte = tl.load(packed_ptr + byte_idx, mask=mask)
    bit = (packed_byte >> bit_idx) & 1
    sign = tl.where(bit == 1, 1.0, -1.0)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    tl.store(grad_ptr + offsets, grad + coeff * sign, mask=mask)


class SPSATrainer:
    def __init__(self, model, lr=1e-4, epsilon=1e-4, n_perts=40,
                 use_curvature=False, saturating_alpha=0.1, lambda_reg=1.0,
                 memory_efficient=False,
                 accum_steps=1):
        self.lr = lr
        self.epsilon = epsilon
        self.n_perts = n_perts
        self.use_curvature = use_curvature
        self.saturating_alpha = saturating_alpha
        self.lambda_reg = lambda_reg
        self.memory_efficient = memory_efficient
        self.accum_steps = accum_steps

        self.params = [p for p in model.parameters() if p.requires_grad]
        self.total = sum(p.numel() for p in self.params)
        self.packed_size = (self.total + 7) // 8

        # Compute offsets
        self.param_info = []
        offset = 0
        for p in self.params:
            numel = p.numel()
            self.param_info.append({
                'param': p,
                'offset': offset,
                'packed_offset': offset // 8,
                'numel': numel,
                'grid': ((numel + 1023) // 1024,),
            })
            offset += numel

        # Gradient accumulator per param (bf16) - skip if memory_efficient
        if not memory_efficient:
            self.grads = [torch.zeros(info['numel'], device='cuda', dtype=torch.bfloat16)
                          for info in self.param_info]
        else:
            self.grads = None

        mode_str = " [MEMORY EFFICIENT]" if memory_efficient else ""
        accum_str = f", accum={accum_steps}" if accum_steps > 1 else ""
        log(f"SPSATrainer: {self.total/1e9:.2f}B params, packed={self.packed_size/1e6:.0f}MB{mode_str}{accum_str}")

    def probe_loss_at_lr(self, loss_fn, test_lr, seed=0, accum_batches=1):
        """
        Probe the loss after taking 1 SPSA step with test_lr.
        Does NOT update weights - just computes what the loss would be.

        accum_batches: Number of batches to accumulate for more stable loss estimate.
        """
        if self.memory_efficient:
            return self._probe_loss_at_lr_memory_efficient(loss_fn, test_lr, seed, accum_batches)

        test_eps = test_lr  # Tied

        # Zero out gradient accumulators
        for g in self.grads:
            g.zero_()

        # For 1.5-SPSA, get clean loss
        if self.use_curvature:
            loss_clean = loss_fn()

        # Do n_perts perturbations to estimate gradient
        for pert_idx in range(self.n_perts):
            torch.manual_seed(seed * 10000 + pert_idx)
            packed = torch.randint(0, 256, (self.packed_size,), device='cuda', dtype=torch.uint8)

            # Apply +epsilon
            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], test_eps, BLOCK_SIZE=1024)

            # Accumulate loss_plus over multiple batches for stability
            loss_plus = 0.0
            for _ in range(accum_batches):
                loss_plus += loss_fn()
            loss_plus /= accum_batches

            # Apply -2*epsilon
            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], -2*test_eps, BLOCK_SIZE=1024)

            # Accumulate loss_minus over multiple batches for stability
            loss_minus = 0.0
            for _ in range(accum_batches):
                loss_minus += loss_fn()
            loss_minus /= accum_batches

            # Restore to original (apply +epsilon to undo the -2*epsilon)
            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], test_eps, BLOCK_SIZE=1024)

            # Compute gradient coefficient
            if self.use_curvature:
                curv = abs(loss_plus - 2*loss_clean + loss_minus) / (test_eps ** 2)
                curvature = max(curv ** self.saturating_alpha, self.lambda_reg)
                grad_coeff = (loss_plus - loss_minus) / (2 * test_eps * self.n_perts * curvature)
            else:
                grad_coeff = (loss_plus - loss_minus) / (2 * test_eps * self.n_perts)

            # Accumulate gradient
            for i, info in enumerate(self.param_info):
                _unpack_and_accumulate[info['grid']](
                    self.grads[i], packed[info['packed_offset']:],
                    info['numel'], grad_coeff, BLOCK_SIZE=1024)

            del packed

        # Now apply the step temporarily to measure loss
        for info, grad in zip(self.param_info, self.grads):
            info['param'].data.view(-1).sub_(grad, alpha=test_lr)

        # Measure loss after step (accumulate for stability)
        loss_after = 0.0
        for _ in range(accum_batches):
            loss_after += loss_fn()
        loss_after /= accum_batches

        # Restore weights by adding back what we subtracted
        for info, grad in zip(self.param_info, self.grads):
            info['param'].data.view(-1).add_(grad, alpha=test_lr)

        return loss_after

    def _probe_loss_at_lr_memory_efficient(self, loss_fn, test_lr, seed=0, accum_batches=1):
        """Memory-efficient version of probe_loss_at_lr. Regenerates directions via RNG."""
        test_eps = test_lr
        grad_coeffs = []

        if self.use_curvature:
            loss_clean = loss_fn()

        for pert_idx in range(self.n_perts):
            torch.manual_seed(seed * 10000 + pert_idx)
            packed = torch.randint(0, 256, (self.packed_size,), device='cuda', dtype=torch.uint8)

            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], test_eps, BLOCK_SIZE=1024)

            loss_plus = 0.0
            for _ in range(accum_batches):
                loss_plus += loss_fn()
            loss_plus /= accum_batches

            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], -2*test_eps, BLOCK_SIZE=1024)

            loss_minus = 0.0
            for _ in range(accum_batches):
                loss_minus += loss_fn()
            loss_minus /= accum_batches

            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], test_eps, BLOCK_SIZE=1024)

            if self.use_curvature:
                curv = abs(loss_plus - 2*loss_clean + loss_minus) / (test_eps ** 2)
                curvature = max(curv ** self.saturating_alpha, self.lambda_reg)
                grad_coeff = (loss_plus - loss_minus) / (2 * test_eps * self.n_perts * curvature)
            else:
                grad_coeff = (loss_plus - loss_minus) / (2 * test_eps * self.n_perts)

            grad_coeffs.append(grad_coeff)
            del packed

        for pert_idx, grad_coeff in enumerate(grad_coeffs):
            torch.manual_seed(seed * 10000 + pert_idx)
            packed = torch.randint(0, 256, (self.packed_size,), device='cuda', dtype=torch.uint8)
            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], -test_lr * grad_coeff, BLOCK_SIZE=1024)
            del packed

        loss_after = 0.0
        for _ in range(accum_batches):
            loss_after += loss_fn()
        loss_after /= accum_batches

        for pert_idx, grad_coeff in enumerate(grad_coeffs):
            torch.manual_seed(seed * 10000 + pert_idx)
            packed = torch.randint(0, 256, (self.packed_size,), device='cuda', dtype=torch.uint8)
            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], test_lr * grad_coeff, BLOCK_SIZE=1024)
            del packed

        return loss_after

    def line_search_lr(self, loss_fn, lr_min, lr_max, n_points=20, seed=0, n_seeds=10, resample_batch_fn=None, explicit_lrs=None):
        """
        Line search for optimal lr in log space.

        Tests n_points evenly spaced in log space from lr_min to lr_max.
        Each point is evaluated n_seeds times with different batches and perturbations.

        Args:
            loss_fn: Function to compute loss
            lr_min: Minimum LR to test (e.g., 1e-10)
            lr_max: Maximum LR to test (e.g., 1e-1)
            n_points: Number of points to test along the line
            seed: Base seed for reproducibility
            n_seeds: Number of seeds to average over for each LR (reduces variance)
            resample_batch_fn: Optional callback(seed) to resample batch
            explicit_lrs: Optional list of explicit LR values to test (overrides lr_min/lr_max/n_points)

        Returns best lr found.
        """
        import numpy as np

        if explicit_lrs is not None:
            # Use explicit LR values
            lrs = explicit_lrs
            log(f"  Line search: testing {len(lrs)} explicit points {[f'{x:.0e}' for x in lrs]} (n_seeds={n_seeds})")
        else:
            # Generate evenly spaced points in log space
            log_min = math.log10(lr_min)
            log_max = math.log10(lr_max)
            log_lrs = np.linspace(log_min, log_max, n_points)
            lrs = [10 ** log_lr for log_lr in log_lrs]
            log(f"  Line search: testing {n_points} points in [{lr_min:.0e}, {lr_max:.0e}] (n_seeds={n_seeds})")

        results = []
        for i, lr in enumerate(lrs):

            # Average over n_seeds, each with different batch and perturbations
            total_loss = 0.0
            for s in range(n_seeds):
                # Each seed is unique: combines point index, seed index, and base seed
                probe_seed = seed + i * 100000 + s * 1000

                # Resample batch with this seed
                if resample_batch_fn is not None:
                    resample_batch_fn(probe_seed)

                # Probe loss (perturbations will use probe_seed internally)
                loss = self.probe_loss_at_lr(loss_fn, lr, probe_seed, accum_batches=1)
                total_loss += loss

            avg_loss = total_loss / n_seeds
            results.append((lr, avg_loss))
            log(f"    [{i+1}/{len(lrs)}] lr={lr:.2e} -> loss={avg_loss:.4f}")

        # Find best
        best_lr, best_loss = min(results, key=lambda x: x[1])
        log(f"  Line search complete: best lr={best_lr:.2e} (loss={best_loss:.4f})")
        return best_lr

    def binary_search_lr(self, loss_fn, lr_min, lr_max, depth=3, seed=0, accum_batches=1, n_seeds=3, resample_batch_fn=None):
        """
        Binary search for optimal lr in log space. (DEPRECATED - use line_search_lr instead)
        """
        # Just use line search with fewer points as a fallback
        return self.line_search_lr(loss_fn, lr_min, lr_max, n_points=5, seed=seed, n_seeds=n_seeds, resample_batch_fn=resample_batch_fn)

    def local_search_lr(self, loss_fn, current_lr, seed=0, n_seeds=10, resample_batch_fn=None):
        """
        Local search: only test current_lr * 10 and current_lr / 10.
        Returns (best_lr, improved) where improved is True if we found something better.
        """
        candidates = [current_lr / 10, current_lr, current_lr * 10]
        log(f"  Local search: testing {current_lr/10:.1e}, {current_lr:.1e}, {current_lr*10:.1e} (n_seeds={n_seeds})")

        results = []
        for i, lr in enumerate(candidates):
            total_loss = 0.0
            for s in range(n_seeds):
                probe_seed = seed + i * 100000 + s * 1000
                if resample_batch_fn is not None:
                    resample_batch_fn(probe_seed)
                loss = self.probe_loss_at_lr(loss_fn, lr, probe_seed, accum_batches=1)
                total_loss += loss
            avg_loss = total_loss / n_seeds
            results.append((lr, avg_loss))
            log(f"    lr={lr:.2e} -> loss={avg_loss:.4f}")

        best_lr, best_loss = min(results, key=lambda x: x[1])
        current_loss = results[1][1]  # Middle candidate is current_lr

        improved = best_lr != current_lr
        if improved:
            log(f"  Local search: moving from {current_lr:.2e} to {best_lr:.2e} (loss {current_loss:.4f} -> {best_loss:.4f})")
        else:
            log(f"  Local search: staying at {current_lr:.2e} (already best)")

        return best_lr, improved

    def quadratic_search_lr(self, loss_fn, lr_min, lr_max, depth=3, seed=0, accum_batches=1):
        """
        Quadratic fit search for optimal lr in log space.

        Uses quadratic interpolation to predict optimal point.
        accum_batches: Number of batches to accumulate for more stable loss estimate.
        """
        log_min = math.log10(lr_min)
        log_max = math.log10(lr_max)

        # Depth 1: Test 3 points
        log_mid = (log_min + log_max) / 2
        points = [log_min, log_mid, log_max]
        results = []

        log(f"  Quadratic search depth 1: testing {len(points)} points in [{lr_min:.0e}, {lr_max:.0e}] (accum={accum_batches})")
        for log_lr in points:
            lr = 10 ** log_lr
            loss = self.probe_loss_at_lr(loss_fn, lr, seed, accum_batches=accum_batches)
            results.append((log_lr, loss))
            log(f"    lr={lr:.2e} -> loss={loss:.4f}")

        tested_lrs = set(log_lr for log_lr, _ in results)

        for d in range(2, depth + 1):
            # Fit quadratic to the 3 best points
            results.sort(key=lambda x: x[1])
            best_3 = results[:3]
            best_3.sort(key=lambda x: x[0])  # Sort by x for fitting

            x1, y1 = best_3[0]
            x2, y2 = best_3[1]
            x3, y3 = best_3[2]

            # Quadratic fit: find vertex of parabola through 3 points
            # vertex x = x2 - 0.5 * ((x2-x1)^2*(y2-y3) - (x2-x3)^2*(y2-y1)) /
            #                       ((x2-x1)*(y2-y3) - (x2-x3)*(y2-y1))
            denom = (x2 - x1) * (y2 - y3) - (x2 - x3) * (y2 - y1)

            if abs(denom) < 1e-10:
                # Fallback to midpoint of best region
                log_next = (x1 + x3) / 2
            else:
                numer = (x2 - x1) ** 2 * (y2 - y3) - (x2 - x3) ** 2 * (y2 - y1)
                log_next = x2 - 0.5 * numer / denom

                # Clamp to search range
                log_next = max(log_min, min(log_max, log_next))

            # Avoid retesting same lr (with tolerance)
            min_dist = min(abs(log_next - t) for t in tested_lrs)
            if min_dist < 0.01:  # Too close to already-tested point
                # Try midpoint between two best points instead
                log_next = (best_3[0][0] + best_3[1][0]) / 2
                if min(abs(log_next - t) for t in tested_lrs) < 0.01:
                    # Still too close, try other midpoint
                    log_next = (best_3[1][0] + best_3[2][0]) / 2

            lr = 10 ** log_next
            loss = self.probe_loss_at_lr(loss_fn, lr, seed, accum_batches=accum_batches)
            log(f"  Quadratic search depth {d}: lr={lr:.2e} -> loss={loss:.4f}")

            results.append((log_next, loss))
            tested_lrs.add(log_next)

        best_log_lr, best_loss = min(results, key=lambda x: x[1])
        best_lr = 10 ** best_log_lr
        log(f"  Quadratic search complete: best lr={best_lr:.2e} (loss={best_loss:.4f})")
        return best_lr

    def step(self, loss_fn, iteration):
        if self.memory_efficient:
            return self._step_memory_efficient(loss_fn, iteration)

        for g in self.grads:
            g.zero_()

        total_loss = 0.0

        # For 1.5-SPSA, get clean loss once per iteration
        if self.use_curvature:
            loss_clean = 0.0
            for batch_idx in range(self.accum_steps):
                loss_clean += loss_fn(batch_idx)
            loss_clean /= self.accum_steps

        for pert_idx in range(self.n_perts):
            # Generate bit-packed random (8x less data!)
            torch.manual_seed(iteration * 10000 + pert_idx)
            packed = torch.randint(0, 256, (self.packed_size,), device='cuda', dtype=torch.uint8)

            # Apply +epsilon
            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], self.epsilon, BLOCK_SIZE=1024)

            # Accumulate loss_plus over accum_steps batches (SAME batches as loss_minus)
            loss_plus = 0.0
            for batch_idx in range(self.accum_steps):
                loss_plus += loss_fn(batch_idx)
            loss_plus /= self.accum_steps

            # Apply -2*epsilon
            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], -2*self.epsilon, BLOCK_SIZE=1024)

            # Accumulate loss_minus over accum_steps batches (SAME batches as loss_plus)
            loss_minus = 0.0
            for batch_idx in range(self.accum_steps):
                loss_minus += loss_fn(batch_idx)
            loss_minus /= self.accum_steps

            # Restore
            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], self.epsilon, BLOCK_SIZE=1024)

            # Compute gradient coefficient
            if self.use_curvature:
                # 1.5-SPSA: use curvature to scale gradient
                curv = abs(loss_plus - 2*loss_clean + loss_minus) / (self.epsilon ** 2)
                curvature = max(curv ** self.saturating_alpha, self.lambda_reg)
                grad_coeff = (loss_plus - loss_minus) / (2 * self.epsilon * self.n_perts * curvature)
            else:
                # Standard SPSA
                grad_coeff = (loss_plus - loss_minus) / (2 * self.epsilon * self.n_perts)

            # Accumulate gradient
            for i, info in enumerate(self.param_info):
                _unpack_and_accumulate[info['grid']](
                    self.grads[i], packed[info['packed_offset']:],
                    info['numel'], grad_coeff, BLOCK_SIZE=1024)

            total_loss += (loss_plus + loss_minus) / 2
            del packed

        # Apply update
        for info, grad in zip(self.param_info, self.grads):
            info['param'].data.view(-1).sub_(grad, alpha=self.lr)

        return total_loss / self.n_perts

    def _step_memory_efficient(self, loss_fn, iteration):
        """Memory-efficient step. Regenerates directions via RNG instead of caching grads."""
        total_loss = 0.0
        grad_coeffs = []

        if self.use_curvature:
            loss_clean = 0.0
            for batch_idx in range(self.accum_steps):
                loss_clean += loss_fn(batch_idx)
            loss_clean /= self.accum_steps

        for pert_idx in range(self.n_perts):
            torch.manual_seed(iteration * 10000 + pert_idx)
            packed = torch.randint(0, 256, (self.packed_size,), device='cuda', dtype=torch.uint8)

            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], self.epsilon, BLOCK_SIZE=1024)

            # Accumulate loss_plus over accum_steps batches (SAME batches as loss_minus)
            loss_plus = 0.0
            for batch_idx in range(self.accum_steps):
                loss_plus += loss_fn(batch_idx)
            loss_plus /= self.accum_steps

            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], -2*self.epsilon, BLOCK_SIZE=1024)

            # Accumulate loss_minus over accum_steps batches (SAME batches as loss_plus)
            loss_minus = 0.0
            for batch_idx in range(self.accum_steps):
                loss_minus += loss_fn(batch_idx)
            loss_minus /= self.accum_steps

            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], self.epsilon, BLOCK_SIZE=1024)

            if self.use_curvature:
                curv = abs(loss_plus - 2*loss_clean + loss_minus) / (self.epsilon ** 2)
                curvature = max(curv ** self.saturating_alpha, self.lambda_reg)
                grad_coeff = (loss_plus - loss_minus) / (2 * self.epsilon * self.n_perts * curvature)
            else:
                grad_coeff = (loss_plus - loss_minus) / (2 * self.epsilon * self.n_perts)

            grad_coeffs.append(grad_coeff)
            total_loss += (loss_plus + loss_minus) / 2
            del packed

        for pert_idx, grad_coeff in enumerate(grad_coeffs):
            torch.manual_seed(iteration * 10000 + pert_idx)
            packed = torch.randint(0, 256, (self.packed_size,), device='cuda', dtype=torch.uint8)

            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], -self.lr * grad_coeff, BLOCK_SIZE=1024)

            del packed

        return total_loss / self.n_perts


class RandomSearchTrainer:
    """
    Random Search optimizer for LLMs.

    Instead of estimating gradients like SPSA, this performs random search:
    - Try n_perts random directions (Rademacher distributed)
    - Jump epsilon distance in each direction and measure loss
    - Keep the best direction (lowest loss) if it improves on current
    - Never accept a worse position
    """
    def __init__(self, model, epsilon=1e-4, n_perts=40):
        self.epsilon = epsilon
        self.n_perts = n_perts
        self.lr = epsilon  # For compatibility with line search (lr=eps always)

        self.params = [p for p in model.parameters() if p.requires_grad]
        self.total = sum(p.numel() for p in self.params)
        self.packed_size = (self.total + 7) // 8

        # Compute offsets (same as SPSA)
        self.param_info = []
        offset = 0
        for p in self.params:
            numel = p.numel()
            self.param_info.append({
                'param': p,
                'offset': offset,
                'packed_offset': offset // 8,
                'numel': numel,
                'grid': ((numel + 1023) // 1024,),
            })
            offset += numel

        # Track best loss for greedy acceptance
        self.best_loss = float('inf')

        log(f"RandomSearchTrainer: {self.total/1e9:.2f}B params, packed={self.packed_size/1e6:.0f}MB")

    def step(self, loss_fn, iteration):
        """
        Perform one random search step:
        1. Try n_perts random directions
        2. Find the one with lowest loss
        3. If it beats current best, jump there permanently
        4. Otherwise, stay put

        Returns the loss after the step (best found or current).
        """
        # Get current loss as baseline
        current_loss = loss_fn()

        # Initialize tracking
        best_seed = None
        best_loss = current_loss  # Must beat current to be accepted

        # Try each perturbation
        for pert_idx in range(self.n_perts):
            seed = iteration * 10000 + pert_idx

            # Generate bit-packed random perturbation
            torch.manual_seed(seed)
            packed = torch.randint(0, 256, (self.packed_size,), device='cuda', dtype=torch.uint8)

            # Apply +epsilon perturbation
            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], self.epsilon, BLOCK_SIZE=1024)

            # Measure loss at this position
            loss = loss_fn()

            # Restore weights (apply -epsilon)
            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], -self.epsilon, BLOCK_SIZE=1024)

            # Track best
            if loss < best_loss:
                best_loss = loss
                best_seed = seed

            del packed

        # If we found an improvement, apply that perturbation permanently
        if best_seed is not None:
            torch.manual_seed(best_seed)
            packed = torch.randint(0, 256, (self.packed_size,), device='cuda', dtype=torch.uint8)

            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], self.epsilon, BLOCK_SIZE=1024)

            del packed
            self.best_loss = best_loss
            return best_loss
        else:
            # No improvement found, stay put
            return current_loss

    def line_search_lr(self, loss_fn, lr_min, lr_max, n_points=20, seed=0, n_seeds=10, resample_batch_fn=None):
        """
        Line search for optimal epsilon in log space.
        Same as SPSA's line search but for epsilon (since lr=eps for RandomSearch).
        """
        import numpy as np

        log_min = math.log10(lr_min)
        log_max = math.log10(lr_max)

        log_lrs = np.linspace(log_min, log_max, n_points)

        log(f"  Line search: testing {n_points} points in [{lr_min:.0e}, {lr_max:.0e}] (n_seeds={n_seeds})")

        results = []
        for i, log_lr in enumerate(log_lrs):
            test_eps = 10 ** log_lr

            # Average over n_seeds
            total_loss = 0.0
            for s in range(n_seeds):
                probe_seed = seed + i * 100000 + s * 1000

                if resample_batch_fn is not None:
                    resample_batch_fn(probe_seed)

                # Probe: try one random jump with test_eps
                loss = self._probe_at_epsilon(loss_fn, test_eps, probe_seed)
                total_loss += loss

            avg_loss = total_loss / n_seeds
            results.append((test_eps, avg_loss))
            log(f"    [{i+1}/{n_points}] eps={test_eps:.2e} -> loss={avg_loss:.4f}")

        best_eps, best_loss = min(results, key=lambda x: x[1])
        log(f"  Line search complete: best eps={best_eps:.2e} (loss={best_loss:.4f})")
        return best_eps

    def _probe_at_epsilon(self, loss_fn, test_eps, seed):
        """
        Probe: try n_perts random directions with test_eps, return best loss found.
        Does NOT permanently update weights.
        """
        current_loss = loss_fn()
        best_loss = current_loss

        for pert_idx in range(self.n_perts):
            pert_seed = seed * 10000 + pert_idx

            torch.manual_seed(pert_seed)
            packed = torch.randint(0, 256, (self.packed_size,), device='cuda', dtype=torch.uint8)

            # Apply +test_eps
            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], test_eps, BLOCK_SIZE=1024)

            loss = loss_fn()

            # Restore
            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], -test_eps, BLOCK_SIZE=1024)

            if loss < best_loss:
                best_loss = loss

            del packed

        return best_loss

    def local_search_lr(self, loss_fn, current_lr, seed=0, n_seeds=10, resample_batch_fn=None):
        """
        Local search: only test current_lr * 10 and current_lr / 10.
        Returns (best_lr, improved) where improved is True if we found something better.
        """
        candidates = [current_lr / 10, current_lr, current_lr * 10]
        log(f"  Local search: testing {current_lr/10:.1e}, {current_lr:.1e}, {current_lr*10:.1e} (n_seeds={n_seeds})")

        results = []
        for i, eps in enumerate(candidates):
            total_loss = 0.0
            for s in range(n_seeds):
                probe_seed = seed + i * 100000 + s * 1000
                if resample_batch_fn is not None:
                    resample_batch_fn(probe_seed)
                loss = self._probe_at_epsilon(loss_fn, eps, probe_seed)
                total_loss += loss
            avg_loss = total_loss / n_seeds
            results.append((eps, avg_loss))
            log(f"    eps={eps:.2e} -> loss={avg_loss:.4f}")

        best_eps, best_loss = min(results, key=lambda x: x[1])
        current_loss = results[1][1]  # Middle candidate is current_lr

        improved = best_eps != current_lr
        if improved:
            log(f"  Local search: moving from {current_lr:.2e} to {best_eps:.2e} (loss {current_loss:.4f} -> {best_loss:.4f})")
        else:
            log(f"  Local search: staying at {current_lr:.2e} (already best)")

        return best_eps, improved


class MeZOTrainer:
    """
    MeZO (Memory-Efficient ZerO-order) Optimizer.

    This implements the MeZO baseline from the paper:
    "Fine-Tuning Language Models with Just Forward Passes"

    MeZO is essentially 1-SPSA with:
    - Single perturbation per step (n_perts=1)
    - In-place perturbation to save memory
    - Same random seed used for +/- perturbations
    """
    def __init__(self, model, lr=1e-5, epsilon=1e-3):
        self.lr = lr
        self.epsilon = epsilon

        self.params = [p for p in model.parameters() if p.requires_grad]
        self.total = sum(p.numel() for p in self.params)
        self.packed_size = (self.total + 7) // 8

        # Compute offsets
        self.param_info = []
        offset = 0
        for p in self.params:
            numel = p.numel()
            self.param_info.append({
                'param': p,
                'offset': offset,
                'packed_offset': offset // 8,
                'numel': numel,
                'grid': ((numel + 1023) // 1024,),
            })
            offset += numel

        log(f"MeZOTrainer: {self.total/1e9:.2f}B params, packed={self.packed_size/1e6:.0f}MB")

    def step(self, loss_fn, iteration):
        """
        Perform one MeZO step with a single perturbation.
        """
        # Generate random perturbation
        torch.manual_seed(iteration)
        packed = torch.randint(0, 256, (self.packed_size,), device='cuda', dtype=torch.uint8)

        # Apply +epsilon
        for info in self.param_info:
            flat = info['param'].data.view(-1)
            _unpack_and_apply[info['grid']](
                flat, packed[info['packed_offset']:],
                info['numel'], self.epsilon, BLOCK_SIZE=1024)

        loss_plus = loss_fn()

        # Apply -2*epsilon (to get to -epsilon from +epsilon)
        for info in self.param_info:
            flat = info['param'].data.view(-1)
            _unpack_and_apply[info['grid']](
                flat, packed[info['packed_offset']:],
                info['numel'], -2*self.epsilon, BLOCK_SIZE=1024)

        loss_minus = loss_fn()

        # Compute projected gradient and apply update
        # Instead of restoring and then updating, we directly compute the final position
        # Current position: theta - epsilon*z
        # Want: theta - lr * g * z where g = (loss_plus - loss_minus) / (2*epsilon)
        # So we need to apply: +epsilon*z - lr*g*z = (epsilon - lr*g)*z
        grad = (loss_plus - loss_minus) / (2 * self.epsilon)
        update_scale = self.epsilon - self.lr * grad

        for info in self.param_info:
            flat = info['param'].data.view(-1)
            _unpack_and_apply[info['grid']](
                flat, packed[info['packed_offset']:],
                info['numel'], update_scale, BLOCK_SIZE=1024)

        del packed
        return (loss_plus + loss_minus) / 2

    def line_search_lr(self, loss_fn, lr_min, lr_max, n_points=20, seed=0, n_seeds=10, resample_batch_fn=None, explicit_lrs=None):
        """Line search for optimal lr."""
        import numpy as np

        if explicit_lrs is not None:
            lrs = explicit_lrs
        else:
            log_min = math.log10(lr_min)
            log_max = math.log10(lr_max)
            log_lrs = np.linspace(log_min, log_max, n_points)
            lrs = [10 ** log_lr for log_lr in log_lrs]

        log(f"  Line search: testing {len(lrs)} points (n_seeds={n_seeds})")

        results = []
        for i, lr in enumerate(lrs):
            total_loss = 0.0
            for s in range(n_seeds):
                probe_seed = seed + i * 100000 + s * 1000
                if resample_batch_fn is not None:
                    resample_batch_fn(probe_seed)
                loss = self._probe_at_lr(loss_fn, lr, probe_seed)
                total_loss += loss
            avg_loss = total_loss / n_seeds
            results.append((lr, avg_loss))
            log(f"    [{i+1}/{len(lrs)}] lr={lr:.2e} -> loss={avg_loss:.4f}")

        best_lr, best_loss = min(results, key=lambda x: x[1])
        log(f"  Line search complete: best lr={best_lr:.2e} (loss={best_loss:.4f})")
        return best_lr

    def _probe_at_lr(self, loss_fn, test_lr, seed):
        """Probe loss after one MeZO step with test_lr."""
        torch.manual_seed(seed)
        packed = torch.randint(0, 256, (self.packed_size,), device='cuda', dtype=torch.uint8)

        # Apply +epsilon
        for info in self.param_info:
            flat = info['param'].data.view(-1)
            _unpack_and_apply[info['grid']](
                flat, packed[info['packed_offset']:],
                info['numel'], self.epsilon, BLOCK_SIZE=1024)

        loss_plus = loss_fn()

        # Apply -2*epsilon
        for info in self.param_info:
            flat = info['param'].data.view(-1)
            _unpack_and_apply[info['grid']](
                flat, packed[info['packed_offset']:],
                info['numel'], -2*self.epsilon, BLOCK_SIZE=1024)

        loss_minus = loss_fn()

        # Restore to original (+epsilon)
        for info in self.param_info:
            flat = info['param'].data.view(-1)
            _unpack_and_apply[info['grid']](
                flat, packed[info['packed_offset']:],
                info['numel'], self.epsilon, BLOCK_SIZE=1024)

        del packed
        return (loss_plus + loss_minus) / 2

    def local_search_lr(self, loss_fn, current_lr, seed=0, n_seeds=10, resample_batch_fn=None):
        """Local search: only test current_lr * 10 and current_lr / 10."""
        candidates = [current_lr / 10, current_lr, current_lr * 10]
        log(f"  Local search: testing {current_lr/10:.1e}, {current_lr:.1e}, {current_lr*10:.1e}")

        results = []
        for i, lr in enumerate(candidates):
            total_loss = 0.0
            for s in range(n_seeds):
                probe_seed = seed + i * 100000 + s * 1000
                if resample_batch_fn is not None:
                    resample_batch_fn(probe_seed)
                loss = self._probe_at_lr(loss_fn, lr, probe_seed)
                total_loss += loss
            avg_loss = total_loss / n_seeds
            results.append((lr, avg_loss))

        best_lr, best_loss = min(results, key=lambda x: x[1])
        improved = best_lr != current_lr
        return best_lr, improved


class BackpropTrainer:
    """
    Standard backpropagation trainer for comparison.

    Uses AdamW optimizer with gradient accumulation.
    """
    def __init__(self, model, lr=1e-5, accum_steps=1):
        self.lr = lr
        self.accum_steps = accum_steps
        self.model = model

        # Enable gradients for backprop
        for p in model.parameters():
            p.requires_grad = True

        # Use AdamW optimizer
        from torch.optim import AdamW
        self.optimizer = AdamW(model.parameters(), lr=lr)

        self.total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"BackpropTrainer: {self.total/1e9:.2f}B params, lr={lr}, accum_steps={accum_steps}")

        self._accum_count = 0
        self._accum_loss = 0.0

    def step(self, loss_fn_with_backward, iteration):
        """
        Perform one backprop step with gradient accumulation.

        Note: loss_fn_with_backward should return the loss tensor (not .item())
        and the loss should have requires_grad=True.
        """
        # Get loss and do backward
        loss = loss_fn_with_backward()
        scaled_loss = loss / self.accum_steps
        scaled_loss.backward()

        self._accum_loss += loss.item()
        self._accum_count += 1

        # Update weights after accumulation
        if self._accum_count >= self.accum_steps:
            self.optimizer.step()
            self.optimizer.zero_grad()
            avg_loss = self._accum_loss / self._accum_count
            self._accum_count = 0
            self._accum_loss = 0.0
            return avg_loss

        return loss.item()

    def line_search_lr(self, *args, **kwargs):
        """Not implemented for backprop - uses fixed LR."""
        log("  Line search not supported for backprop trainer")
        return self.lr

    def local_search_lr(self, *args, **kwargs):
        """Not implemented for backprop."""
        return self.lr, False


def main():
    parser = argparse.ArgumentParser(description='Train LLM with SPSA/RandomSearch - Adaptive LR Search')
    parser.add_argument('--model', type=str, default='facebook/opt-13b', help='Model name or path')
    parser.add_argument('--solver', type=str, default='spsa',
                        choices=['spsa', 'random', 'mezo', 'backprop'],
                        help='Solver: spsa (1-SPSA), mezo (MeZO baseline), random (greedy random search), or backprop')
    parser.add_argument('--task', type=str, default='static',
                        choices=['static', 'sst2', 'rte', 'boolq', 'wsc', 'wic', 'squad'],
                        help='Task: static (overfit random batch), classification task, or squad (generative)')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='spsa-llm', help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name (auto-generated if not set)')
    parser.add_argument('--accum_steps', type=int, default=1, help='Batch accumulation steps (effective_batch = batch_size * accum_steps)')
    parser.add_argument('--n_perts', type=int, default=40, help='Perturbations per iteration')
    parser.add_argument('--n_iterations', type=int, default=1000, help='Total training iterations')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epsilon', type=float, default=None, help='Epsilon for perturbations (default: same as lr)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length')
    parser.add_argument('--eval_interval', type=int, default=50, help='Evaluation interval for tasks')
    parser.add_argument('--checkpoint_interval', type=int, default=100, help='Save checkpoint every N iters')
    parser.add_argument('--use_1_5_spsa', action='store_true', help='Use 1.5-SPSA with curvature scaling')
    parser.add_argument('--saturating_alpha', type=float, default=0.1, help='Exponent for saturating curvature')
    parser.add_argument('--lambda_reg', type=float, default=1.0, help='Minimum curvature regularization')
    parser.add_argument('--memory_efficient', action='store_true', help='Memory efficient mode: regenerate directions via RNG instead of caching gradients')

    # Adaptive search options
    parser.add_argument('--search_strategy', type=str, default='none',
                        choices=['none', 'binary', 'quadratic', 'line', 'local'],
                        help='LR search strategy: local (10x up/down only, recommended), line (full range)')
    parser.add_argument('--n_points', type=int, default=20, help='Number of points for line search')
    parser.add_argument('--lr_points', type=str, default=None,
                        help='Comma-separated LR values to test (e.g., "1e-3,5e-4,1e-4,5e-5"). Overrides n_points/lr_min/lr_max.')
    parser.add_argument('--lr_min', type=float, default=1e-5, help='Min LR for search')
    parser.add_argument('--lr_max', type=float, default=1e-1, help='Max LR for search')
    parser.add_argument('--search_depth', type=int, default=3, help='Search depth')
    parser.add_argument('--probe_accum', type=int, default=1, help='Batches to accumulate during LR probing for stability')
    parser.add_argument('--n_seeds', type=int, default=3, help='Number of seeds to average over for each LR evaluation in binary search')
    parser.add_argument('--search_patience', type=int, default=20, help='Iters without improvement before re-search')
    parser.add_argument('--diverge_threshold', type=float, default=1.5, help='Loss increase ratio to trigger re-search')
    parser.add_argument('--ema_alpha', type=float, default=0.1, help='EMA smoothing factor for plateau detection')
    parser.add_argument('--static_batch', action='store_true', help='Use a static batch for classification tasks (for debugging)')
    # LR decay on plateau
    parser.add_argument('--lr_decay_patience', type=int, default=50, help='Decay LR if train acc plateaus for this many iters')
    parser.add_argument('--lr_decay_factor', type=float, default=0.5, help='Factor to multiply LR by on plateau')
    parser.add_argument('--lr_min_decay', type=float, default=1e-7, help='Minimum LR after decay')

    args = parser.parse_args()

    log("=" * 70)
    if args.solver == 'random':
        method = "RANDOM SEARCH"
    elif args.solver == 'mezo':
        method = "MeZO"
    elif args.solver == 'backprop':
        method = "BACKPROP"
    else:
        method = "1.5-SPSA" if args.use_1_5_spsa else "1-SPSA"
    search_str = f" + {args.search_strategy.upper()} search" if args.search_strategy != 'none' else ""
    task_str = f" on {args.task.upper()}" if args.task != 'static' else ""
    model_name = args.model.split('/')[-1]
    log(f"{method}{search_str} TRAINING{task_str}: {model_name} (bit-packed)")
    log("=" * 70)

    # Initialize W&B if enabled
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            run_name = args.wandb_run_name or f"{method}_{args.task}_{model_name}_np{args.n_perts}_bs{args.batch_size}"
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=vars(args),
            )
            log(f"W&B initialized: {wandb_run.url}")
        except Exception as e:
            log(f"W&B initialization failed: {e}")
            wandb_run = None

    n_perts = args.n_perts
    n_iterations = args.n_iterations
    lr = args.lr
    epsilon = args.epsilon if args.epsilon is not None else args.lr

    # Print all hyperparameters
    log(f"Hyperparameters:")
    log(f"  solver: {args.solver}")
    log(f"  n_perts: {n_perts}")
    log(f"  n_iterations: {n_iterations}")
    log(f"  lr: {lr}")
    log(f"  epsilon: {epsilon}")
    log(f"  batch_size: {args.batch_size}")
    log(f"  seq_len: {args.seq_len}")
    log(f"  model: {args.model}")
    if args.task != 'static':
        log(f"  task: {args.task}")
        log(f"  eval_interval: {args.eval_interval}")
    if args.search_strategy != 'none':
        log(f"  search_strategy: {args.search_strategy}")
        log(f"  lr_min: {args.lr_min}")
        log(f"  lr_max: {args.lr_max}")
        if args.search_strategy == 'line':
            log(f"  n_points: {args.n_points}")
        else:
            log(f"  search_depth: {args.search_depth}")
        log(f"  n_seeds: {args.n_seeds} (avg over seeds for stable LR eval)")
        log(f"  search_patience: {args.search_patience}")
        log(f"  diverge_threshold: {args.diverge_threshold}")
    if args.use_1_5_spsa:
        log(f"  1.5-SPSA: saturating_alpha={args.saturating_alpha}, lambda_reg={args.lambda_reg}")
    if args.task != 'static':
        log(f"  lr_decay_patience: {args.lr_decay_patience} (decay if train acc plateaus)")
        log(f"  lr_decay_factor: {args.lr_decay_factor}")
        log(f"  lr_min_decay: {args.lr_min_decay}")

    log("Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map='cuda')
    for p in model.parameters():
        p.requires_grad = True
    log(f"Loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

    # CRITICAL: Set to eval mode to disable dropout
    # Dropout causes different outputs each forward pass, breaking SPSA gradient estimation
    model.eval()

    # Task-specific setup
    dataset = None
    evaluate_fn = None
    resample_probe_batch = None  # Will be defined for classification tasks

    quick_train_acc = None  # Will be defined for classification tasks

    if args.task == 'static':
        log("Creating static batch...")
        torch.manual_seed(42)
        input_ids = torch.randint(0, model.config.vocab_size, (args.batch_size, args.seq_len), device='cuda')
        labels = input_ids.clone()

        def loss_fn():
            with torch.no_grad():
                return model(input_ids=input_ids, labels=labels).loss.item()

        # For static task, probe and train use the same loss function
        probe_loss_fn = loss_fn
    else:
        # Load tokenizer and task dataset
        log(f"Loading tokenizer and {args.task} dataset...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        from tasks.tasks_llm import get_task_dataset
        dataset = get_task_dataset(
            task_name=args.task,
            tokenizer=tokenizer,
            max_length=args.seq_len,
            seed=42,
            cache_dir='./data_cache',
        )

        import torch.nn.functional as F

        # Check if this is a generative task (like SQuAD)
        is_generative = dataset.task_type == "generative"

        if is_generative:
            # Generative task (SQuAD) - use language modeling loss
            def get_batch(split='train'):
                """Get a random batch from the dataset for generative task."""
                if split == 'train':
                    data = dataset.train_data
                else:
                    data = dataset.val_data

                indices = torch.randint(0, len(data), (args.batch_size,)).tolist()
                batch_data = [data[i] for i in indices]

                texts = [dataset.format_example(ex) for ex in batch_data]
                answers = [dataset.get_answer(ex) for ex in batch_data]

                # Concatenate prompt + answer for training
                texts_with_answers = [f"{t} {a}" for t, a in zip(texts, answers)]

                # Get prompt lengths for loss masking
                prompt_encodings = tokenizer(
                    texts, padding="max_length", truncation=True,
                    max_length=args.seq_len, return_tensors="pt"
                )
                prompt_lengths = prompt_encodings["attention_mask"].sum(dim=1).tolist()

                # Full sequence encoding
                encodings = tokenizer(
                    texts_with_answers, padding="max_length", truncation=True,
                    max_length=args.seq_len, return_tensors="pt"
                )
                input_ids = encodings["input_ids"].to('cuda')
                attention_mask = encodings["attention_mask"].to('cuda')

                return input_ids, attention_mask, prompt_lengths

            # Create a batch for LR probing
            probe_batch = list(get_batch('train'))

            def _resample_probe_batch(seed):
                torch.manual_seed(seed)
                probe_batch[0], probe_batch[1], probe_batch[2] = get_batch('train')
            resample_probe_batch = _resample_probe_batch

            def compute_generative_loss(input_ids, attention_mask, prompt_lengths, return_tensor=False):
                """Compute LM loss on answer portion only."""
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                logits = outputs.logits

                # Shift for causal LM
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()

                batch_size, seq_len_minus_1 = shift_labels.shape

                # Create mask for answer tokens only
                answer_mask = torch.zeros(batch_size, seq_len_minus_1, device='cuda')
                for i, prompt_len in enumerate(prompt_lengths):
                    start_pos = max(0, prompt_len - 1)
                    end_pos = int(attention_mask[i].sum().item()) - 1
                    if start_pos < end_pos:
                        answer_mask[i, start_pos:end_pos] = 1.0

                # Compute loss
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='none'
                ).view(batch_size, seq_len_minus_1)

                masked_loss = (loss * answer_mask).sum() / (answer_mask.sum() + 1e-8)

                if return_tensor:
                    return masked_loss, 0.0  # No accuracy for generative
                return masked_loss.item(), 0.0

            def probe_loss_fn():
                with torch.no_grad():
                    loss, _ = compute_generative_loss(probe_batch[0], probe_batch[1], probe_batch[2])
                    return loss

            # Variables to hold current iteration's batches (for SPSA consistency)
            batch_list = []  # List of (input_ids, attention_mask, prompt_lengths) tuples

            def sample_new_batch(n_batches=1):
                """Sample n_batches for the current iteration (for gradient accumulation)."""
                batch_list.clear()
                for _ in range(n_batches):
                    batch_list.append(get_batch('train'))

            def loss_fn(batch_idx=0):
                """Loss on batch at given index (for consistent +/- perturbation evaluation)."""
                with torch.no_grad():
                    idx = batch_idx % len(batch_list) if batch_list else 0
                    batch = batch_list[idx] if batch_list else get_batch('train')
                    loss, _ = compute_generative_loss(batch[0], batch[1], batch[2])
                    return loss

            def backprop_loss_fn():
                """Loss with gradient for backprop (uses first batch in list)."""
                batch = batch_list[0] if batch_list else get_batch('train')
                loss, _ = compute_generative_loss(batch[0], batch[1], batch[2], return_tensor=True)
                return loss

            quick_train_acc = None  # No quick accuracy for generative tasks

        else:
            # Classification task
            def get_batch(split='train'):
                """Get a random batch from the dataset."""
                if split == 'train':
                    data = dataset.train_data
                else:
                    data = dataset.val_data

                indices = torch.randint(0, len(data), (args.batch_size,)).tolist()
                batch_data = [data[i] for i in indices]

                texts = [dataset.format_example(ex) for ex in batch_data]
                labels_list = [ex[dataset.config["label_column"]] for ex in batch_data]

                encodings = tokenizer(
                    texts, padding="max_length", truncation=True,
                    max_length=args.seq_len, return_tensors="pt"
                )
                input_ids = encodings["input_ids"].to('cuda')
                attention_mask = encodings["attention_mask"].to('cuda')
                labels_tensor = torch.tensor(labels_list, dtype=torch.long, device='cuda')

                return input_ids, attention_mask, labels_tensor

            # Create a batch for LR probing (will be resampled with different seeds)
            probe_batch = list(get_batch('train'))  # [input_ids, attention_mask, labels]
            label_token_ids_tensor = torch.tensor(dataset.label_token_ids, device='cuda')

            def _resample_probe_batch(seed):
                """Resample the probe batch with a given seed for reproducible variation."""
                torch.manual_seed(seed)
                probe_batch[0], probe_batch[1], probe_batch[2] = get_batch('train')
            resample_probe_batch = _resample_probe_batch  # Assign to outer scope variable

            def compute_loss_and_acc(input_ids, attention_mask, labels, return_tensor=False):
                """Compute classification loss and accuracy for a batch."""
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                batch_size = input_ids.size(0)
                seq_lengths = attention_mask.sum(dim=1) - 1
                final_logits = outputs.logits[torch.arange(batch_size, device='cuda'), seq_lengths]
                label_logits = final_logits[:, label_token_ids_tensor].float()
                loss = F.cross_entropy(label_logits, labels)
                preds = label_logits.argmax(dim=1)
                acc = (preds == labels).float().mean().item()
                if return_tensor:
                    return loss, acc
                return loss.item(), acc

            def compute_loss(input_ids, attention_mask, labels, return_tensor=False):
                """Compute classification loss for a batch (for backward compat)."""
                loss, _ = compute_loss_and_acc(input_ids, attention_mask, labels, return_tensor=return_tensor)
                return loss

            def probe_loss_fn():
                """Loss on probe batch for LR probing."""
                with torch.no_grad():
                    return compute_loss(probe_batch[0], probe_batch[1], probe_batch[2])

            # Variables to hold current iteration's batches (for SPSA consistency)
            # We store accum_steps batches so that loss_plus and loss_minus use the SAME batches
            batch_list = []  # List of (input_ids, attention_mask, labels) tuples

            def sample_new_batch(n_batches=1):
                """Sample n_batches for the current iteration (for gradient accumulation)."""
                batch_list.clear()
                for _ in range(n_batches):
                    batch_list.append(get_batch('train'))

            def loss_fn(batch_idx=0):
                """Loss on batch at given index (for consistent +/- perturbation evaluation)."""
                with torch.no_grad():
                    idx = batch_idx % len(batch_list) if batch_list else 0
                    batch = batch_list[idx] if batch_list else get_batch('train')
                    return compute_loss(batch[0], batch[1], batch[2])

            def backprop_loss_fn():
                """Loss with gradient for backprop (uses first batch in list)."""
                batch = batch_list[0] if batch_list else get_batch('train')
                return compute_loss(batch[0], batch[1], batch[2], return_tensor=True)

            def _quick_train_acc():
                """Quick accuracy on a random training batch."""
                with torch.no_grad():
                    input_ids, attention_mask, labels = get_batch('train')
                    _, acc = compute_loss_and_acc(input_ids, attention_mask, labels)
                    return acc
            quick_train_acc = _quick_train_acc  # Assign to outer variable

        if is_generative:
            # Generative evaluation (SQuAD) - use F1 score
            from tasks.tasks_llm import compute_squad_f1, normalize_answer

            def evaluate_fn(split='val'):
                """Evaluate F1 score on a dataset split for generative task."""
                if split == 'val':
                    data = dataset.val_data
                else:
                    data = dataset.test_data

                total_f1 = 0.0
                total_samples = 0

                with torch.no_grad():
                    for start_idx in range(0, len(data), args.batch_size):
                        end_idx = min(start_idx + args.batch_size, len(data))
                        batch_data = data[start_idx:end_idx]

                        texts = [dataset.format_example(ex) for ex in batch_data]
                        gold_answers = [dataset.get_answer(ex) for ex in batch_data]

                        encodings = tokenizer(
                            texts, padding="max_length", truncation=True,
                            max_length=args.seq_len, return_tensors="pt"
                        )
                        input_ids = encodings["input_ids"].to('cuda')
                        attention_mask = encodings["attention_mask"].to('cuda')

                        # Generate predictions
                        outputs = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=50,
                            num_beams=1,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                        )

                        # Decode predictions (only new tokens)
                        for i, (output, gold) in enumerate(zip(outputs, gold_answers)):
                            prompt_len = input_ids[i].shape[0]
                            pred_tokens = output[prompt_len:]
                            pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()

                            # Compute F1
                            f1 = compute_squad_f1(pred_text, gold)
                            total_f1 += f1
                            total_samples += 1

                return total_f1 / total_samples if total_samples > 0 else 0.0
        else:
            # Classification evaluation
            def evaluate_fn(split='val'):
                """Evaluate accuracy on a dataset split."""
                if split == 'val':
                    data = dataset.val_data
                else:
                    data = dataset.test_data

                total_correct = 0
                total_samples = 0

                with torch.no_grad():
                    for start_idx in range(0, len(data), args.batch_size):
                        end_idx = min(start_idx + args.batch_size, len(data))
                        batch_data = data[start_idx:end_idx]

                        texts = [dataset.format_example(ex) for ex in batch_data]
                        labels_list = [ex[dataset.config["label_column"]] for ex in batch_data]

                        encodings = tokenizer(
                            texts, padding="max_length", truncation=True,
                            max_length=args.seq_len, return_tensors="pt"
                        )
                        input_ids = encodings["input_ids"].to('cuda')
                        attention_mask = encodings["attention_mask"].to('cuda')
                        labels_tensor = torch.tensor(labels_list, dtype=torch.long, device='cuda')

                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                        batch_size = input_ids.size(0)
                        seq_lengths = attention_mask.sum(dim=1) - 1
                        final_logits = outputs.logits[torch.arange(batch_size, device='cuda'), seq_lengths]

                        label_token_ids = torch.tensor(dataset.label_token_ids, device='cuda')
                        label_logits = final_logits[:, label_token_ids]
                        predictions = label_logits.argmax(dim=-1)

                        total_correct += (predictions == labels_tensor).sum().item()
                        total_samples += len(batch_data)

                return total_correct / total_samples if total_samples > 0 else 0.0

        log(f"Dataset loaded: train={len(dataset.train_data)}, val={len(dataset.val_data)}")

    log("Warmup...")
    # Initialize batch for tasks (sample_new_batch was defined above for tasks)
    # Sample accum_steps batches for proper gradient accumulation
    if args.task != 'static':
        sample_new_batch(args.accum_steps)
    for _ in range(3):
        _ = loss_fn(0)
    if args.task != 'static':
        sample_new_batch(args.accum_steps)
    initial_loss = loss_fn(0)
    log(f"Initial loss: {initial_loss:.4f}")

    # Initial evaluation for tasks
    if evaluate_fn:
        initial_acc = evaluate_fn('val')
        log(f"Initial val accuracy: {initial_acc:.4f}")

    # Create trainer based on solver choice
    if args.solver == 'random':
        trainer = RandomSearchTrainer(model, epsilon=epsilon, n_perts=n_perts)
    elif args.solver == 'mezo':
        trainer = MeZOTrainer(model, lr=lr, epsilon=epsilon)
    elif args.solver == 'backprop':
        trainer = BackpropTrainer(model, lr=lr, accum_steps=args.accum_steps)
    else:  # spsa
        trainer = SPSATrainer(model, lr=lr, epsilon=epsilon, n_perts=n_perts,
                              use_curvature=args.use_1_5_spsa,
                              saturating_alpha=args.saturating_alpha,
                              lambda_reg=args.lambda_reg,
                              memory_efficient=args.memory_efficient,
                              accum_steps=args.accum_steps)

    # Parse explicit LR points if provided
    explicit_lrs = None
    if args.lr_points is not None:
        explicit_lrs = [float(x.strip()) for x in args.lr_points.split(',')]
        log(f"Using explicit LR points: {[f'{x:.0e}' for x in explicit_lrs]}")

    # Initial LR search if enabled (skip for 'local' - we start with provided lr)
    if args.search_strategy != 'none' and args.search_strategy != 'local':
        log("=" * 70)
        log("INITIAL LR SEARCH")
        log("=" * 70)

        if args.search_strategy == 'line':
            lr = trainer.line_search_lr(probe_loss_fn, args.lr_min, args.lr_max, n_points=args.n_points, seed=99999, n_seeds=args.n_seeds, resample_batch_fn=resample_probe_batch, explicit_lrs=explicit_lrs)
        elif args.search_strategy == 'binary':
            lr = trainer.binary_search_lr(probe_loss_fn, args.lr_min, args.lr_max, args.search_depth, seed=99999, accum_batches=args.probe_accum, n_seeds=args.n_seeds, resample_batch_fn=resample_probe_batch)
        else:  # quadratic
            lr = trainer.quadratic_search_lr(probe_loss_fn, args.lr_min, args.lr_max, args.search_depth, seed=99999, accum_batches=args.probe_accum)

        # Only tie epsilon to lr if not explicitly set
        if args.epsilon is None:
            epsilon = lr
            trainer.lr = lr
            trainer.epsilon = epsilon
            log(f"Using lr=eps={lr:.2e}")
        else:
            trainer.lr = lr
            log(f"Using lr={lr:.2e} (eps={epsilon:.2e} fixed)")
    elif args.search_strategy == 'local':
        log(f"Starting with lr=eps={lr:.2e} (local search on plateau)")

    log("=" * 70)
    log("TRAINING")
    log("=" * 70)

    losses = [initial_loss]
    best = initial_loss
    t0 = time.time()
    last_t = t0
    last_i = 0

    # For re-search triggering with EMA-based plateau detection
    stall_count = 0
    last_search_loss = initial_loss
    search_count = 1 if args.search_strategy != 'none' else 0
    loss_ema = initial_loss  # EMA of loss for plateau detection
    best_ema = initial_loss  # Best EMA seen

    # For LR decay on train accuracy plateau
    acc_ema = 0.5  # EMA of training accuracy
    best_acc_ema = 0.0  # Best train accuracy EMA
    best_test_acc = 0.0  # Best test accuracy (for checkpointing)
    acc_plateau_count = 0  # Iterations without accuracy improvement
    lr_decay_count = 0  # Number of LR decays performed

    for i in range(n_iterations):
        # Sample accum_steps fresh batches for this iteration
        # This ensures loss_plus and loss_minus use the SAME batches for proper gradient estimation
        if args.task != 'static' and not args.static_batch:
            sample_new_batch(args.accum_steps)

        # Use appropriate loss function for solver
        if args.solver == 'backprop':
            loss = trainer.step(backprop_loss_fn, i)
        else:
            loss = trainer.step(loss_fn, i)
        losses.append(loss)

        # Update EMA
        loss_ema = args.ema_alpha * loss + (1 - args.ema_alpha) * loss_ema

        # Check if we need to re-search
        need_search = False

        if args.search_strategy != 'none':
            # Check for divergence (against EMA, not raw loss)
            if loss_ema > last_search_loss * args.diverge_threshold:
                log(f"DIVERGENCE detected: EMA {loss_ema:.4f} > {last_search_loss:.4f} * {args.diverge_threshold}")
                need_search = True

            # Check for plateau using EMA (is the trend improving?)
            if loss_ema < best_ema - 0.01:
                best_ema = loss_ema
                stall_count = 0
            else:
                stall_count += 1
                if stall_count >= args.search_patience:
                    log(f"PLATEAU detected: EMA {loss_ema:.4f} not improving (best_ema={best_ema:.4f}) for {args.search_patience} iters")
                    need_search = True

            if need_search:
                log(f"RE-SEARCHING for optimal LR...")
                search_count += 1

                if args.search_strategy == 'local':
                    # Local search: only test 10x up and 10x down
                    new_lr, improved = trainer.local_search_lr(
                        probe_loss_fn, lr, seed=99999 + search_count * 1000000,
                        n_seeds=args.n_seeds, resample_batch_fn=resample_probe_batch)

                    if not improved:
                        log("=" * 70)
                        log("EARLY STOPPING: Local search found no improvement")
                        log("Neither 10x higher nor 10x lower LR helps. Training complete.")
                        log("=" * 70)
                        break  # Exit training loop

                    lr = new_lr
                elif args.search_strategy == 'line':
                    lr = trainer.line_search_lr(probe_loss_fn, args.lr_min, args.lr_max,
                                                 n_points=args.n_points, seed=99999 + search_count * 1000000, n_seeds=args.n_seeds, resample_batch_fn=resample_probe_batch, explicit_lrs=explicit_lrs)
                elif args.search_strategy == 'binary':
                    lr = trainer.binary_search_lr(probe_loss_fn, args.lr_min, args.lr_max,
                                                   args.search_depth, seed=99999 + search_count, accum_batches=args.probe_accum, n_seeds=args.n_seeds, resample_batch_fn=resample_probe_batch)
                else:
                    lr = trainer.quadratic_search_lr(probe_loss_fn, args.lr_min, args.lr_max,
                                                      args.search_depth, seed=99999 + search_count, accum_batches=args.probe_accum)

                # Only tie epsilon to lr if not explicitly set
                if args.epsilon is None:
                    epsilon = lr
                    trainer.lr = lr
                    trainer.epsilon = epsilon
                    log(f"New lr=eps={lr:.2e}")
                else:
                    trainer.lr = lr
                    log(f"New lr={lr:.2e} (eps={epsilon:.2e} fixed)")
                last_search_loss = loss_ema  # Use EMA as reference for divergence check
                best_ema = loss_ema  # Reset best_ema after search
                stall_count = 0

        if loss < best:
            best = loss

        # Log every iteration
        now = time.time()
        dt = now - last_t
        di = i - last_i if i > 0 else 1
        avg = dt / di if di > 0 else 0
        eta = (now - t0) / (i + 1) * (n_iterations - i - 1) if i > 0 else 0
        ema_str = f" | EMA: {loss_ema:.4f}" if args.search_strategy != 'none' else ""
        # Get quick accuracy for tasks
        acc_str = ""
        if quick_train_acc is not None:
            train_acc = quick_train_acc()
            acc_str = f" | Acc: {train_acc*100:.1f}%"

            # Update accuracy EMA and check for plateau (only if NOT using binary search)
            # Binary search handles LR adjustment, so LR decay should be disabled
            if args.search_strategy == 'none':
                acc_ema = args.ema_alpha * train_acc + (1 - args.ema_alpha) * acc_ema

                # Check for improvement (with 0.5% tolerance)
                if acc_ema > best_acc_ema + 0.005:
                    best_acc_ema = acc_ema
                    acc_plateau_count = 0
                else:
                    acc_plateau_count += 1

                # Decay LR if plateaued
                if acc_plateau_count >= args.lr_decay_patience and lr > args.lr_min_decay:
                    old_lr = lr
                    lr = lr * args.lr_decay_factor
                    epsilon = lr  # Always keep lr = eps
                    trainer.lr = lr
                    trainer.epsilon = epsilon
                    lr_decay_count += 1
                    log(f"  >>> LR DECAY #{lr_decay_count}: Train acc plateaued for {acc_plateau_count} iters. "
                        f"lr={old_lr:.1e} -> {lr:.1e}")
                    acc_plateau_count = 0  # Reset counter after decay
                    best_acc_ema = acc_ema  # Reset best after decay

        log(f"Iter {i:6d} | Loss: {loss:.4f}{acc_str} | Best: {best:.4f}{ema_str} | "
            f"Red: {initial_loss/loss:.2f}x | lr={lr:.0e} eps={epsilon:.0e} | {avg:.1f}s/it | ETA: {eta/3600:.1f}h")

        # W&B logging
        if wandb_run is not None:
            log_dict = {
                "iteration": i,
                "loss": loss,
                "best_loss": best,
                "lr": lr,
                "epsilon": epsilon,
                "loss_reduction": initial_loss / loss if loss > 0 else 0,
            }
            if quick_train_acc is not None:
                log_dict["train_acc"] = train_acc
            if args.search_strategy != 'none':
                log_dict["loss_ema"] = loss_ema
            wandb_run.log(log_dict)

        # Periodic evaluation for tasks (val AND test)
        if evaluate_fn and (i + 1) % args.eval_interval == 0:
            val_acc = evaluate_fn('val')
            test_acc = evaluate_fn('test')
            log(f"  >>> [Eval] Val: {val_acc:.4f} ({val_acc*100:.1f}%) | Test: {test_acc:.4f} ({test_acc*100:.1f}%)")

            # W&B logging for eval
            if wandb_run is not None:
                wandb_run.log({
                    "iteration": i,
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                })

            # Checkpoint on best test accuracy
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                import os
                os.makedirs('checkpoints', exist_ok=True)
                ckpt = {
                    'iteration': i + 1,
                    'model_state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    'losses': losses,
                    'best': best,
                    'lr': trainer.lr if hasattr(trainer, 'lr') else lr,
                    'epsilon': trainer.epsilon if hasattr(trainer, 'epsilon') else epsilon,
                    'val_acc': val_acc,
                    'test_acc': test_acc,
                    'args': vars(args),
                }
                ckpt_path = f'checkpoints/best_{args.task}_np{args.n_perts}_bs{args.batch_size}.pt'
                torch.save(ckpt, ckpt_path)
                log(f"  >>> NEW BEST TEST! Checkpoint saved: {ckpt_path}")

                # W&B best metrics
                if wandb_run is not None:
                    wandb_run.log({"best_test_acc": best_test_acc})
        last_t = now
        last_i = i

        if (i + 1) % args.checkpoint_interval == 0:
            import os
            os.makedirs('checkpoints', exist_ok=True)
            ckpt = {
                'iteration': i + 1,
                'model_state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                'losses': losses,
                'best': best,
                'lr': trainer.lr,
                'epsilon': trainer.epsilon,
            }
            ckpt_path = f'checkpoints/ckpt_{args.task}_{i+1}.pt'
            torch.save(ckpt, ckpt_path)
            log(f"  >>> Checkpoint saved: {ckpt_path}")

    log(f"DONE: {initial_loss:.4f} -> {losses[-1]:.4f} ({initial_loss/losses[-1]:.2f}x)")
    if args.search_strategy != 'none':
        log(f"Total searches: {search_count}")

    # Final W&B summary
    if wandb_run is not None:
        wandb_run.summary["final_loss"] = losses[-1]
        wandb_run.summary["loss_reduction"] = initial_loss / losses[-1] if losses[-1] > 0 else 0
        wandb_run.summary["best_loss"] = best

    # Final evaluation for tasks
    if evaluate_fn:
        log("=" * 70)
        log("FINAL EVALUATION")
        log("=" * 70)
        final_val_acc = evaluate_fn('val')
        log(f"Final Val Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.1f}%)")
        try:
            final_test_acc = evaluate_fn('test')
            log(f"Final Test Accuracy: {final_test_acc:.4f} ({final_test_acc*100:.1f}%)")

            # W&B final summary
            if wandb_run is not None:
                wandb_run.summary["final_val_acc"] = final_val_acc
                wandb_run.summary["final_test_acc"] = final_test_acc
                wandb_run.summary["best_test_acc"] = best_test_acc
        except Exception as e:
            log(f"Test evaluation skipped: {e}")

    # Close W&B
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
