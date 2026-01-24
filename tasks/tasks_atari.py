"""
Atari Breakout Environment and Parallel Evaluation for SPSA.

This module provides:
- AtariEnvWrapper: Gymnasium wrapper with preprocessing (grayscale, resize, frame stack)
- run_rollout: Single episode execution
- ParallelAtariEvaluator: Multiprocessing pool for parallel perturbation evaluation
- apply_packed_perturbation: CPU-based weight perturbation for workers

Key seed strategy:
- Each perturbation j has a pert_seed for generating delta
- Each perturbation j has K env_seeds for rollouts
- All 3 evaluations (clean, plus, minus) use the SAME env_seeds for variance reduction
"""

import torch
import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Optional
import multiprocessing as mp


class AtariEnvWrapper:
    """
    Wraps ALE Breakout with preprocessing using direct ale_py interface.

    Preprocessing:
    - Grayscale conversion
    - Resize to 84x84
    - Frame stacking (4 frames)
    - Normalization to [0, 1]

    Uses ale_py.ALEInterface directly for better compatibility.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the wrapper.

        Args:
            seed: Optional random seed for environment
        """
        import ale_py
        from ale_py.roms import Breakout

        # Create ALE interface
        self.ale = ale_py.ALEInterface()

        # Set options
        self.ale.setInt('random_seed', seed if seed is not None else 0)
        self.ale.setFloat('repeat_action_probability', 0.0)  # Deterministic
        self.ale.setBool('display_screen', False)

        # Load the ROM
        self.ale.loadROM(Breakout)

        # Get minimal action set (NOOP, FIRE, LEFT, RIGHT for Breakout)
        self.action_set = self.ale.getMinimalActionSet()
        self.num_actions = len(self.action_set)

        self.frame_stack = deque(maxlen=4)
        self._seed = seed
        self.frameskip = 4  # Standard Atari preprocessing

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset the environment.

        Args:
            seed: Optional seed for reproducibility

        Returns:
            Initial state as (4, 84, 84) numpy array
        """
        if seed is not None:
            self.ale.setInt('random_seed', seed)

        self.ale.reset_game()

        frame = self._get_frame()
        self.frame_stack.clear()
        for _ in range(4):
            self.frame_stack.append(frame)

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take a step in the environment with frameskip.

        Args:
            action: Action index (0-3 for Breakout minimal actions)

        Returns:
            Tuple of (state, reward, done, info)
        """
        # Map action index to ALE action
        ale_action = self.action_set[action]

        # Apply frameskip
        total_reward = 0.0
        for _ in range(self.frameskip):
            reward = self.ale.act(ale_action)
            total_reward += reward
            if self.ale.game_over():
                break

        frame = self._get_frame()
        self.frame_stack.append(frame)

        done = self.ale.game_over()
        info = {'lives': self.ale.lives()}

        return self._get_state(), total_reward, done, info

    def _get_frame(self) -> np.ndarray:
        """
        Get current frame, preprocessed.

        Returns:
            Grayscale 84x84 frame normalized to [0, 1]
        """
        import cv2

        # Get grayscale screen directly from ALE
        screen = self.ale.getScreenGrayscale()

        # Resize to 84x84
        resized = cv2.resize(screen, (84, 84), interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        return normalized

    def _get_state(self) -> np.ndarray:
        """
        Get current state as stacked frames.

        Returns:
            State as (4, 84, 84) numpy array
        """
        return np.array(self.frame_stack, dtype=np.float32)

    def close(self):
        """Close the environment."""
        # ALE doesn't need explicit close
        pass


def run_rollout(
    model: torch.nn.Module,
    env: AtariEnvWrapper,
    seed: int,
    max_steps: int = 10000,
    device: str = 'cpu'
) -> float:
    """
    Run a single episode with deterministic seed.

    Args:
        model: AtariCNN policy network
        env: AtariEnvWrapper instance
        seed: Environment seed for reproducibility
        max_steps: Maximum episode length
        device: Device for inference ('cpu' for workers)

    Returns:
        Total reward for the episode
    """
    state = env.reset(seed=seed)
    total_reward = 0.0

    for _ in range(max_steps):
        with torch.no_grad():
            state_tensor = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
            logits = model(state_tensor)
            action = logits.argmax(dim=1).item()  # Greedy action

        state, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    return total_reward


def apply_packed_perturbation(
    model: torch.nn.Module,
    packed: torch.Tensor,
    alpha: float
) -> None:
    """
    Apply bit-packed perturbation to model weights in-place.

    CPU version for workers (no GPU/Triton).

    Args:
        model: PyTorch model
        packed: Bit-packed perturbation tensor (uint8)
        alpha: Scale factor (positive for +delta, negative for -delta)
    """
    offset = 0
    for param in model.parameters():
        if not param.requires_grad:
            continue

        numel = param.numel()
        flat = param.data.view(-1)

        for i in range(numel):
            byte_idx = (offset + i) // 8
            bit_idx = (offset + i) % 8
            bit = (packed[byte_idx].item() >> bit_idx) & 1
            sign = 1.0 if bit == 1 else -1.0
            flat[i] += alpha * sign

        offset += numel


def apply_packed_perturbation_fast(
    model: torch.nn.Module,
    packed: torch.Tensor,
    alpha: float
) -> None:
    """
    Vectorized version of apply_packed_perturbation for better performance.

    Args:
        model: PyTorch model
        packed: Bit-packed perturbation tensor (uint8)
        alpha: Scale factor
    """
    offset = 0
    for param in model.parameters():
        if not param.requires_grad:
            continue

        numel = param.numel()
        flat = param.data.view(-1)

        # Calculate byte and bit indices for all elements at once
        indices = torch.arange(numel, dtype=torch.long)
        byte_indices = (offset + indices) // 8
        bit_indices = (offset + indices) % 8

        # Extract bits
        packed_bytes = packed[byte_indices].long()
        bits = (packed_bytes >> bit_indices) & 1

        # Convert to signs: 1 -> +1, 0 -> -1
        signs = (2 * bits - 1).float()

        # Apply perturbation
        flat.add_(signs, alpha=alpha)

        offset += numel


# Global worker state (initialized in worker_init)
_worker_model = None
_worker_env = None


def worker_init(num_actions: int):
    """
    Initialize worker with model and environment.

    Called once per worker process.

    Args:
        num_actions: Number of actions for the CNN
    """
    global _worker_model, _worker_env

    # Import here to avoid issues with multiprocessing
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from models.atari_cnn import AtariCNN

    _worker_model = AtariCNN(num_actions=num_actions)
    _worker_model.eval()
    _worker_env = AtariEnvWrapper()


def worker_evaluate_perturbation(args: Tuple) -> Tuple[float, float, float]:
    """
    Worker function: evaluate one perturbation.

    CRITICAL: Same env_seeds are used for clean, plus, AND minus evaluations.
    This is essential for variance reduction in SPSA.

    Args:
        args: Tuple of (base_weights, pert_seed, env_seeds, epsilon, packed_size)

    Returns:
        Tuple of (f_clean, f_plus, f_minus) - mean rewards for each evaluation
    """
    global _worker_model, _worker_env

    base_weights, pert_seed, env_seeds, epsilon, packed_size = args

    # Load base weights
    _worker_model.load_state_dict(base_weights)

    # Generate perturbation using the perturbation seed
    torch.manual_seed(pert_seed)
    packed = torch.randint(0, 256, (packed_size,), dtype=torch.uint8)

    # Evaluate f_clean (at theta)
    rewards_clean = []
    for seed in env_seeds:
        reward = run_rollout(_worker_model, _worker_env, seed, device='cpu')
        rewards_clean.append(reward)
    f_clean = np.mean(rewards_clean)

    # Apply +epsilon perturbation
    apply_packed_perturbation_fast(_worker_model, packed, +epsilon)

    # Evaluate f_plus (at theta + epsilon*delta) with SAME env_seeds
    rewards_plus = []
    for seed in env_seeds:
        reward = run_rollout(_worker_model, _worker_env, seed, device='cpu')
        rewards_plus.append(reward)
    f_plus = np.mean(rewards_plus)

    # Apply -2*epsilon (now at theta - epsilon*delta)
    apply_packed_perturbation_fast(_worker_model, packed, -2 * epsilon)

    # Evaluate f_minus (at theta - epsilon*delta) with SAME env_seeds
    rewards_minus = []
    for seed in env_seeds:
        reward = run_rollout(_worker_model, _worker_env, seed, device='cpu')
        rewards_minus.append(reward)
    f_minus = np.mean(rewards_minus)

    return (f_clean, f_plus, f_minus)


class ParallelAtariEvaluator:
    """
    Manages worker pool for parallel perturbation evaluation.

    Each worker has its own AtariCNN model and environment instance.
    Perturbations are distributed across workers for parallel evaluation.

    Set num_workers=0 for sequential evaluation (useful for debugging).
    """

    def __init__(self, num_workers: int, num_actions: int = 4):
        """
        Initialize the evaluator with a worker pool.

        Args:
            num_workers: Number of parallel worker processes (0 for sequential)
            num_actions: Number of actions for the CNN
        """
        self.num_workers = num_workers
        self.num_actions = num_actions
        self.pool = None

        if num_workers > 0:
            # Use 'spawn' start method to avoid CUDA fork issues
            # This is important because CUDA doesn't work well with fork
            ctx = mp.get_context('spawn')

            # Create process pool with initialization
            self.pool = ctx.Pool(
                num_workers,
                initializer=worker_init,
                initargs=(num_actions,)
            )
        else:
            # Sequential mode - initialize worker state in main process
            worker_init(num_actions)

    def evaluate_perturbations(
        self,
        base_weights: Dict[str, torch.Tensor],
        pert_assignments: List[Tuple[int, int, List[int]]],
        epsilon: float,
        packed_size: int,
    ) -> List[Tuple[float, float, float]]:
        """
        Evaluate multiple perturbations in parallel (or sequentially if num_workers=0).

        Args:
            base_weights: Model state_dict (will be transferred to each worker)
            pert_assignments: List of (pert_idx, pert_seed, env_seeds) tuples
            epsilon: Perturbation scale
            packed_size: Size of packed perturbation tensor

        Returns:
            List of (f_clean, f_plus, f_minus) tuples, one per perturbation
        """
        # Prepare task arguments
        tasks = [
            (base_weights, pert_seed, env_seeds, epsilon, packed_size)
            for (pert_idx, pert_seed, env_seeds) in pert_assignments
        ]

        if self.pool is not None:
            # Execute in parallel
            results = self.pool.map(worker_evaluate_perturbation, tasks)
        else:
            # Execute sequentially
            results = [worker_evaluate_perturbation(task) for task in tasks]

        return results

    def close(self):
        """Close the worker pool."""
        if self.pool is not None:
            self.pool.close()
            self.pool.join()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass


def evaluate_policy(
    model: torch.nn.Module,
    n_episodes: int = 20,
    max_steps: int = 10000,
    seed_base: int = 999999,
    device: str = 'cuda'
) -> float:
    """
    Evaluate policy performance over multiple episodes.

    Uses fixed seeds for reproducible evaluation.

    Args:
        model: Policy network
        n_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        seed_base: Base seed for evaluation episodes
        device: Device for model inference

    Returns:
        Mean reward across episodes
    """
    model.eval()
    env = AtariEnvWrapper()

    rewards = []
    for i in range(n_episodes):
        seed = seed_base + i
        state = env.reset(seed=seed)
        total_reward = 0.0

        for _ in range(max_steps):
            with torch.no_grad():
                state_tensor = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
                logits = model(state_tensor)
                action = logits.argmax(dim=1).item()

            state, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)

    env.close()
    return np.mean(rewards)


def generate_env_seeds(iteration: int, pert_idx: int, k: int) -> List[int]:
    """
    Generate K environment seeds for a perturbation.

    Uses a deterministic hash to ensure reproducibility.

    Args:
        iteration: Current SPSA iteration
        pert_idx: Perturbation index within iteration
        k: Number of seeds to generate

    Returns:
        List of K environment seeds
    """
    seeds = []
    for roll_idx in range(k):
        # Combine iteration, perturbation, and rollout index
        combined = hash((iteration, pert_idx, roll_idx)) % (2**31)
        seeds.append(combined)
    return seeds


def generate_pert_seed(iteration: int, pert_idx: int) -> int:
    """
    Generate perturbation seed for delta generation.

    Args:
        iteration: Current SPSA iteration
        pert_idx: Perturbation index within iteration

    Returns:
        Seed for torch.manual_seed when generating delta
    """
    return iteration * 10000 + pert_idx
