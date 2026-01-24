# Tasks module for LLM classification datasets and Atari RL
#
# Note: Imports are explicit to avoid loading unnecessary dependencies.
# - tasks_llm requires datasets/pandas (for LLM tasks)
# - tasks_atari requires ale_py/opencv (for Atari RL)
#
# Import what you need explicitly:
#   from tasks.tasks_llm import get_task_dataset
#   from tasks.tasks_atari import AtariEnvWrapper, ParallelAtariEvaluator

def get_task_dataset(*args, **kwargs):
    """Lazy import for LLM task dataset."""
    from .tasks_llm import get_task_dataset as _get_task_dataset
    return _get_task_dataset(*args, **kwargs)


def get_atari_components():
    """Get all Atari-related components."""
    from .tasks_atari import (
        AtariEnvWrapper,
        ParallelAtariEvaluator,
        run_rollout,
        evaluate_policy,
        apply_packed_perturbation_fast,
        generate_env_seeds,
        generate_pert_seed,
    )
    return {
        'AtariEnvWrapper': AtariEnvWrapper,
        'ParallelAtariEvaluator': ParallelAtariEvaluator,
        'run_rollout': run_rollout,
        'evaluate_policy': evaluate_policy,
        'apply_packed_perturbation_fast': apply_packed_perturbation_fast,
        'generate_env_seeds': generate_env_seeds,
        'generate_pert_seed': generate_pert_seed,
    }


__all__ = [
    'get_task_dataset',
    'get_atari_components',
]
