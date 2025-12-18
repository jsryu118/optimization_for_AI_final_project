"""
DOG optimizer variants wrapper.
This module provides a unified interface for DOG, L-DOG, and T-DOG optimizers.
"""
from dog import DoG, LDoG, PolynomialDecayAverager


def get_dog_optimizer(model_params, variant="dog", lr=1.0, reps_rel=1e-6, weight_decay=0.0):
    """
    Get a DOG optimizer variant.

    Args:
        model_params: Model parameters to optimize
        variant: Variant of DOG to use ("dog", "ldog", or "tdog")
        lr: Base learning rate (should be 1.0 for parameter-free)
        reps_rel: Normalized version of r_epsilon parameter
        weight_decay: Weight decay coefficient

    Returns:
        DOG optimizer instance
    """
    variant = variant.lower()

    if variant == "dog":
        optimizer = DoG(
            model_params,
            lr=lr,
            reps_rel=reps_rel,
            weight_decay=weight_decay
        )
    elif variant == "ldog" or variant == "l-dog":
        optimizer = LDoG(
            model_params,
            lr=lr,
            reps_rel=reps_rel,
            weight_decay=weight_decay
        )
    elif variant == "tdog" or variant == "t-dog":
        # T-DOG is DoG with tamed updates (using different reps_rel)
        # Typically requires smaller reps_rel to prevent divergence
        optimizer = DoG(
            model_params,
            lr=lr,
            reps_rel=reps_rel * 0.01,  # Tamed version uses smaller reps_rel
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown DOG variant: {variant}. Choose from 'dog', 'ldog', or 'tdog'")

    return optimizer


def get_dog_averager(model, gamma=8):
    """
    Get a PolynomialDecayAverager for DOG optimization.

    Note: DOG/L-DOG should be combined with iterate averaging for best performance.

    Args:
        model: PyTorch model to average
        gamma: Polynomial decay parameter (default: 8)

    Returns:
        PolynomialDecayAverager instance
    """
    return PolynomialDecayAverager(model, gamma=gamma)
