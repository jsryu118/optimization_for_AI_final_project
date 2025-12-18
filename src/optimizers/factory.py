"""
Optimizer factory for creating all optimizers used in experiments.
"""
import torch.optim as optim
from prodigyopt import Prodigy
from .dog import get_dog_optimizer

try:
    import schedulefree
    SCHEDULEFREE_AVAILABLE = True
except ImportError:
    SCHEDULEFREE_AVAILABLE = False


def create_optimizer(optimizer_name, model_params, lr=None, **kwargs):
    """
    Create an optimizer based on the given name and parameters.

    Args:
        optimizer_name: Name of the optimizer (e.g., 'sgd', 'adam', 'adamw', 'dog', 'ldog', 'tdog', 'prodigy')
        model_params: Model parameters to optimize
        lr: Learning rate (required for baseline optimizers, optional for parameter-free)
        **kwargs: Additional optimizer-specific parameters

    Returns:
        Optimizer instance
    """
    optimizer_name = optimizer_name.lower()

    # Baseline optimizers (require learning rate)
    if optimizer_name == "sgd":
        if lr is None:
            raise ValueError("Learning rate must be specified for SGD")
        momentum = kwargs.get("momentum", 0.9)
        weight_decay = kwargs.get("weight_decay", 0.01)
        return optim.SGD(
            model_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )

    elif optimizer_name == "adam":
        if lr is None:
            raise ValueError("Learning rate must be specified for Adam")
        weight_decay = kwargs.get("weight_decay", 0.0)
        return optim.Adam(
            model_params,
            lr=lr,
            weight_decay=weight_decay
        )

    elif optimizer_name == "adamw":
        if lr is None:
            raise ValueError("Learning rate must be specified for AdamW")
        weight_decay = kwargs.get("weight_decay", 0.01)
        return optim.AdamW(
            model_params,
            lr=lr,
            weight_decay=weight_decay
        )

    # Parameter-free optimizers
    elif optimizer_name in ["dog", "ldog", "l-dog", "tdog", "t-dog"]:
        lr = lr if lr is not None else 1.0
        reps_rel = kwargs.get("reps_rel", 1e-6)
        weight_decay = kwargs.get("weight_decay", 0.0001)
        return get_dog_optimizer(
            model_params,
            variant=optimizer_name,
            lr=lr,
            reps_rel=reps_rel,
            weight_decay=weight_decay
        )

    elif optimizer_name == "prodigy":
        lr = lr if lr is not None else 1.0
        weight_decay = kwargs.get("weight_decay", 0.0001)
        return Prodigy(
            model_params,
            lr=lr,
            weight_decay=weight_decay
        )

    # ScheduleFree optimizers (no scheduler needed!)
    elif optimizer_name in ["schedulefree_sgd", "schedulefree-sgd"]:
        if not SCHEDULEFREE_AVAILABLE:
            raise ImportError("schedulefree package not installed. Run: pip install schedulefree")
        if lr is None:
            raise ValueError("Learning rate must be specified for ScheduleFree-SGD")
        momentum = kwargs.get("momentum", 0.9)
        weight_decay = kwargs.get("weight_decay", 0.0001)
        return schedulefree.SGDScheduleFree(
            model_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )

    elif optimizer_name in ["schedulefree_adam", "schedulefree-adam"]:
        if not SCHEDULEFREE_AVAILABLE:
            raise ImportError("schedulefree package not installed. Run: pip install schedulefree")
        if lr is None:
            raise ValueError("Learning rate must be specified for ScheduleFree-Adam")
        weight_decay = kwargs.get("weight_decay", 0.0001)
        return schedulefree.AdamWScheduleFree(
            model_params,
            lr=lr,
            weight_decay=weight_decay
        )

    elif optimizer_name in ["schedulefree_adamw", "schedulefree-adamw"]:
        if not SCHEDULEFREE_AVAILABLE:
            raise ImportError("schedulefree package not installed. Run: pip install schedulefree")
        if lr is None:
            raise ValueError("Learning rate must be specified for ScheduleFree-AdamW")
        weight_decay = kwargs.get("weight_decay", 0.0)
        return schedulefree.AdamWScheduleFree(
            model_params,
            lr=lr,
            weight_decay=weight_decay
        )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_optimizer_configs():
    """
    Get all optimizer configurations for experiments as specified in guide.md.

    Returns:
        Dictionary mapping optimizer names to their configurations
    """
    configs = {
        # Parameter-free optimizers (lr = 1.0)
        "dog": {"lr": 1.0, "reps_rel": 1e-6},
        "ldog": {"lr": 1.0, "reps_rel": 1e-6},
        "tdog": {"lr": 1.0, "reps_rel": 1e-6},
        "prodigy": {"lr": 1.0},

        # Baseline optimizers with various learning rates
        # SGD: Larger learning rates (0.1, 0.01, 0.001)
        "sgd_0.1": {"lr": 0.1, "momentum": 0.9},
        "sgd_0.01": {"lr": 0.01, "momentum": 0.9},
        "sgd_0.001": {"lr": 0.001, "momentum": 0.9},

        # Adam: Smaller learning rates (0.001, 0.0001, 0.00001)
        "adam_0.001": {"lr": 0.001},
        "adam_0.0001": {"lr": 0.0001},
        "adam_0.00001": {"lr": 0.00001},

        # AdamW: Smaller learning rates (0.001, 0.0001, 0.00001)
        "adamw_0.001": {"lr": 0.001},
        "adamw_0.0001": {"lr": 0.0001},
        "adamw_0.00001": {"lr": 0.00001},
    }
    return configs
