"""
Configuration management for experiments.
"""
import argparse


def get_config():
    """
    Parse command-line arguments and return configuration.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Comparative Evaluation of Parameter-Free Optimizers"
    )

    # Experiment settings
    parser.add_argument(
        "--task",
        type=str,
        choices=["cifar10", "oxford_pet", "sst2"],
        required=True,
        help="Task to run (cifar10, oxford_pet, or sst2)"
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        required=True,
        help="Optimizer name (e.g., 'dog', 'ldog', 'tdog', 'prodigy', 'sgd_0.01', 'adam_0.001', etc.)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (defaults: 50 for CIFAR-10, 10 for Oxford-Pet, 5 for SST-2)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (defaults: 128 for CIFAR-10, 64 for Oxford-Pet, 32 for SST-2)"
    )

    # Optimizer-specific settings
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides default for the optimizer)"
    )

    parser.add_argument(
        "--reps-rel",
        type=float,
        default=1e-6,
        help="DOG reps_rel parameter (default: 1e-6)"
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay coefficient (default: 0.0)"
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    # Data paths
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory for datasets (default: ./data)"
    )

    # Results
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Directory to save results (default: ./results)"
    )

    # Logging
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default="optimizer-comparison",
        help="W&B project name (default: optimizer-comparison)"
    )

    args = parser.parse_args()

    # Set task-specific defaults if not provided
    if args.task == "cifar10":
        if args.epochs is None:
            args.epochs = 50
        if args.batch_size is None:
            args.batch_size = 128
    elif args.task == "oxford_pet":
        if args.epochs is None:
            args.epochs = 10
        if args.batch_size is None:
            args.batch_size = 64
    elif args.task == "sst2":
        if args.epochs is None:
            args.epochs = 5
        if args.batch_size is None:
            args.batch_size = 32

    return args


def print_config(args):
    """
    Print the configuration.

    Args:
        args: Parsed arguments
    """
    print("\n" + "=" * 60)
    print("Experiment Configuration")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr if args.lr else 'Auto'}")
    print(f"Seed: {args.seed}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Results Directory: {args.results_dir}")
    if args.use_wandb:
        print(f"W&B Project: {args.wandb_project}")
    print("=" * 60 + "\n")
