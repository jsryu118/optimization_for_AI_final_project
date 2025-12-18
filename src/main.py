"""
Main entry point for running optimizer comparison experiments.
"""
import os
from config import get_config, print_config
from utils import set_seed, get_device, count_parameters
from data.vision_data import get_cifar10_loaders, get_oxford_pet_loaders
from data.nlp_data import get_sst2_loaders
from models.vision_models import get_resnet20_cifar10, get_vit_tiny_pretrained
from models.nlp_models import get_distilbert_sst2
from optimizers.factory import create_optimizer, get_optimizer_configs
from trainer import Trainer


def setup_wandb(args):
    """
    Setup Weights & Biases logging if enabled.

    Args:
        args: Configuration arguments
    """
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"{args.task}_{args.optimizer}"
        )


def get_data_loaders(task, batch_size, data_dir):
    """
    Get data loaders for the specified task.

    Args:
        task: Task name
        batch_size: Batch size
        data_dir: Data directory

    Returns:
        Tuple of (train_loader, val_loader, test_loader, task_type)
    """
    if task == "cifar10":
        train_loader, val_loader, test_loader = get_cifar10_loaders(
            batch_size=batch_size,
            data_dir=data_dir
        )
        task_type = "vision"
    elif task == "oxford_pet":
        train_loader, val_loader, test_loader = get_oxford_pet_loaders(
            batch_size=batch_size,
            data_dir=data_dir
        )
        task_type = "vision"
    elif task == "sst2":
        train_loader, val_loader, test_loader, _ = get_sst2_loaders(
            batch_size=batch_size
        )
        task_type = "nlp"
    else:
        raise ValueError(f"Unknown task: {task}")

    return train_loader, val_loader, test_loader, task_type


def get_model(task, device):
    """
    Get model for the specified task.

    Args:
        task: Task name
        device: Device to load model on

    Returns:
        Model instance
    """
    if task == "cifar10":
        model = get_resnet20_cifar10(num_classes=10)
    elif task == "oxford_pet":
        model = get_vit_tiny_pretrained(num_classes=37)
    elif task == "sst2":
        model = get_distilbert_sst2(num_labels=2)
    else:
        raise ValueError(f"Unknown task: {task}")

    return model


def parse_optimizer_name(optimizer_name):
    """
    Parse optimizer name to extract base name and learning rate.

    Args:
        optimizer_name: Optimizer name (e.g., 'sgd_0.01', 'dog', 'adam_0.001')

    Returns:
        Tuple of (base_name, lr)
    """
    parts = optimizer_name.split("_")

    # Check if optimizer name contains 'schedulefree'
    if 'schedulefree' in optimizer_name:
        # ScheduleFree optimizers: lr must be provided via --lr argument
        return optimizer_name, None

    if len(parts) == 1:
        # Parameter-free optimizer (dog, ldog, tdog, prodigy)
        return optimizer_name, None
    elif len(parts) == 2:
        # Baseline optimizer with LR (sgd_0.01, adam_0.001, etc.)
        base_name = parts[0]
        try:
            lr = float(parts[1])
            return base_name, lr
        except ValueError:
            # Not a number, treat as full optimizer name
            return optimizer_name, None
    else:
        # Multi-part name without LR suffix
        return optimizer_name, None


def main():
    """Main execution function."""
    # Get configuration
    args = get_config()
    print_config(args)

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Setup W&B if enabled
    setup_wandb(args)

    # Get device
    device = get_device()
    print(f"Using device: {device}\n")

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, task_type = get_data_loaders(
        args.task,
        args.batch_size,
        args.data_dir
    )
    print(f"Data loaded successfully. Task type: {task_type}\n")

    # Create model
    print("Creating model...")
    model = get_model(args.task, device)
    num_params = count_parameters(model)
    print(f"Model created with {num_params:,} trainable parameters\n")

    # Parse optimizer name and get learning rate
    base_optimizer_name, lr_from_name = parse_optimizer_name(args.optimizer)

    # Use provided LR if available, otherwise use LR from optimizer name
    lr = args.lr if args.lr is not None else lr_from_name

    # Create optimizer
    print(f"Creating optimizer: {args.optimizer}")
    optimizer = create_optimizer(
        base_optimizer_name,
        model.parameters(),
        lr=lr,
        reps_rel=args.reps_rel,
        weight_decay=args.weight_decay
    )
    print(f"Optimizer created successfully\n")

    # Determine if we should use DOG averaging
    use_dog_averaging = base_optimizer_name.lower() in ["dog", "ldog", "l-dog", "tdog", "t-dog"]

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        task_type=task_type,
        use_dog_averaging=use_dog_averaging
    )

    # Train the model
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=args.epochs
    )

    # Save results
    os.makedirs(args.results_dir, exist_ok=True)
    lr_str = f"{lr}" if lr is not None else "1.0"
    result_filename = f"{args.task}_{args.optimizer}_lr{lr_str}.json"
    trainer.save_results(result_filename)

    # Print final summary
    print("\n" + "=" * 60)
    print("Experiment Summary")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Test Accuracy: {results['test_accuracy']:.2f}%")
    print(f"Best Val Accuracy: {results['best_val_accuracy']:.2f}%")
    print(f"Training Time: {results['training_time']:.2f}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
