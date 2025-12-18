"""
Scheduler comparison experiment - separate training runs.
스케줄러 방식: 10, 20, 30, 40, 50 epoch을 각각 별도로 학습.
"""
import torch
import argparse
from src.data.vision_data import get_cifar10_loaders
from src.models.vision_models import get_resnet20_cifar10
from src.optimizers.factory import create_optimizer
from src.trainer_with_scheduler import TrainerWithScheduler
from src.utils import set_seed
import json
from pathlib import Path


def train_with_scheduler(optimizer_name, epochs, best_lr, seed=42):
    """
    Train with scheduler for specific number of epochs.
    각 epoch 설정마다 별도로 학습.

    Args:
        optimizer_name: 'sgd', 'adam', or 'adamw'
        epochs: Number of epochs to train
        best_lr: Best learning rate
        seed: Random seed

    Returns:
        Final test accuracy
    """
    # Set seed
    set_seed(seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    print(f"\nLoading CIFAR-10...")
    train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=128)

    # Create model
    print(f"Creating ResNet-20...")
    model = get_resnet20_cifar10()

    # Create optimizer
    print(f"Creating optimizer: {optimizer_name} with LR={best_lr}")
    optimizer = create_optimizer(
        optimizer_name,
        model.parameters(),
        lr=best_lr
    )

    # Create Cosine Annealing Scheduler
    print(f"Creating Cosine Scheduler for {epochs} epochs...")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=0
    )

    # Experiment name
    experiment_name = f"cifar10_{optimizer_name}_scheduler_{epochs}epochs"

    # Create trainer
    trainer = TrainerWithScheduler(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        task_type="vision",
        checkpoint_dir="scheduler_experiments"
    )

    # Train (only save checkpoint at final epoch)
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=epochs,
        checkpoint_epochs=[epochs],  # Only save at final epoch
        experiment_name=experiment_name
    )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, required=True,
                       choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Best LR mapping
    best_lr_map = {
        'sgd': 0.01,
        'adam': 0.001,
        'adamw': 0.001
    }

    best_lr = best_lr_map[args.optimizer]

    # Train for different epoch settings
    epoch_settings = [10, 20, 30, 40, 50]

    print("\n" + "="*80)
    print(f"SCHEDULER EXPERIMENTS: {args.optimizer.upper()}")
    print(f"Best LR: {best_lr}")
    print(f"Epoch settings: {epoch_settings}")
    print("="*80)

    all_results = {}

    for epochs in epoch_settings:
        print(f"\n{'='*80}")
        print(f"Training with {epochs} epochs + Cosine Scheduler")
        print(f"{'='*80}")

        results = train_with_scheduler(
            optimizer_name=args.optimizer,
            epochs=epochs,
            best_lr=best_lr,
            seed=args.seed
        )

        # Store result
        all_results[f"{epochs}_epochs"] = {
            'epochs': epochs,
            'test_accuracy': results['final_test_accuracy'],
            'training_time': results['training_time']
        }

        print(f"\n✓ {epochs} epochs complete: {results['final_test_accuracy']:.2f}%")

    # Save summary
    summary_path = Path("scheduler_experiments") / f"{args.optimizer}_scheduler_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print("ALL SCHEDULER EXPERIMENTS COMPLETE")
    print("="*80)
    print("\nSummary:")
    for epoch_key, result in all_results.items():
        print(f"  {epoch_key}: {result['test_accuracy']:.2f}% ({result['training_time']/60:.1f} min)")
    print(f"\n✓ Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
