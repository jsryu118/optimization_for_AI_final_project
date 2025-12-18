"""
Training loop with optional scheduler and checkpoint support.
Scheduler 비교 실험용.
"""
import time
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from .utils import ResultLogger
import json


class TrainerWithScheduler:
    """
    Trainer with scheduler support and checkpoint saving.
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        device,
        task_type="vision",
        checkpoint_dir="checkpoints"
    ):
        """
        Initialize the trainer.

        Args:
            model: PyTorch model to train
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler (or None)
            device: Device to run training on
            task_type: Type of task ('vision' or 'nlp')
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.task_type = task_type
        self.criterion = nn.CrossEntropyLoss()
        self.logger = ResultLogger()

        # Create checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()

        # ScheduleFree: Enable training mode
        if hasattr(self.optimizer, 'train'):
            self.optimizer.train()

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            if self.task_type == "vision":
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
            else:  # NLP
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                targets = batch["labels"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if self.task_type == "vision":
                outputs = self.model(inputs)
            else:
                outputs = self.model(**inputs).logits

            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def evaluate(self, data_loader):
        """
        Evaluate the model on a data loader.

        Args:
            data_loader: Data loader for evaluation

        Returns:
            Accuracy as a percentage
        """
        self.model.eval()

        # ScheduleFree: Enable evaluation mode
        if hasattr(self.optimizer, 'eval'):
            self.optimizer.eval()

        correct = 0
        total = 0

        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            if self.task_type == "vision":
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
            else:  # NLP
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                targets = batch["labels"].to(self.device)

            # Forward pass
            if self.task_type == "vision":
                outputs = self.model(inputs)
            else:
                outputs = self.model(**inputs).logits

            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        accuracy = 100.0 * correct / total
        return accuracy

    def save_checkpoint(self, epoch, train_loss, val_acc, test_acc, filename):
        """
        Save checkpoint.

        Args:
            epoch: Current epoch
            train_loss: Training loss
            val_acc: Validation accuracy
            test_acc: Test accuracy
            filename: Checkpoint filename
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        print(f"  → Checkpoint saved: {filepath}")

    def train(self, train_loader, val_loader, test_loader, epochs,
              checkpoint_epochs=None, experiment_name="experiment"):
        """
        Run full training loop with checkpoints.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            epochs: Number of epochs to train
            checkpoint_epochs: List of epochs to save checkpoints (e.g., [10, 20, 30, 40, 50])
            experiment_name: Name for checkpoint files

        Returns:
            Dictionary with final results
        """
        if checkpoint_epochs is None:
            checkpoint_epochs = []

        start_time = time.time()

        # Training history
        train_losses = []
        val_accuracies = []
        checkpoint_results = []

        print(f"\n{'='*60}")
        print(f"Training: {experiment_name}")
        print(f"Epochs: {epochs}, Checkpoints at: {checkpoint_epochs}")
        print(f"Scheduler: {'Yes' if self.scheduler is not None else 'No'}")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")

            # Train
            train_loss = self.train_epoch(train_loader)
            train_losses.append({'epoch': epoch, 'loss': train_loss})

            # Validate
            val_acc = self.evaluate(val_loader)
            val_accuracies.append({'epoch': epoch, 'accuracy': val_acc})

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            print(f"  Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")

            # Update scheduler if exists
            if self.scheduler is not None:
                self.scheduler.step()

            # Save checkpoint if requested
            if epoch in checkpoint_epochs:
                test_acc = self.evaluate(test_loader)
                print(f"  Test Acc at epoch {epoch}: {test_acc:.2f}%")

                checkpoint_filename = f"{experiment_name}_epoch{epoch}.pt"
                self.save_checkpoint(epoch, train_loss, val_acc, test_acc, checkpoint_filename)

                checkpoint_results.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_accuracy': val_acc,
                    'test_accuracy': test_acc,
                    'learning_rate': current_lr
                })

        # Final evaluation
        final_test_acc = self.evaluate(test_loader)
        training_time = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Final Test Accuracy: {final_test_acc:.2f}%")
        print(f"Total Training Time: {training_time:.1f}s ({training_time/60:.1f}min)")
        print(f"{'='*60}\n")

        # Prepare results
        results = {
            'experiment_name': experiment_name,
            'total_epochs': epochs,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'final_test_accuracy': final_test_acc,
            'training_time': training_time,
            'checkpoint_results': checkpoint_results,
            'used_scheduler': self.scheduler is not None
        }

        # Save results JSON
        results_filename = f"{experiment_name}_results.json"
        results_path = self.checkpoint_dir / results_filename

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✓ Results saved: {results_path}")

        return results
