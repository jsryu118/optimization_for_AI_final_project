"""
Utility functions for reproducibility and logging.
"""
import random
import json
import os
from pathlib import Path
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ResultLogger:
    """
    Simple logger for tracking and saving experiment results.
    """
    def __init__(self, save_dir: str = "results"):
        """
        Initialize the logger.

        Args:
            save_dir: Directory to save results
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {
            "train_losses": [],
            "val_accuracies": [],
            "test_accuracy": None,
            "training_time": None
        }

    def log_epoch(self, epoch: int, train_loss: float, val_acc: float):
        """
        Log metrics for a single epoch.

        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_acc: Validation accuracy
        """
        self.metrics["train_losses"].append({
            "epoch": epoch,
            "loss": train_loss
        })
        self.metrics["val_accuracies"].append({
            "epoch": epoch,
            "accuracy": val_acc
        })

    def log_test(self, test_acc: float):
        """
        Log final test accuracy.

        Args:
            test_acc: Test accuracy
        """
        self.metrics["test_accuracy"] = test_acc

    def log_training_time(self, time_seconds: float):
        """
        Log total training time.

        Args:
            time_seconds: Training time in seconds
        """
        self.metrics["training_time"] = time_seconds

    def save(self, filename: str):
        """
        Save metrics to JSON file.

        Args:
            filename: Name of the output file (e.g., 'cifar10_resnet20_sgd_0.01.json')
        """
        filepath = self.save_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Results saved to {filepath}")

    def get_best_val_acc(self) -> float:
        """
        Get the best validation accuracy.

        Returns:
            Best validation accuracy
        """
        if not self.metrics["val_accuracies"]:
            return 0.0
        return max(entry["accuracy"] for entry in self.metrics["val_accuracies"])


def get_device() -> torch.device:
    """
    Get the available device (CUDA if available, else CPU).

    Returns:
        PyTorch device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
