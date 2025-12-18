"""
Training and evaluation loop (no scheduler logic).
"""
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import ResultLogger
from optimizers.dog import get_dog_averager


class Trainer:
    """
    Trainer class for running experiments without learning rate schedulers.
    """

    def __init__(
        self,
        model,
        optimizer,
        device,
        task_type="vision",
        use_dog_averaging=False
    ):
        """
        Initialize the trainer.

        Args:
            model: PyTorch model to train
            optimizer: Optimizer instance
            device: Device to run training on
            task_type: Type of task ('vision' or 'nlp')
            use_dog_averaging: Whether to use DOG iterate averaging
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.task_type = task_type
        self.criterion = nn.CrossEntropyLoss()
        self.logger = ResultLogger()

        # Setup DOG averaging if needed
        self.use_dog_averaging = use_dog_averaging
        self.averager = None
        if use_dog_averaging:
            self.averager = get_dog_averager(model)

    def train_epoch(self, train_loader):
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss
        """
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

            # Update averager if using DOG
            if self.averager is not None:
                self.averager.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def evaluate(self, data_loader, use_averaged_model=False):
        """
        Evaluate the model on a data loader.

        Args:
            data_loader: Data loader for evaluation
            use_averaged_model: Whether to use averaged model (for DOG)

        Returns:
            Accuracy as a percentage
        """
        # Select model to evaluate
        if use_averaged_model and self.averager is not None:
            model = self.averager.averaged_model
        else:
            model = self.model

        model.eval()

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
                outputs = model(inputs)
            else:
                outputs = model(**inputs).logits

            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        accuracy = 100.0 * correct / total
        return accuracy

    def train(self, train_loader, val_loader, test_loader, epochs):
        """
        Run full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            epochs: Number of epochs to train

        Returns:
            Dictionary with final results
        """
        print(f"Starting training for {epochs} epochs...")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_acc = self.evaluate(val_loader, use_averaged_model=self.use_dog_averaging)

            # Log metrics
            self.logger.log_epoch(epoch, train_loss, val_acc)

            print(f"Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Final test evaluation
        test_acc = self.evaluate(test_loader, use_averaged_model=self.use_dog_averaging)
        training_time = time.time() - start_time

        self.logger.log_test(test_acc)
        self.logger.log_training_time(training_time)

        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Test Accuracy: {test_acc:.2f}%")

        return {
            "test_accuracy": test_acc,
            "best_val_accuracy": self.logger.get_best_val_acc(),
            "training_time": training_time
        }

    def save_results(self, filename):
        """
        Save results to file.

        Args:
            filename: Name of the output file
        """
        self.logger.save(filename)
