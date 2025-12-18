"""
Data loaders for NLP tasks.
"""
import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_sst2_loaders(batch_size: int = 32, model_name: str = "distilbert-base-uncased"):
    """
    Get SST-2 train, validation, and test data loaders.

    Args:
        batch_size: Batch size for data loaders
        model_name: Name of the model (for tokenizer)

    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_labels)
    """
    # Load SST-2 dataset from GLUE benchmark
    dataset = load_dataset("glue", "sst2")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        """Tokenize the text data."""
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    # Tokenize all splits
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["sentence", "idx"]
    )

    # Rename 'label' to 'labels' for consistency with transformers
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # Set format for PyTorch
    tokenized_datasets.set_format("torch")

    # Get train, validation, and test splits
    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]

    # Note: SST-2 test set doesn't have labels, so we use validation as test
    # For proper evaluation, split validation into val and test
    val_size = len(val_dataset)
    test_size = val_size // 2
    val_size = val_size - test_size

    val_dataset, test_dataset = torch.utils.data.random_split(
        val_dataset,
        [val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    # Note: Using num_workers=2 for NLP to avoid tokenizer parallelism issues
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # SST-2 is binary classification (2 labels)
    num_labels = 2

    return train_loader, val_loader, test_loader, num_labels
