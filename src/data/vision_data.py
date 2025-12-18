"""
Data loaders for vision tasks.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path


def get_cifar10_loaders(batch_size: int = 128, data_dir: str = "./data"):
    """
    Get CIFAR-10 train, validation, and test data loaders.

    Args:
        batch_size: Batch size for data loaders
        data_dir: Directory to download/store the dataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # CIFAR-10 normalization constants
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Test/validation transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Download and load training data
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    # Download and load test data
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )

    # Split training data into train and validation (45k train, 5k val)
    train_size = 45000
    val_size = 5000
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Update validation dataset transform
    val_dataset.dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=test_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def get_oxford_pet_loaders(batch_size: int = 64, data_dir: str = "./data"):
    """
    Get Oxford-IIIT Pet train, validation, and test data loaders.

    Args:
        batch_size: Batch size for data loaders
        data_dir: Directory to download/store the dataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # ImageNet normalization (for pre-trained models)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Test/validation transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Download and load training data
    train_dataset = datasets.OxfordIIITPet(
        root=data_dir,
        split="trainval",
        download=True,
        transform=train_transform
    )

    # Download and load test data
    test_dataset = datasets.OxfordIIITPet(
        root=data_dir,
        split="test",
        download=True,
        transform=test_transform
    )

    # Split training data into train and validation (90% train, 10% val)
    total_train = len(train_dataset)
    train_size = int(0.9 * total_train)
    val_size = total_train - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Update validation dataset transform
    val_dataset.dataset = datasets.OxfordIIITPet(
        root=data_dir,
        split="trainval",
        download=False,
        transform=test_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
