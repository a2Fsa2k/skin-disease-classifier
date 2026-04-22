"""Simple training pipeline for a small HAM10000 mini project."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from .config import AppConfig
from .data import HAM10000Dataset, create_splits, load_metadata
from .model import create_model
from .utils import ensure_parent_dir, seed_everything


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Return simple train and validation transforms."""

    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, val_transform


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def run_training(config: AppConfig, data_dir: Path, project_root: Path) -> None:
    """Train a classifier and save a modern checkpoint."""

    seed_everything(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata = load_metadata(data_dir)
    train_df, val_df = create_splits(
        metadata,
        validation_split=config.validation_split,
        random_seed=config.random_seed,
    )

    train_transform, val_transform = build_transforms(config.image_size)
    train_dataset = HAM10000Dataset(train_df, transform=train_transform)
    val_dataset = HAM10000Dataset(val_df, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = create_model(
        model_name=config.model_name,
        num_classes=len(config.class_names),
        freeze_backbone=config.freeze_backbone,
        pretrained=True,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=config.learning_rate)

    best_accuracy = 0.0
    checkpoint_path = project_root / config.checkpoint_path
    ensure_parent_dir(checkpoint_path)

    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                {
                    "model_name": config.model_name,
                    "image_size": config.image_size,
                    "class_names": config.class_names,
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint to {checkpoint_path}")
