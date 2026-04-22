"""Model creation and checkpoint loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models


def create_model(
    model_name: str,
    num_classes: int,
    freeze_backbone: bool = False,
    pretrained: bool = True,
) -> nn.Module:
    """Create a torchvision classifier suitable for CPU inference."""

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        if model_name == "resnet18":
            for parameter in model.fc.parameters():
                parameter.requires_grad = True
        else:
            for parameter in model.classifier[-1].parameters():
                parameter.requires_grad = True

    return model


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Checkpoint format is not supported.")


def load_checkpoint(
    checkpoint_path: Path,
    model_name: str,
    class_names: list[str],
    device: torch.device,
    freeze_backbone: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load either a legacy state_dict or the new structured checkpoint format."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Run train.py first or place a compatible checkpoint in the configured path."
        )

    model = create_model(
        model_name=model_name,
        num_classes=len(class_names),
        freeze_backbone=freeze_backbone,
        pretrained=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    return model.to(device), metadata
