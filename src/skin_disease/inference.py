"""Prediction helpers for the Streamlit app and CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .config import AppConfig
from .model import load_checkpoint


@dataclass(slots=True)
class PredictionResult:
    label: str
    confidence: float
    top_predictions: list[tuple[str, float]]
    warning: str | None = None


class Predictor:
    """Thin inference wrapper around a PyTorch image classifier."""

    def __init__(
        self,
        model: torch.nn.Module,
        transform: transforms.Compose,
        class_names: list[str],
        top_k: int,
        device: torch.device,
        warning: str | None = None,
    ) -> None:
        self.model = model.eval()
        self.transform = transform
        self.class_names = class_names
        self.top_k = top_k
        self.device = device
        self.warning = warning

    @classmethod
    def from_config(cls, config: AppConfig, project_root: Path) -> "Predictor":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = project_root / config.checkpoint_path
        model_name = "resnet18" if checkpoint_path.name.startswith("model_resnet18") else config.model_name
        model, metadata = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            class_names=config.class_names,
            device=device,
            freeze_backbone=False,
        )
        transform = transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        warning = None
        if isinstance(metadata, dict) and "class_names" not in metadata:
            warning = "Loaded a legacy checkpoint. Retrain with train.py for a cleaner 2025-ready checkpoint."
        return cls(
            model=model,
            transform=transform,
            class_names=config.class_names,
            top_k=config.top_k,
            device=device,
            warning=warning,
        )

    def predict(self, image: Image.Image) -> PredictionResult:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]

        top_scores, top_indices = torch.topk(probabilities, k=min(self.top_k, len(self.class_names)))
        top_predictions = [
            (self.class_names[index], float(score))
            for score, index in zip(top_scores.cpu().tolist(), top_indices.cpu().tolist())
        ]
        label, confidence = top_predictions[0]
        return PredictionResult(
            label=label,
            confidence=confidence,
            top_predictions=top_predictions,
            warning=self.warning,
        )
