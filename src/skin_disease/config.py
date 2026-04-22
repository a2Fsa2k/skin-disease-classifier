"""Configuration helpers for the mini project."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    """Project settings loaded from a JSON file."""

    model_name: str
    checkpoint_path: str
    image_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    validation_split: float
    random_seed: int
    freeze_backbone: bool
    num_workers: int
    top_k: int
    class_names: list[str]

    @classmethod
    def from_json(cls, path: Path) -> "AppConfig":
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return cls(**data)
