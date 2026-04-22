"""Small utility helpers."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed the common random number generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for a file if it does not exist."""

    path.parent.mkdir(parents=True, exist_ok=True)
