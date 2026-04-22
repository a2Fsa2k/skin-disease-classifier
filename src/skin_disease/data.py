"""Dataset utilities for HAM10000."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .constants import DX_TO_CLASS_NAME, DX_TO_INDEX


SUPPORTED_IMAGE_FOLDERS = [
    "HAM10000_images_part_1",
    "HAM10000_images_part_2",
    "HAM10000_images_train",
    "HAM10000_images_test",
    "images",
]


def resolve_image_paths(data_dir: Path) -> dict[str, Path]:
    """Build a map from image id to image path for the dataset directory."""

    image_paths: dict[str, Path] = {}
    for folder_name in SUPPORTED_IMAGE_FOLDERS:
        folder = data_dir / folder_name
        if not folder.exists():
            continue
        for image_path in folder.glob("*.jpg"):
            image_paths[image_path.stem] = image_path
    if not image_paths:
        for image_path in data_dir.rglob("*.jpg"):
            image_paths[image_path.stem] = image_path
    return image_paths


def load_metadata(data_dir: Path) -> pd.DataFrame:
    """Load HAM10000 metadata and attach resolved image paths."""

    metadata_path = data_dir / "HAM10000_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)[["image_id", "dx"]].copy()
    image_map = resolve_image_paths(data_dir)
    df["image_path"] = df["image_id"].map(image_map)
    df = df.dropna(subset=["image_path"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No matching images were found for the metadata entries.")

    df["label"] = df["dx"].map(DX_TO_INDEX)
    df["class_name"] = df["dx"].map(DX_TO_CLASS_NAME)
    return df


def create_splits(df: pd.DataFrame, validation_split: float, random_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create train and validation splits with class balance."""

    train_df, val_df = train_test_split(
        df,
        test_size=validation_split,
        random_state=random_seed,
        stratify=df["label"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


class HAM10000Dataset(Dataset):
    """PyTorch dataset for HAM10000 image classification."""

    def __init__(self, dataframe: pd.DataFrame, transform=None) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])
