"""Minimal training entry point for the HAM10000 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.skin_disease.config import AppConfig
from src.skin_disease.training import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a skin disease classifier on HAM10000.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Folder containing HAM10000_metadata.csv and the image files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.json"),
        help="Path to the training config JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    data_dir = args.data_dir if args.data_dir.is_absolute() else project_root / args.data_dir
    config = AppConfig.from_json(config_path)
    run_training(config=config, data_dir=data_dir, project_root=project_root)


if __name__ == "__main__":
    main()
