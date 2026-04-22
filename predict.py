"""Command-line prediction helper for single images."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from src.skin_disease.config import AppConfig
from src.skin_disease.inference import Predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict the skin disease class for one image.")
    parser.add_argument("image", type=Path, help="Path to the input image.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.json"),
        help="Path to the JSON config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = AppConfig.from_json(config_path)
    predictor = Predictor.from_config(config, project_root=project_root)

    image_path = args.image if args.image.is_absolute() else project_root / args.image
    image = Image.open(image_path).convert("RGB")
    result = predictor.predict(image)

    print(f"Prediction : {result.label}")
    print(f"Confidence : {result.confidence * 100:.2f}%")
    print("Top predictions:")
    for class_name, score in result.top_predictions:
        print(f"  - {class_name}: {score * 100:.2f}%")
    if result.warning:
        print(f"Warning    : {result.warning}")


if __name__ == "__main__":
    main()
