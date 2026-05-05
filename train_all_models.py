"""Train all models in pipeline order: car presence -> damage -> parts."""

from __future__ import annotations

import argparse
from pathlib import Path

from train_car_presence import train_model as train_car_presence_model
from train_damage import train_model as train_damage_model
from train_part import train_model as train_part_model

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = BASE_DIR / "datasets"
DEFAULT_MODELS_DIR = BASE_DIR / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train all models in the required detection order."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    models_dir = args.models_dir
    img_size = args.img_size
    batch_size = args.batch_size

    print("== Stage 1/3: Training car presence model ==")
    train_car_presence_model(
        data_dir=dataset_root / "car_presence",
        model_out=models_dir / "car_presence_model.keras",
        img_size=img_size,
        batch_size=batch_size,
    )

    print("== Stage 2/3: Training damage model ==")
    train_damage_model(
        data_dir=dataset_root / "damage",
        model_out=models_dir / "damage_model.keras",
        img_size=img_size,
        batch_size=batch_size,
    )

    print("== Stage 3/3: Training part model ==")
    train_part_model(
        data_dir=dataset_root / "parts",
        model_out=models_dir / "part_classifier_mobilenetv2.keras",
        img_size=img_size,
        batch_size=batch_size,
    )

    print("All models trained and saved successfully.")


if __name__ == "__main__":
    main()
