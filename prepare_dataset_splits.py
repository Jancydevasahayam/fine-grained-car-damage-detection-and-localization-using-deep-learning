"""Create train/val folder splits from a raw class-organized image dataset.

Example:
python prepare_dataset_splits.py --source-dir raw_datasets/parts --output-dir datasets/parts
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare train/val image dataset splits.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Remove existing output split folders before creating new ones.",
    )
    return parser.parse_args()


def collect_images(class_dir: Path) -> list[Path]:
    images: list[Path] = []
    for path in class_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            images.append(path)
    return images


def ensure_dirs(output_dir: Path, class_names: list[str]) -> None:
    for split in ["train", "val"]:
        for class_name in class_names:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)


def clear_existing_output(output_dir: Path) -> None:
    for split in ["train", "val"]:
        split_dir = output_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)


def copy_split(images: list[Path], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for idx, src in enumerate(images, start=1):
        out_name = f"{idx:05d}_{src.name}"
        shutil.copy2(src, target_dir / out_name)


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir
    output_dir = args.output_dir
    train_ratio = args.train_ratio
    seed = args.seed

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    if train_ratio <= 0.0 or train_ratio >= 1.0:
        raise ValueError("train-ratio must be between 0 and 1 (exclusive).")

    class_names = sorted(
        [path.name for path in source_dir.iterdir() if path.is_dir()]
    )
    if not class_names:
        raise ValueError(
            f"No class folders found in source directory: {source_dir}"
        )

    if args.clear_output:
        clear_existing_output(output_dir)

    ensure_dirs(output_dir, class_names)

    random.seed(seed)
    for class_name in class_names:
        class_dir = source_dir / class_name
        images = collect_images(class_dir)
        if not images:
            print(f"Skipping empty class '{class_name}'")
            continue

        random.shuffle(images)
        split_idx = max(1, int(len(images) * train_ratio))
        if split_idx >= len(images):
            split_idx = len(images) - 1

        train_images = images[:split_idx]
        val_images = images[split_idx:]

        copy_split(train_images, output_dir / "train" / class_name)
        copy_split(val_images, output_dir / "val" / class_name)

        print(
            f"{class_name}: train={len(train_images)}, val={len(val_images)} "
            f"(total={len(images)})"
        )

    print(f"Dataset splits created at: {output_dir}")


if __name__ == "__main__":
    main()
