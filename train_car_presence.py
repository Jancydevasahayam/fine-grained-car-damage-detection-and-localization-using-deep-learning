"""Train car presence model (car / no_car) using transfer learning.

Expected dataset layout:

datasets/car_presence/
  train/
    car/
    no_car/
  val/
    car/
    no_car/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils.training_utils import (
    compute_balanced_class_weights,
    ensure_dataset_layout,
    ensure_non_empty_classes,
    pretty_class_distribution,
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 14
EPOCHS_FINE_TUNE = 6
CLASS_ORDER = ["no_car", "car"]
SEED = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets" / "car_presence"
MODEL_OUT = BASE_DIR / "models" / "car_presence_model.keras"


def build_model(input_size: tuple[int, int]) -> tuple[tf.keras.Model, tf.keras.Model]:
    base_model = MobileNetV2(
        input_shape=(input_size[0], input_size[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(input_size[0], input_size[1], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model, base_model


def _make_generators(
    data_dir: Path,
    input_size: tuple[int, int],
    batch_size: int,
) -> tuple:
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        rotation_range=14,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        brightness_range=(0.85, 1.15),
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=input_size,
        batch_size=batch_size,
        classes=CLASS_ORDER,
        class_mode="binary",
        shuffle=True,
        seed=SEED,
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=input_size,
        batch_size=batch_size,
        classes=CLASS_ORDER,
        class_mode="binary",
        shuffle=False,
    )
    return train_gen, val_gen


def train_model(
    data_dir: Path,
    model_out: Path,
    img_size: int = 224,
    batch_size: int = 32,
    epochs_head: int = 14,
    epochs_finetune: int = 6,
) -> None:
    tf.random.set_seed(SEED)

    input_size = (img_size, img_size)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    ensure_dataset_layout(data_dir, CLASS_ORDER)
    ensure_non_empty_classes(train_dir, CLASS_ORDER, "train")
    ensure_non_empty_classes(val_dir, CLASS_ORDER, "val")

    print("Car presence dataset distribution:")
    print(f"  train -> {pretty_class_distribution(train_dir, CLASS_ORDER)}")
    print(f"  val   -> {pretty_class_distribution(val_dir, CLASS_ORDER)}")

    train_gen, val_gen = _make_generators(data_dir, input_size, batch_size)
    class_weights = compute_balanced_class_weights(train_dir, CLASS_ORDER)
    print("Class weights:", class_weights)

    model, base_model = build_model(input_size=input_size)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_out,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
    ]

    print(f"Training head for {epochs_head} epochs...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_head,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    print(f"Fine-tuning backbone for {epochs_finetune} epochs...")
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_finetune,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    model.save(model_out)
    print(f"Car presence model saved to: {model_out}")
    print("Class indices:", train_gen.class_indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train car presence classifier.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model-out", type=Path, default=MODEL_OUT)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE[0])
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs-head", type=int, default=EPOCHS_HEAD)
    parser.add_argument("--epochs-finetune", type=int, default=EPOCHS_FINE_TUNE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(
        data_dir=args.data_dir,
        model_out=args.model_out,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs_head=args.epochs_head,
        epochs_finetune=args.epochs_finetune,
    )


if __name__ == "__main__":
    main()
