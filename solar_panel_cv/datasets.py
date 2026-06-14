"""TensorFlow dataset loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tensorflow as tf

from solar_panel_cv.config import TrainingConfig


def load_image_datasets(
    directory: Path | str,
    training: TrainingConfig,
    *,
    img_height: int | None = None,
    img_width: int | None = None,
    cache: bool = False,
    shuffle_train: bool = True,
) -> tuple[Any, Any, list[str]]:
    directory = str(directory)
    height = img_height if img_height is not None else training.img_height
    width = img_width if img_width is not None else training.img_width
    common = {
        "directory": directory,
        "validation_split": training.validation_split,
        "image_size": (height, width),
        "batch_size": training.batch_size,
        "seed": training.seed,
    }
    train_ds = tf.keras.utils.image_dataset_from_directory(
        subset="training",
        shuffle=shuffle_train,
        **common,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        subset="validation",
        shuffle=False,
        **common,
    )
    if cache:
        autotune = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
        val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    return train_ds, val_ds, train_ds.class_names
