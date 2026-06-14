"""Shared training utilities."""

import tensorflow as tf

from solar_panel_cv.config import TrainingConfig


def early_stopping(training: TrainingConfig) -> tf.keras.callbacks.EarlyStopping:
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.01,
        patience=training.early_stopping_patience,
        verbose=1,
        restore_best_weights=True,
    )
