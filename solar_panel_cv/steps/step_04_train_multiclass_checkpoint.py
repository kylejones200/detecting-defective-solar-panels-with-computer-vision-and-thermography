"""Train multiclass model with early stopping."""

from __future__ import annotations

import tensorflow as tf

from solar_panel_cv.context import PipelineContext
from solar_panel_cv.training import early_stopping


def train_multiclass_checkpoint(ctx: PipelineContext) -> None:
    if ctx.model is None or ctx.train_ds is None or ctx.val_ds is None:
        raise RuntimeError("Run build_vgg16_and_explore before train_multiclass_checkpoint.")

    training = ctx.config.training
    ctx.model.compile(
        optimizer=tf.keras.optimizers.Adam(training.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    ctx.model.fit(
        ctx.train_ds,
        validation_data=ctx.val_ds,
        epochs=training.fine_tune_epochs,
        callbacks=[early_stopping(training)],
    )
