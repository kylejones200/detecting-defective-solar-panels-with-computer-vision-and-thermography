"""Train a multiclass classifier on the faulty solar panel dataset."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

from solar_panel_cv.context import PipelineContext
from solar_panel_cv.datasets import load_image_datasets
from solar_panel_cv.models import create_mobilenet_model
from solar_panel_cv.plotting import plot_images_with_predictions, plot_training_history
from solar_panel_cv.training import early_stopping


def train_multiclass_faulty_panel(ctx: PipelineContext) -> None:
    training = ctx.config.training
    train_ds, val_ds, class_names = load_image_datasets(
        ctx.faulty_panel_dir(),
        training,
        cache=True,
    )
    ctx.train_ds = train_ds
    ctx.val_ds = val_ds
    ctx.class_names = class_names
    print("Classes:", class_names)

    model, base = create_mobilenet_model(len(class_names), training.img_height, training.img_width)
    ctx.model = model
    ctx.base_model = base
    model.compile(
        optimizer=tf.keras.optimizers.Adam(training.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=training.epochs,
        callbacks=[early_stopping(training)],
    )
    plot_training_history(ctx, history, name="07_multiclass_training")

    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))
    print(classification_report(y_true, y_pred, target_names=class_names))
    plot_images_with_predictions(ctx, model, val_ds, class_names, name="07_multiclass_predictions")
