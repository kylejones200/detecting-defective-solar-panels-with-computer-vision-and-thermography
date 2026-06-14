"""Train a binary clean vs. not-clean classifier."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from solar_panel_cv.context import PipelineContext
from solar_panel_cv.data import to_binary_labels
from solar_panel_cv.datasets import load_image_datasets
from solar_panel_cv.models import create_binary_model
from solar_panel_cv.plotting import finalize_plot, plot_images_with_predictions
from solar_panel_cv.training import early_stopping


def train_binary_classifier(ctx: PipelineContext) -> None:
    training = ctx.config.training
    train_ds, val_ds, class_names = load_image_datasets(
        ctx.faulty_panel_dir(),
        training,
        cache=False,
    )
    ctx.original_class_names = class_names

    train_ds_binary = train_ds.map(
        lambda images, labels: to_binary_labels(images, labels, class_names)
    )
    val_ds_binary = val_ds.map(
        lambda images, labels: to_binary_labels(images, labels, class_names)
    )
    autotune = tf.data.AUTOTUNE
    ctx.train_ds_binary = train_ds_binary.cache().shuffle(1000).prefetch(buffer_size=autotune)
    ctx.val_ds_binary = val_ds_binary.cache().prefetch(buffer_size=autotune)
    ctx.class_names = ["Clean", "Not Clean"]

    model, base = create_binary_model(training.img_height, training.img_width)
    ctx.model = model
    ctx.base_model = base
    model.compile(
        optimizer=tf.keras.optimizers.Adam(training.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        ctx.train_ds_binary,
        validation_data=ctx.val_ds_binary,
        epochs=training.epochs,
        callbacks=[early_stopping(training)],
    )

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.tight_layout()
    finalize_plot(ctx, "10_binary_training")

    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in ctx.val_ds_binary:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend((predictions > 0.5).astype(int).flatten())

    print(classification_report(y_true, y_pred, target_names=ctx.class_names))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Binary classifier confusion matrix")
    finalize_plot(ctx, "10_binary_confusion_matrix")
    plot_images_with_predictions(
        ctx, model, ctx.val_ds_binary, ctx.class_names, name="10_binary_predictions"
    )
