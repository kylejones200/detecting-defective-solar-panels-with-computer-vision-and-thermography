"""Visualization helpers for model evaluation."""

from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from solar_panel_cv.context import PipelineContext


def finalize_plot(ctx: PipelineContext, name: str) -> None:
    if ctx.config.output.save_plots:
        path = ctx.output_dir / f"{name}.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        print(f"Saved plot: {path}")
    if ctx.config.output.show_plots:
        plt.show()
    plt.close()


def plot_sample_grid(
    ctx: PipelineContext,
    dataset,
    class_names: list[str],
    *,
    name: str,
    num_images: int = 25,
    grid: tuple[int, int] = (5, 5),
) -> None:
    rows, cols = grid
    plt.figure(figsize=(15, 15))
    for images, labels in dataset.take(1):
        for i in range(min(num_images, len(images))):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i].numpy()])
            plt.axis("off")
    plt.tight_layout()
    finalize_plot(ctx, name)


def plot_comparison_predictions(
    ctx: PipelineContext,
    val_ds,
    class_names: list[str],
    mobile_model,
    clip_predictor,
    *,
    name: str = "comparison_predictions",
    num_images: int = 25,
) -> None:
    plt.figure(figsize=(20, 20))
    for images, labels in val_ds.take(1):
        mobile_predictions = mobile_model.predict(images, verbose=0)
        clip_predictions = clip_predictor.predict_multiclass(images, class_names)
        for i in range(min(num_images, len(images))):
            plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            mobile_pred = int(np.argmax(mobile_predictions[i]))
            clip_pred = int(np.argmax(clip_predictions[i]))
            actual_class = int(labels[i].numpy())
            title = f"Actual: {class_names[actual_class]}\n"
            title += f"MobileNet: {class_names[mobile_pred]}\n"
            title += f"CLIP: {class_names[clip_pred]}"
            plt.title(title, fontsize=10)
            plt.axis("off")
    plt.tight_layout()
    finalize_plot(ctx, name)


def plot_images_with_predictions(
    ctx: PipelineContext,
    model,
    dataset,
    class_names: list[str],
    *,
    name: str = "predictions",
    num_images: int = 25,
) -> None:
    plt.figure(figsize=(20, 20))
    for images, labels in dataset.take(1):
        predictions = model.predict(images, verbose=0)
        for i in range(min(num_images, len(images))):
            plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            if predictions.shape[-1] == 1:
                predicted_class = int(predictions[i][0] > 0.5)
            else:
                predicted_class = int(np.argmax(predictions[i]))
            actual_class = int(labels[i].numpy())
            color = "green" if predicted_class == actual_class else "red"
            plt.title(
                f"Actual: {class_names[actual_class]}\nPredicted: {class_names[predicted_class]}",
                color=color,
                fontsize=10,
            )
            plt.axis("off")
    plt.tight_layout()
    finalize_plot(ctx, name)


def plot_clip_predictions(
    ctx: PipelineContext,
    dataset,
    class_names: list[str],
    predict_fn: Callable,
    *,
    threshold: float,
    name: str = "clip_predictions",
    num_images: int = 25,
) -> None:
    plt.figure(figsize=(20, 20))
    for images, labels in dataset.take(1):
        predictions = predict_fn(images)
        predicted_classes = (predictions[:, 1] > threshold).astype(int)
        for i in range(min(num_images, len(images))):
            plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            pred_class = class_names[predicted_classes[i]]
            true_class = class_names[int(labels[i].numpy())]
            prob = float(predictions[i][predicted_classes[i]])
            color = "green" if pred_class == true_class else "red"
            plt.title(
                f"True: {true_class}\nPred: {pred_class}\nConf: {prob:.2f}",
                color=color,
                fontsize=10,
            )
            plt.axis("off")
    plt.tight_layout()
    finalize_plot(ctx, name)


def plot_training_history(ctx: PipelineContext, history, *, name: str = "training_history") -> None:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    finalize_plot(ctx, name)
