"""Tune CLIP temperature and classification threshold."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from solar_panel_cv.context import PipelineContext
from solar_panel_cv.datasets import load_image_datasets
from solar_panel_cv.plotting import finalize_plot, plot_clip_predictions


def clip_temperature_tuning(ctx: PipelineContext) -> None:
    training = ctx.config.training
    clip_cfg = ctx.config.clip
    _, val_ds, class_names = load_image_datasets(
        ctx.binary_panel_dir(),
        training,
        img_height=training.clip_img_size,
        img_width=training.clip_img_size,
    )
    ctx.class_names = class_names
    clip = ctx.ensure_clip()

    best_accuracy = 0.0
    best_temperature: float | None = None
    best_threshold: float | None = None
    thresholds = np.arange(
        clip_cfg.threshold_search_start,
        clip_cfg.threshold_search_end,
        clip_cfg.threshold_search_step,
    )

    for temp in clip_cfg.temperature_candidates:
        print(f"Testing temperature: {temp}")
        y_true: list[int] = []
        y_pred_probs: list[float] = []
        for images, labels in val_ds:
            predictions = clip.predict_binary(images, temperature=temp)
            y_true.extend(labels.numpy())
            y_pred_probs.extend(predictions[:, 1])
        y_true_arr = np.array(y_true)
        y_pred_probs_arr = np.array(y_pred_probs)
        for threshold in thresholds:
            y_pred = (y_pred_probs_arr > threshold).astype(int)
            accuracy = float(np.mean(y_pred == y_true_arr))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_temperature = temp
                best_threshold = float(threshold)

    assert best_temperature is not None and best_threshold is not None
    ctx.best_temperature = best_temperature
    ctx.best_threshold = best_threshold
    print(f"Best temperature: {best_temperature}")
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Best accuracy: {best_accuracy:.3f}")

    y_true: list[int] = []
    y_pred_probs: list[float] = []
    for images, labels in val_ds:
        predictions = clip.predict_binary(images, temperature=best_temperature)
        y_true.extend(labels.numpy())
        y_pred_probs.extend(predictions[:, 1])
    y_true_arr = np.array(y_true)
    y_pred_probs_arr = np.array(y_pred_probs)
    y_pred = (y_pred_probs_arr > best_threshold).astype(int)
    print(classification_report(y_true_arr, y_pred, target_names=class_names))

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_true_arr, y_pred), annot=True, fmt="d", cmap="Blues")
    finalize_plot(ctx, "14_clip_temperature_confusion_matrix")
    plot_clip_predictions(
        ctx,
        val_ds,
        class_names,
        lambda images: clip.predict_binary(images, temperature=best_temperature),
        threshold=best_threshold,
        name="14_clip_temperature_predictions",
    )
