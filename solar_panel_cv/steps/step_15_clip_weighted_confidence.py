"""Evaluate CLIP with class-weighted confidence scoring."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from solar_panel_cv.context import PipelineContext
from solar_panel_cv.datasets import load_image_datasets
from solar_panel_cv.plotting import finalize_plot, plot_clip_predictions


def clip_weighted_confidence(ctx: PipelineContext) -> None:
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

    class_counts = np.zeros(2)
    for _, labels in val_ds:
        for label in labels:
            class_counts[int(label.numpy())] += 1
    class_weights = class_counts.sum() / (2 * class_counts)
    ctx.class_weights = class_weights
    print(f"Class weights: Clean={class_weights[0]:.2f}, Not Clean={class_weights[1]:.2f}")

    y_true: list[int] = []
    y_pred_probs: list[np.ndarray] = []
    for images, labels in val_ds:
        predictions = clip.predict_with_confidence(images, class_weights)
        y_true.extend(labels.numpy())
        y_pred_probs.extend(predictions)
    y_true_arr = np.array(y_true)
    y_pred_probs_arr = np.array(y_pred_probs)

    thresholds = np.arange(
        clip_cfg.threshold_search_start,
        clip_cfg.threshold_search_end,
        clip_cfg.threshold_search_step,
    )
    best_f1 = 0.0
    best_threshold: float | None = None
    for threshold in thresholds:
        y_pred = (y_pred_probs_arr[:, 1] > threshold).astype(int)
        f1 = f1_score(y_true_arr, y_pred, average="weighted")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    assert best_threshold is not None
    ctx.best_threshold = best_threshold
    y_pred = (y_pred_probs_arr[:, 1] > best_threshold).astype(int)
    print(f"Best threshold: {best_threshold:.3f}")
    print(classification_report(y_true_arr, y_pred, target_names=class_names))

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_true_arr, y_pred), annot=True, fmt="d", cmap="Blues")
    finalize_plot(ctx, "15_clip_weighted_confusion_matrix")
    plot_clip_predictions(
        ctx,
        val_ds,
        class_names,
        lambda images: clip.predict_with_confidence(images, class_weights),
        threshold=best_threshold,
        name="15_clip_weighted_predictions",
    )

    plt.figure(figsize=(10, 6))
    plt.hist(
        y_pred_probs_arr[y_true_arr == 0][:, 1],
        alpha=0.5,
        label="Clean",
        bins=20,
        density=True,
    )
    plt.hist(
        y_pred_probs_arr[y_true_arr == 1][:, 1],
        alpha=0.5,
        label="Not Clean",
        bins=20,
        density=True,
    )
    plt.axvline(best_threshold, color="r", linestyle="--")
    plt.legend()
    finalize_plot(ctx, "15_clip_weighted_probability_distribution")
