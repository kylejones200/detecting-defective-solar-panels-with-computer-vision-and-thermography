"""Evaluate CLIP binary classification with ROC analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from solar_panel_cv.context import PipelineContext
from solar_panel_cv.data import to_binary_labels
from solar_panel_cv.datasets import load_image_datasets
from solar_panel_cv.plotting import finalize_plot, plot_clip_predictions


def clip_binary_evaluation(ctx: PipelineContext) -> None:
    training = ctx.config.training
    train_ds, val_ds, class_names = load_image_datasets(
        ctx.faulty_panel_dir(),
        training,
        img_height=training.clip_img_size,
        img_width=training.clip_img_size,
    )
    ctx.original_class_names = class_names
    val_ds_binary = val_ds.map(
        lambda images, labels: to_binary_labels(images, labels, class_names)
    )
    ctx.val_ds_binary = val_ds_binary
    ctx.class_names = ["Clean", "Not Clean"]

    clip = ctx.ensure_clip()
    y_true: list[int] = []
    y_pred_probs: list[float] = []
    for images, labels in val_ds_binary:
        predictions = clip.predict_binary(images)
        y_true.extend(labels.numpy())
        y_pred_probs.extend(predictions[:, 1])

    y_pred_probs_arr = np.array(y_pred_probs)
    y_true_arr = np.array(y_true)
    fpr, tpr, thresholds = roc_curve(y_true_arr, y_pred_probs_arr)
    optimal_idx = int(np.argmax(tpr - fpr))
    optimal_threshold = float(thresholds[optimal_idx])
    ctx.best_threshold = optimal_threshold
    print(f"Optimal threshold: {optimal_threshold:.3f}")

    y_pred = (y_pred_probs_arr > optimal_threshold).astype(int)
    print(classification_report(y_true_arr, y_pred, target_names=ctx.class_names))

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_true_arr, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title("CLIP binary confusion matrix")
    finalize_plot(ctx, "12_clip_binary_confusion_matrix")

    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend()
    finalize_plot(ctx, "12_clip_binary_roc")

    plot_clip_predictions(
        ctx,
        val_ds_binary,
        ctx.class_names,
        clip.predict_binary,
        threshold=optimal_threshold,
        name="12_clip_binary_predictions",
    )

    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_probs_arr[y_true_arr == 0], alpha=0.5, label="Clean", bins=20, density=True)
    plt.hist(y_pred_probs_arr[y_true_arr == 1], alpha=0.5, label="Not Clean", bins=20, density=True)
    plt.axvline(optimal_threshold, color="r", linestyle="--", label=f"Threshold ({optimal_threshold:.3f})")
    plt.legend()
    finalize_plot(ctx, "12_clip_binary_probability_distribution")

    print(f"Accuracy: {accuracy_score(y_true_arr, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true_arr, y_pred):.3f}")
    print(f"Recall: {recall_score(y_true_arr, y_pred):.3f}")
    print(f"F1: {f1_score(y_true_arr, y_pred):.3f}")
    print(f"AUC-ROC: {roc_auc:.3f}")
