"""Run CLIP zero-shot classification on the Kaggle dataset."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from solar_panel_cv.context import PipelineContext
from solar_panel_cv.datasets import load_image_datasets
from solar_panel_cv.plotting import finalize_plot


def clip_zero_shot_kaggle(ctx: PipelineContext) -> None:
    training = ctx.config.training
    train_ds, val_ds, class_names = load_image_datasets(
        ctx.resolve_dataset_root(),
        training,
        img_height=training.clip_img_size,
        img_width=training.clip_img_size,
    )
    ctx.train_ds = train_ds
    ctx.val_ds = val_ds
    ctx.class_names = class_names

    clip = ctx.ensure_clip()
    correct = 0
    total = 0
    for images, labels in val_ds:
        predictions = clip.predict_multiclass(images, class_names)
        predicted_classes = np.argmax(predictions, axis=1)
        correct += int((predicted_classes == labels.numpy()).sum())
        total += len(labels)

    print(f"Validation accuracy: {correct / total:.2f}")
    plt.figure(figsize=(20, 20))
    for images, labels in val_ds.take(1):
        predictions = clip.predict_multiclass(images, class_names)
        predicted_classes = np.argmax(predictions, axis=1)
        for i in range(min(16, len(images))):
            plt.subplot(4, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            color = "green" if labels[i] == predicted_classes[i] else "red"
            plt.title(f"Actual: {class_names[int(labels[i])]}")
            plt.ylabel(
                f"Predicted: {class_names[int(predicted_classes[i])]}",
                fontdict={"color": color},
            )
            plt.axis("off")
    plt.tight_layout()
    finalize_plot(ctx, "03_clip_zero_shot_kaggle")
