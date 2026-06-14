"""Fine-tune the multiclass model."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from solar_panel_cv.context import PipelineContext
from solar_panel_cv.plotting import finalize_plot
from solar_panel_cv.training import early_stopping


def fine_tune_multiclass(ctx: PipelineContext) -> None:
    if ctx.model is None or ctx.base_model is None:
        raise RuntimeError("Run build_vgg16_and_explore before fine_tune_multiclass.")

    training = ctx.config.training
    ctx.base_model.trainable = True
    for layer in ctx.base_model.layers[:14]:
        layer.trainable = False

    ctx.model.summary()
    ctx.model.compile(
        optimizer=tf.keras.optimizers.Adam(training.fine_tune_learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = ctx.model.fit(
        ctx.train_ds,
        validation_data=ctx.val_ds,
        epochs=training.fine_tune_epochs,
        callbacks=[early_stopping(training)],
    )

    epochs = range(len(history.history["accuracy"]))
    plt.figure()
    plt.plot(epochs, history.history["accuracy"], label="Train accuracy")
    plt.plot(epochs, history.history["val_accuracy"], label="Val accuracy")
    plt.legend()
    plt.title("Fine-tune accuracy")
    finalize_plot(ctx, "05_fine_tune_accuracy")

    loss, accuracy = ctx.model.evaluate(ctx.val_ds, verbose=0)
    print(f"Validation loss: {loss:.3f}, accuracy: {accuracy:.3f}")

    class_names = ctx.class_names
    plt.figure(figsize=(20, 20))
    for images, labels in ctx.val_ds.take(1):
        for i in range(min(16, len(images))):
            plt.subplot(4, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            predictions = ctx.model.predict(tf.expand_dims(images[i], 0), verbose=0)
            score = tf.nn.softmax(predictions[0])
            pred_idx = int(np.argmax(score))
            true_idx = int(labels[i].numpy())
            color = "green" if pred_idx == true_idx else "red"
            plt.title(f"Actual: {class_names[true_idx]}")
            plt.ylabel(
                f"Predicted: {class_names[pred_idx]}",
                fontdict={"color": color},
            )
            plt.axis("off")
    plt.tight_layout()
    finalize_plot(ctx, "05_fine_tune_predictions")
