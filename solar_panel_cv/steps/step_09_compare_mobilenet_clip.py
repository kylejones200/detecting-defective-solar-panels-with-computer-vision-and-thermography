"""Compare MobileNetV2 and CLIP on the faulty solar panel dataset."""

from __future__ import annotations

import tensorflow as tf
from sklearn.metrics import classification_report

from solar_panel_cv.clip import evaluate_both_models
from solar_panel_cv.context import PipelineContext
from solar_panel_cv.datasets import load_image_datasets
from solar_panel_cv.models import create_mobilenet_model
from solar_panel_cv.plotting import plot_comparison_predictions, plot_training_history
from solar_panel_cv.training import early_stopping


def compare_mobilenet_clip(ctx: PipelineContext) -> None:
    training = ctx.config.training
    train_ds, val_ds, class_names = load_image_datasets(
        ctx.faulty_panel_dir(),
        training,
        cache=True,
    )
    ctx.class_names = class_names
    autotune = tf.data.AUTOTUNE
    train_ds_mobile = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds_mobile = val_ds.cache().prefetch(buffer_size=autotune)

    mobile_model, _ = create_mobilenet_model(
        len(class_names),
        training.img_height,
        training.img_width,
    )
    ctx.mobile_model = mobile_model
    mobile_model.compile(
        optimizer=tf.keras.optimizers.Adam(training.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    clip = ctx.ensure_clip()
    print("Training MobileNetV2 model...")
    history = mobile_model.fit(
        train_ds_mobile,
        validation_data=val_ds_mobile,
        epochs=training.epochs,
        callbacks=[early_stopping(training)],
    )

    y_true, y_pred_mobile, y_pred_clip = evaluate_both_models(
        val_ds, class_names, mobile_model, clip
    )
    print("\nMobileNetV2 classification report:")
    print(classification_report(y_true, y_pred_mobile, target_names=class_names))
    print("\nCLIP classification report:")
    print(classification_report(y_true, y_pred_clip, target_names=class_names))
    plot_training_history(ctx, history, name="09_mobilenet_vs_clip_training")
    plot_comparison_predictions(
        ctx, val_ds, class_names, mobile_model, clip, name="09_mobilenet_vs_clip"
    )
