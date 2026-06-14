"""Build a VGG16 baseline and explore the downloaded dataset."""

from __future__ import annotations

import warnings

from tensorflow.keras.utils import plot_model

from solar_panel_cv.context import PipelineContext
from solar_panel_cv.datasets import load_image_datasets
from solar_panel_cv.models import create_vgg16_model
from solar_panel_cv.plotting import plot_sample_grid


def build_vgg16_and_explore(ctx: PipelineContext) -> None:
    warnings.filterwarnings("ignore")
    dataset_root = ctx.resolve_dataset_root()
    print("Dataset root:", dataset_root)

    training = ctx.config.training
    train_ds, val_ds, class_names = load_image_datasets(dataset_root, training)
    ctx.train_ds = train_ds
    ctx.val_ds = val_ds
    ctx.class_names = class_names
    print("Classes:", class_names)
    plot_sample_grid(ctx, train_ds, class_names, name="02_build_vgg16_samples")

    model, base_model = create_vgg16_model(
        len(class_names),
        training.img_height,
        training.img_width,
    )
    ctx.model = model
    ctx.base_model = base_model
    model.summary()
    plot_path = ctx.output_dir / "vgg16_architecture.png"
    plot_model(model, to_file=str(plot_path), show_shapes=True, show_layer_names=True)
    print(f"Saved model diagram: {plot_path}")
