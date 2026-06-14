"""Explore the Kaggle solar panel image dataset."""

from __future__ import annotations

import warnings

from solar_panel_cv.context import PipelineContext
from solar_panel_cv.datasets import load_image_datasets
from solar_panel_cv.plotting import plot_sample_grid


def explore_kaggle_dataset(ctx: PipelineContext) -> None:
    warnings.filterwarnings("ignore")
    train_ds, val_ds, class_names = load_image_datasets(
        ctx.faulty_panel_dir(),
        ctx.config.training,
    )
    ctx.train_ds = train_ds
    ctx.val_ds = val_ds
    ctx.class_names = class_names
    print("Classes:", class_names)
    plot_sample_grid(ctx, train_ds, class_names, name="01_explore_kaggle_dataset")
