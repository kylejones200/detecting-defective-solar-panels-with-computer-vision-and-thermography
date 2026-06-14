"""Pipeline step registry."""

from __future__ import annotations

from collections.abc import Callable

from solar_panel_cv.context import PipelineContext
from solar_panel_cv.steps.step_01_explore_kaggle_dataset import explore_kaggle_dataset
from solar_panel_cv.steps.step_02_build_vgg16_and_explore import build_vgg16_and_explore
from solar_panel_cv.steps.step_03_clip_zero_shot_kaggle import clip_zero_shot_kaggle
from solar_panel_cv.steps.step_04_train_multiclass_checkpoint import train_multiclass_checkpoint
from solar_panel_cv.steps.step_05_fine_tune_multiclass import fine_tune_multiclass
from solar_panel_cv.steps.step_06_train_mobilenet_faulty_panel import train_mobilenet_faulty_panel
from solar_panel_cv.steps.step_07_train_multiclass_faulty_panel import train_multiclass_faulty_panel
from solar_panel_cv.steps.step_08_extract_archive import extract_archive
from solar_panel_cv.steps.step_08b_prepare_binary_dataset import prepare_binary_dataset
from solar_panel_cv.steps.step_09_compare_mobilenet_clip import compare_mobilenet_clip
from solar_panel_cv.steps.step_10_train_binary_classifier import train_binary_classifier
from solar_panel_cv.steps.step_11_clip_zero_shot_faulty_panel import clip_zero_shot_faulty_panel
from solar_panel_cv.steps.step_12_clip_binary_evaluation import clip_binary_evaluation
from solar_panel_cv.steps.step_13_clip_binary_dataset_evaluation import clip_binary_dataset_evaluation
from solar_panel_cv.steps.step_14_clip_temperature_tuning import clip_temperature_tuning
from solar_panel_cv.steps.step_15_clip_weighted_confidence import clip_weighted_confidence

StepFn = Callable[[PipelineContext], None]

STEP_FUNCTIONS: dict[str, StepFn] = {
    "explore_kaggle_dataset": explore_kaggle_dataset,
    "build_vgg16_and_explore": build_vgg16_and_explore,
    "clip_zero_shot_kaggle": clip_zero_shot_kaggle,
    "train_multiclass_checkpoint": train_multiclass_checkpoint,
    "fine_tune_multiclass": fine_tune_multiclass,
    "train_mobilenet_faulty_panel": train_mobilenet_faulty_panel,
    "train_multiclass_faulty_panel": train_multiclass_faulty_panel,
    "extract_archive": extract_archive,
    "prepare_binary_dataset": prepare_binary_dataset,
    "compare_mobilenet_clip": compare_mobilenet_clip,
    "train_binary_classifier": train_binary_classifier,
    "clip_zero_shot_faulty_panel": clip_zero_shot_faulty_panel,
    "clip_binary_evaluation": clip_binary_evaluation,
    "clip_binary_dataset_evaluation": clip_binary_dataset_evaluation,
    "clip_temperature_tuning": clip_temperature_tuning,
    "clip_weighted_confidence": clip_weighted_confidence,
}

__all__ = ["STEP_FUNCTIONS"]
