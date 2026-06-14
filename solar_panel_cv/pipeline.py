"""Pipeline step registry and runner."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from solar_panel_cv.config import PipelineConfig, load_config
from solar_panel_cv.context import PipelineContext
from solar_panel_cv.steps import STEP_FUNCTIONS

StepFn = Callable[[PipelineContext], None]

DEFAULT_STEPS: list[str] = [
    "explore_kaggle_dataset",
    "build_vgg16_and_explore",
    "train_multiclass_checkpoint",
    "fine_tune_multiclass",
    "clip_zero_shot_kaggle",
    "train_mobilenet_faulty_panel",
    "train_multiclass_faulty_panel",
    "extract_archive",
    "prepare_binary_dataset",
    "compare_mobilenet_clip",
    "train_binary_classifier",
    "clip_zero_shot_faulty_panel",
    "clip_binary_evaluation",
    "clip_binary_dataset_evaluation",
    "clip_temperature_tuning",
    "clip_weighted_confidence",
]


def list_steps() -> list[str]:
    return list(STEP_FUNCTIONS.keys())


def resolve_steps(config: PipelineConfig, selected: list[str] | None = None) -> list[str]:
    steps = selected or config.pipeline.steps or DEFAULT_STEPS
    unknown = [name for name in steps if name not in STEP_FUNCTIONS]
    if unknown:
        raise ValueError(f"Unknown pipeline steps: {', '.join(unknown)}")
    return steps


def run_pipeline(
    config_path: Path | str = "config.yaml",
    *,
    steps: list[str] | None = None,
    root_dir: Path | None = None,
) -> PipelineContext:
    config = load_config(config_path)
    ctx = PipelineContext.from_config(config, root_dir=root_dir)
    step_names = resolve_steps(config, steps)
    for index, name in enumerate(step_names, start=1):
        print(f"\n=== Step {index}/{len(step_names)}: {name} ===")
        STEP_FUNCTIONS[name](ctx)
    return ctx


def main() -> None:
    from solar_panel_cv.cli import main as cli_main

    cli_main()


__all__ = ["DEFAULT_STEPS", "list_steps", "main", "run_pipeline"]
