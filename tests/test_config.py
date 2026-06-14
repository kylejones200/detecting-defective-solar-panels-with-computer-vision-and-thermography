"""Tests for pipeline configuration."""

from pathlib import Path

from solar_panel_cv.config import load_config
from solar_panel_cv.pipeline import DEFAULT_STEPS, list_steps, resolve_steps


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_config_yaml_loads() -> None:
    config = load_config(REPO_ROOT / "config.yaml")
    assert config.data.kaggle_dataset
    assert config.training.batch_size > 0
    assert len(config.pipeline.steps) >= 1


def test_default_steps_are_registered() -> None:
    registered = set(list_steps())
    for step in DEFAULT_STEPS:
        assert step in registered


def test_resolve_steps_uses_config_when_present() -> None:
    config = load_config(REPO_ROOT / "config.yaml")
    steps = resolve_steps(config)
    assert steps == config.pipeline.steps


def test_unknown_step_raises() -> None:
    config = load_config(REPO_ROOT / "config.yaml")
    try:
        resolve_steps(config, ["not_a_real_step"])
        raised = False
    except ValueError:
        raised = True
    assert raised
