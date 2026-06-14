"""Pipeline configuration loaded from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    kaggle_dataset: str = "pythonafroz/solar-panel-images"
    download_kaggle: bool = True
    archive_path: str | None = None
    extract_dir: str = "data/raw"
    faulty_panel_subdir: str = "Faulty_solar_panel"
    binary_dataset_dir: str = "data/binary_solar_panels"


class TrainingConfig(BaseModel):
    img_height: int = 244
    img_width: int = 244
    clip_img_size: int = 224
    batch_size: int = 32
    validation_split: float = 0.2
    seed: int = 42
    epochs: int = 20
    fine_tune_epochs: int = 15
    learning_rate: float = 0.001
    fine_tune_learning_rate: float = 0.0001
    early_stopping_patience: int = 5


class ModelConfig(BaseModel):
    clip_model_id: str = "openai/clip-vit-base-patch32"


class ClipTextDescriptions(BaseModel):
    clean: list[str] = Field(default_factory=list)
    not_clean: list[str] = Field(default_factory=list)


class ClipConfig(BaseModel):
    default_temperature: float = 100.0
    confidence_temperature: float = 50.0
    temperature_candidates: list[float] = Field(default_factory=lambda: [50.0, 100.0, 150.0])
    threshold_search_start: float = 0.3
    threshold_search_end: float = 0.7
    threshold_search_step: float = 0.05
    paired_prompts: list[list[str]] = Field(default_factory=list)
    text_descriptions: ClipTextDescriptions = Field(default_factory=ClipTextDescriptions)


class OutputConfig(BaseModel):
    dir: str = "outputs"
    show_plots: bool = False
    save_plots: bool = True


class PipelineStepsConfig(BaseModel):
    steps: list[str] = Field(default_factory=list)


class PipelineConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    clip: ClipConfig = Field(default_factory=ClipConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    pipeline: PipelineStepsConfig = Field(default_factory=PipelineStepsConfig)


def load_config(path: Path | str) -> PipelineConfig:
    config_path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return PipelineConfig.model_validate(raw)


def default_config_path() -> Path:
    return Path("config.yaml")
