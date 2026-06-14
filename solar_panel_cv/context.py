"""Mutable pipeline state passed between steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from solar_panel_cv.config import PipelineConfig


@dataclass
class PipelineContext:
    config: PipelineConfig
    root_dir: Path
    output_dir: Path

    dataset_root: Path | None = None
    faulty_panel_path: Path | None = None
    binary_dataset_path: Path | None = None

    train_ds: Any = None
    val_ds: Any = None
    train_ds_binary: Any = None
    val_ds_binary: Any = None

    class_names: list[str] = field(default_factory=list)
    original_class_names: list[str] = field(default_factory=list)

    model: Any = None
    base_model: Any = None
    mobile_model: Any = None
    clip_predictor: Any = None

    best_threshold: float | None = None
    best_temperature: float | None = None
    class_weights: np.ndarray | None = None

    @classmethod
    def from_config(cls, config: PipelineConfig, root_dir: Path | None = None) -> PipelineContext:
        root = (root_dir or Path.cwd()).resolve()
        output_dir = root / config.output.dir
        output_dir.mkdir(parents=True, exist_ok=True)
        ctx = cls(config=config, root_dir=root, output_dir=output_dir)
        ctx.binary_dataset_path = root / config.data.binary_dataset_dir
        return ctx

    def resolve_dataset_root(self) -> Path:
        if self.dataset_root is not None:
            return self.dataset_root
        if self.config.data.download_kaggle:
            import kagglehub

            path = Path(kagglehub.dataset_download(self.config.data.kaggle_dataset))
            self.dataset_root = path
            return path
        raise FileNotFoundError(
            "Dataset root is unset. Enable data.download_kaggle or run extract_archive first."
        )

    def faulty_panel_dir(self) -> Path:
        if self.faulty_panel_path is not None:
            return self.faulty_panel_path
        if self.config.data.archive_path:
            extracted = self.root_dir / self.config.data.extract_dir
            candidate = extracted / self.config.data.faulty_panel_subdir
            if candidate.is_dir():
                self.faulty_panel_path = candidate
                return candidate
        root = self.resolve_dataset_root()
        for candidate in (
            root / self.config.data.faulty_panel_subdir,
            root,
        ):
            if candidate.is_dir() and any(candidate.iterdir()):
                self.faulty_panel_path = candidate
                return candidate
        raise FileNotFoundError(
            f"Could not locate faulty panel images under {root}. "
            "Check data.faulty_panel_subdir or run extract_archive."
        )

    def binary_panel_dir(self) -> Path:
        if self.binary_dataset_path is None:
            self.binary_dataset_path = self.root_dir / self.config.data.binary_dataset_dir
        return self.binary_dataset_path

    def ensure_clip(self) -> Any:
        if self.clip_predictor is None:
            from solar_panel_cv.clip import ClipPredictor

            self.clip_predictor = ClipPredictor.from_config(self.config)
        return self.clip_predictor
