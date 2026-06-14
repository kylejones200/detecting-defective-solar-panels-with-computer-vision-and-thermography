"""Prepare a binary Clean / Not_Clean dataset from multiclass source images."""

from __future__ import annotations

from solar_panel_cv.context import PipelineContext
from solar_panel_cv.data import setup_binary_dataset


def prepare_binary_dataset(ctx: PipelineContext) -> None:
    source = ctx.faulty_panel_dir()
    dest = ctx.binary_panel_dir()
    if (dest / "Clean").exists() and (dest / "Not_Clean").exists():
        clean = len(list((dest / "Clean").iterdir()))
        not_clean = len(list((dest / "Not_Clean").iterdir()))
        if clean > 0 and not_clean > 0:
            print(f"Binary dataset already exists at {dest}; skipping.")
            return

    setup_binary_dataset(source, dest)
    ctx.binary_dataset_path = dest
