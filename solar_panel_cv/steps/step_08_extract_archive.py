"""Extract the faulty solar panel dataset archive."""

from __future__ import annotations

from zipfile import ZipFile

from solar_panel_cv.context import PipelineContext


def extract_archive(ctx: PipelineContext) -> None:
    archive = ctx.config.data.archive_path
    if not archive:
        print("No archive_path configured; skipping extract_archive.")
        return

    archive_path = ctx.root_dir / archive
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    extract_dir = ctx.root_dir / ctx.config.data.extract_dir
    extract_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path, "r") as zf:
        zf.extractall(extract_dir)

    ctx.faulty_panel_path = extract_dir / ctx.config.data.faulty_panel_subdir
    ctx.dataset_root = extract_dir
    print(f"Extracted archive to {extract_dir}")
