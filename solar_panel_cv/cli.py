"""Command-line interface for the solar panel CV pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from solar_panel_cv.config import default_config_path, load_config
from solar_panel_cv.pipeline import DEFAULT_STEPS, list_steps, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the solar panel defect detection pipeline.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        help="Pipeline steps to run (default: all steps from config or built-in list)",
    )
    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="Print available step names and exit",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print resolved config and exit",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path.cwd(),
        help="Project root for resolving relative paths",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_steps:
        for name in list_steps():
            print(name)
        return

    if args.print_config:
        config = load_config(args.config)
        print(config.model_dump_json(indent=2))
        return

    run_pipeline(
        args.config,
        steps=args.steps,
        root_dir=args.root_dir.resolve(),
    )


if __name__ == "__main__":
    main()
