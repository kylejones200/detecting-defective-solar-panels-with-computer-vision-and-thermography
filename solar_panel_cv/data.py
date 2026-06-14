"""Dataset preparation utilities."""

import os
import shutil
from pathlib import Path

import tensorflow as tf


def setup_binary_dataset(source_path: Path | str, dest_path: Path | str) -> Path:
    source_path = Path(source_path)
    dest_path = Path(dest_path)
    for subdir in ("Clean", "Not_Clean"):
        (dest_path / subdir).mkdir(parents=True, exist_ok=True)

    for class_dir in sorted(source_path.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        dest_dir = dest_path / "Clean" if class_name == "Clean" else dest_path / "Not_Clean"
        for image_path in class_dir.iterdir():
            if not image_path.is_file():
                continue
            shutil.copy2(image_path, dest_dir / f"{class_name}_{image_path.name}")

    clean_count = len(os.listdir(dest_path / "Clean"))
    not_clean_count = len(os.listdir(dest_path / "Not_Clean"))
    print("Binary dataset created:")
    print(f"  Clean: {clean_count}")
    print(f"  Not Clean: {not_clean_count}")
    return dest_path


def to_binary_labels(images, labels, original_class_names: list[str]):
    clean_idx = original_class_names.index("Clean")
    binary_labels = tf.where(labels == clean_idx, 0, 1)
    return images, binary_labels
