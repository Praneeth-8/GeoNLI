"""
prepare_vrsbench_subset.py
--------------------------
Builds a small training-sized VRSBench subset for GeoNLI.

It keeps the original annotation files as raw references, samples a
captioning subset and a VQA subset, and extracts only the matching images
into the requested training image folder.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_VRSBENCH_DIR = DEFAULT_DATA_DIR / "vrsbench"
DEFAULT_RAW_ANNOTATIONS_DIR = DEFAULT_DATA_DIR / "raw_annotations"
DEFAULT_TRAIN_IMAGES_DIR = DEFAULT_DATA_DIR / "train_images"
DEFAULT_CAPTION_OUTPUT = DEFAULT_DATA_DIR / "train_caption_subset.json"
DEFAULT_VQA_OUTPUT = DEFAULT_DATA_DIR / "train_vqa_subset.json"


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data).__name__}")
    return data


def copy_raw_annotations(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in [
        "VRSBench_EVAL_Cap.json",
        "VRSBench_EVAL_vqa.json",
    ]:
        source_path = source_dir / filename
        target_path = target_dir / filename
        shutil.copy2(source_path, target_path)


def sample_entries(entries: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count > len(entries):
        raise ValueError(f"Requested {count} entries but only found {len(entries)} available")
    rng = random.Random(seed)
    sampled_indexes = sorted(rng.sample(range(len(entries)), count))
    return [entries[index] for index in sampled_indexes]


def extract_images(zip_path: Path, image_names: set[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = set()
    with zipfile.ZipFile(zip_path, "r") as archive:
        available = set(archive.namelist())
        missing = sorted(name for name in image_names if f"Images_val/{name}" not in available)
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} images in archive. Example: {missing[0]}")
        for name in sorted(image_names):
            archive.extract(f"Images_val/{name}", path=output_dir)
            extracted.add(name)
    if extracted != image_names:
        raise RuntimeError("Image extraction did not complete as expected")


def normalize_image_name(entry: dict[str, Any]) -> str:
    image_name = entry.get("image_id") or entry.get("image")
    if not isinstance(image_name, str) or not image_name:
        raise ValueError(f"Entry missing image identifier: {entry}")
    return image_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a small VRSBench training subset")
    parser.add_argument("--caption-count", type=int, default=1500)
    parser.add_argument("--vqa-count", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vrsbench-dir", type=Path, default=DEFAULT_VRSBENCH_DIR)
    parser.add_argument("--raw-annotations-dir", type=Path, default=DEFAULT_RAW_ANNOTATIONS_DIR)
    parser.add_argument("--train-images-dir", type=Path, default=DEFAULT_TRAIN_IMAGES_DIR)
    parser.add_argument("--caption-output", type=Path, default=DEFAULT_CAPTION_OUTPUT)
    parser.add_argument("--vqa-output", type=Path, default=DEFAULT_VQA_OUTPUT)
    args = parser.parse_args()

    vrsbench_dir = args.vrsbench_dir.resolve()
    raw_annotations_dir = args.raw_annotations_dir.resolve()
    train_images_dir = args.train_images_dir.resolve()
    caption_output = args.caption_output.resolve()
    vqa_output = args.vqa_output.resolve()

    caption_path = vrsbench_dir / "VRSBench_EVAL_Cap.json"
    vqa_path = vrsbench_dir / "VRSBench_EVAL_vqa.json"
    images_zip = vrsbench_dir / "Images_val.zip"

    caption_entries = load_json(caption_path)
    vqa_entries = load_json(vqa_path)

    caption_subset = sample_entries(caption_entries, args.caption_count, args.seed)
    vqa_subset = sample_entries(vqa_entries, args.vqa_count, args.seed + 1)

    copy_raw_annotations(vrsbench_dir, raw_annotations_dir)

    caption_output.parent.mkdir(parents=True, exist_ok=True)
    with caption_output.open("w", encoding="utf-8") as handle:
        json.dump(caption_subset, handle, indent=2, ensure_ascii=False)

    with vqa_output.open("w", encoding="utf-8") as handle:
        json.dump(vqa_subset, handle, indent=2, ensure_ascii=False)

    image_names = {normalize_image_name(entry) for entry in caption_subset}
    image_names.update(normalize_image_name(entry) for entry in vqa_subset)
    extract_images(images_zip, image_names, train_images_dir)

    caption_counts = Counter(normalize_image_name(entry) for entry in caption_subset)
    vqa_counts = Counter(normalize_image_name(entry) for entry in vqa_subset)
    shared_images = len(set(caption_counts) & set(vqa_counts))

    print("Subset prepared")
    print(f"Caption entries: {len(caption_subset)}")
    print(f"VQA entries: {len(vqa_subset)}")
    print(f"Unique images: {len(image_names)}")
    print(f"Images shared by both subsets: {shared_images}")
    print(f"Raw annotations copied to: {raw_annotations_dir}")
    print(f"Train images extracted to: {train_images_dir}")
    print(f"Caption subset saved to: {caption_output}")
    print(f"VQA subset saved to: {vqa_output}")
    print("Raw caption examples:")
    for entry in caption_subset[:3]:
        print(json.dumps(entry, ensure_ascii=False))


if __name__ == "__main__":
    main()
