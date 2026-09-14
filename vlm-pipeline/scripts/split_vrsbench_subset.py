"""
split_vrsbench_subset.py
------------------------
Splits the sampled caption and VQA subsets into train/validation files
without leaking any image_id across the splits within the same subset.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_entries(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data).__name__}")
    return data


def split_by_image_id(entries: list[dict[str, Any]], train_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        image_id = entry.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError(f"Entry missing image_id: {entry}")
        buckets[image_id].append(entry)

    image_ids = sorted(buckets)
    rng = random.Random(seed)
    rng.shuffle(image_ids)

    train_target = int(round(len(image_ids) * train_ratio))
    train_ids = set(image_ids[:train_target])
    val_ids = set(image_ids[train_target:])

    train_entries: list[dict[str, Any]] = []
    val_entries: list[dict[str, Any]] = []

    for image_id in image_ids:
        group = buckets[image_id]
        if image_id in train_ids:
            train_entries.extend(group)
        else:
            val_entries.extend(group)

    return train_entries, val_entries


def write_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def unique_image_count(entries: list[dict[str, Any]]) -> int:
    return len({entry["image_id"] for entry in entries})


def main() -> None:
    parser = argparse.ArgumentParser(description="Split caption and VQA VRSBench subsets")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    caption_subset_path = data_dir / "train_caption_subset.json"
    vqa_subset_path = data_dir / "train_vqa_subset.json"

    caption_entries = load_entries(caption_subset_path)
    vqa_entries = load_entries(vqa_subset_path)

    caption_train, caption_val = split_by_image_id(caption_entries, args.train_ratio, args.seed)
    vqa_train, vqa_val = split_by_image_id(vqa_entries, args.train_ratio, args.seed + 1)

    write_json(data_dir / "train_annotations.json", caption_train)
    write_json(data_dir / "val_annotations.json", caption_val)
    write_json(data_dir / "train_vqa_annotations.json", vqa_train)
    write_json(data_dir / "val_vqa_annotations.json", vqa_val)

    print("Split complete")
    print(f"Caption train: {len(caption_train)} entries across {unique_image_count(caption_train)} image_ids")
    print(f"Caption val:   {len(caption_val)} entries across {unique_image_count(caption_val)} image_ids")
    print(f"VQA train:     {len(vqa_train)} entries across {unique_image_count(vqa_train)} image_ids")
    print(f"VQA val:       {len(vqa_val)} entries across {unique_image_count(vqa_val)} image_ids")

    caption_overlap = set(entry["image_id"] for entry in caption_train) & set(entry["image_id"] for entry in caption_val)
    vqa_overlap = set(entry["image_id"] for entry in vqa_train) & set(entry["image_id"] for entry in vqa_val)
    print(f"Caption split overlap: {len(caption_overlap)}")
    print(f"VQA split overlap: {len(vqa_overlap)}")


if __name__ == "__main__":
    main()
