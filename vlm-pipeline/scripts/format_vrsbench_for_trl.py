"""
format_vrsbench_for_trl.py
--------------------------
Converts the split VRSBench caption and VQA files into the conversational
vision format recommended by TRL and Transformers for VLM fine-tuning.

The output keeps a single image per example and stores the image path in the
`image` field so the JSON can be loaded later and materialized as PIL images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_IMAGES_DIR = DEFAULT_DATA_DIR / "train_images"


def load_entries(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data).__name__}")
    return data


def build_image_path(image_id: str, images_dir: Path) -> str:
    matches = sorted(images_dir.rglob(image_id))
    if not matches:
        raise FileNotFoundError(f"Could not find {image_id} under {images_dir}")
    return str(matches[0].resolve())


def to_formatted_example(entry: dict[str, Any], images_dir: Path) -> dict[str, Any]:
    image_id = entry.get("image_id")
    if not isinstance(image_id, str) or not image_id:
        raise ValueError(f"Entry missing image_id: {entry}")

    question = entry.get("question")
    answer = entry.get("ground_truth")
    if not isinstance(question, str) or not isinstance(answer, str):
        raise ValueError(f"Entry missing text fields: {entry}")

    image_path = build_image_path(image_id, images_dir)
    return {
        "image": image_path,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ],
        "source": entry.get("type", "unknown"),
        "image_id": image_id,
    }


def write_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Format VRSBench splits for TRL fine-tuning")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    images_dir = args.images_dir.resolve()

    train_caption = load_entries(data_dir / "train_annotations.json")
    val_caption = load_entries(data_dir / "val_annotations.json")
    train_vqa = load_entries(data_dir / "train_vqa_annotations.json")
    val_vqa = load_entries(data_dir / "val_vqa_annotations.json")

    train_formatted = [to_formatted_example(entry, images_dir) for entry in train_caption + train_vqa]
    val_formatted = [to_formatted_example(entry, images_dir) for entry in val_caption + val_vqa]

    write_json(data_dir / "train_formatted.json", train_formatted)
    write_json(data_dir / "val_formatted.json", val_formatted)

    print("Formatting complete")
    print(f"train_formatted.json: {len(train_formatted)} examples")
    print(f"val_formatted.json: {len(val_formatted)} examples")

    print("First 3 train examples:")
    for example in train_formatted[:3]:
        print(json.dumps(example, ensure_ascii=False))

    print("First 3 val examples:")
    for example in val_formatted[:3]:
        print(json.dumps(example, ensure_ascii=False))


if __name__ == "__main__":
    main()
t