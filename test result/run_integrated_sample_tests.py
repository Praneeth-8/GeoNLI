import argparse
import csv
import json
import os
import sys
import time
from types import SimpleNamespace

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VLM_ROOT = os.path.join(PROJECT_ROOT, "vlm-pipeline")
IMAGE_ROOT = os.path.join(VLM_ROOT, "data", "train_images", "Images_val")
OUTPUT_DIR = os.path.dirname(__file__)
CAPTIONED_IMAGE_DIR = os.path.join(OUTPUT_DIR, "captioned_images")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, VLM_ROOT)

import cv2

from detector import YOLODetector
from tiler import generate_tiles, merge_detections, remap_to_global
from test_inference import (
    DEFAULT_MODEL,
    DEFAULT_VQA_PROMPT,
    build_runtime_config,
    load_model_and_processor,
    run_inference,
)

CAPTION_PROMPT = (
    "Write only the visual caption for this remote-sensing image. "
    "Do not start with phrases like 'the image you provided'. "
    "Describe the visible objects, land use, and spatial layout in 2-3 factual sentences."
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def choose_samples(limit):
    image_names = set(os.listdir(IMAGE_ROOT))
    captions = load_json(os.path.join(VLM_ROOT, "data", "train_caption_subset.json"))
    vqas = load_json(os.path.join(VLM_ROOT, "data", "train_vqa_subset.json"))

    caption_by_image = {}
    for item in captions:
        if item["image_id"] in image_names:
            caption_by_image.setdefault(item["image_id"], item)

    vqa_by_image = {}
    for item in vqas:
        if item["image_id"] in image_names:
            vqa_by_image.setdefault(item["image_id"], item)

    shared = sorted(set(caption_by_image) & set(vqa_by_image))
    selected = shared[:limit]
    return [(name, caption_by_image[name], vqa_by_image[name]) for name in selected]


def run_grounding(detector, image_path, tile_size, overlap):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)

    tiles = generate_tiles(image, tile_size, tile_size, overlap)
    all_detections = []
    for tile in tiles:
        detections = detector.detect(tile)
        all_detections.extend(remap_to_global(detections, tile))

    final_detections = merge_detections(all_detections)
    return len(tiles), len(all_detections), final_detections


def serialize_detections(detections):
    serialized = []
    for det in detections:
        serialized.append(
            {
                "cx": float(det["cx"]),
                "cy": float(det["cy"]),
                "w": float(det["w"]),
                "h": float(det["h"]),
                "angle_deg": float(det["angle_deg"]),
                "score": float(det["score"]),
                "class_id": int(det["class_id"]),
                "class_name": str(det["class_name"]),
                "touches_edge": bool(det.get("touches_edge", False)),
            }
        )
    return serialized


def draw_grounding_boxes(image, detections):
    for det in detections:
        rect = (
            (det["cx"], det["cy"]),
            (det["w"], det["h"]),
            det["angle_deg"],
        )
        box = cv2.boxPoints(rect).astype(int)
        label = f"{det['class_name']} {det['score']:.2f}"

        cv2.polylines(image, [box], True, (0, 255, 0), 2)
        label_x = max(0, min(int(det["cx"]), image.shape[1] - 1))
        label_y = max(16, min(int(det["cy"]), image.shape[0] - 1))
        cv2.putText(
            image,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )


def wrap_text(text, max_chars):
    words = str(text).split()
    lines = []
    current = []
    current_len = 0

    for word in words:
        extra = 1 if current else 0
        if current and current_len + extra + len(word) > max_chars:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += extra + len(word)

    if current:
        lines.append(" ".join(current))
    return lines


def write_captioned_image(row, image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)

    draw_grounding_boxes(image, row.get("detections", []))

    lines = [f"Image: {row['image_id']}"]
    if row.get("caption_output"):
        lines.append("Caption:")
        lines.extend(wrap_text(row["caption_output"], 80))
    if row.get("vqa_question"):
        lines.append(f"Q: {row['vqa_question']}")
    if row.get("vqa_output"):
        lines.append(f"A: {row['vqa_output']}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 22
    padding = 12
    panel_height = padding * 2 + line_height * len(lines)

    canvas = cv2.copyMakeBorder(
        image,
        0,
        panel_height,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    y = image.shape[0] + padding + 14
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (padding, y),
            font,
            font_scale,
            (20, 20, 20),
            thickness,
            cv2.LINE_AA,
        )
        y += line_height

    os.makedirs(CAPTIONED_IMAGE_DIR, exist_ok=True)
    output_name = f"{row['index']:03d}_{os.path.splitext(row['image_id'])[0]}_captioned.png"
    output_path = os.path.join(CAPTIONED_IMAGE_DIR, output_name)
    cv2.imwrite(output_path, canvas)
    return os.path.relpath(output_path, OUTPUT_DIR)


def build_model_args(args):
    return SimpleNamespace(
        model=args.model,
        cache_dir=args.cache_dir,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        no_quantization=args.no_quantization,
    )


def main():
    parser = argparse.ArgumentParser(description="Run integrated GeoNLI sample tests.")
    parser.add_argument("--limit", type=int, default=50, help="Number of real images to test.")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--detector-model", default=os.path.join(PROJECT_ROOT, "yolo11n-obb.pt"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", default=os.path.join(VLM_ROOT, "models"))
    parser.add_argument("--caption-tokens", type=int, default=64)
    parser.add_argument("--vqa-tokens", type=int, default=16)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--no-quantization", action="store_true")
    parser.add_argument("--skip-vlm", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CAPTIONED_IMAGE_DIR, exist_ok=True)
    samples = choose_samples(args.limit)
    started = time.time()

    detector = YOLODetector(args.detector_model)
    model = None
    processor = None
    if not args.skip_vlm:
        config = build_runtime_config(build_model_args(args))
        model, processor = load_model_and_processor(config)

    jsonl_path = os.path.join(OUTPUT_DIR, "integrated_50_image_results.jsonl")
    csv_path = os.path.join(OUTPUT_DIR, "integrated_50_image_results.csv")
    summary_path = os.path.join(OUTPUT_DIR, "INTEGRATED_TEST_SUMMARY.md")

    rows = []
    with open(jsonl_path, "w", encoding="utf-8") as jsonl:
        for index, (image_name, caption_record, vqa_record) in enumerate(samples, start=1):
            image_path = os.path.join(IMAGE_ROOT, image_name)
            row = {
                "index": index,
                "image_id": image_name,
                "image_path": os.path.relpath(image_path, PROJECT_ROOT),
                "caption_expected": caption_record["ground_truth"],
                "vqa_question": vqa_record["question"],
                "vqa_expected": vqa_record["ground_truth"],
                "caption_status": "skipped",
                "vqa_status": "skipped",
                "grounding_status": "not_started",
            }

            image_started = time.time()
            try:
                tile_count, raw_count, final_detections = run_grounding(
                    detector,
                    image_path,
                    args.tile_size,
                    args.overlap,
                )
                detections = serialize_detections(final_detections)
                row.update(
                    {
                        "grounding_status": "pass",
                        "tile_count": tile_count,
                        "raw_detection_count": raw_count,
                        "final_detection_count": len(detections),
                        "detection_classes": ", ".join(
                            sorted({det["class_name"] for det in detections})
                        ),
                        "detections": detections,
                    }
                )
            except Exception as exc:
                row["grounding_status"] = "fail"
                row["grounding_error"] = repr(exc)

            if model is not None and processor is not None:
                try:
                    row["caption_output"] = run_inference(
                        model,
                        processor,
                        image_path,
                        CAPTION_PROMPT,
                        max_new_tokens=args.caption_tokens,
                    )
                    row["caption_status"] = "pass"
                except Exception as exc:
                    row["caption_status"] = "fail"
                    row["caption_error"] = repr(exc)

                try:
                    row["vqa_output"] = run_inference(
                        model,
                        processor,
                        image_path,
                        DEFAULT_VQA_PROMPT.format(question=vqa_record["question"]),
                        max_new_tokens=args.vqa_tokens,
                    )
                    row["vqa_status"] = "pass"
                except Exception as exc:
                    row["vqa_status"] = "fail"
                    row["vqa_error"] = repr(exc)

            try:
                row["captioned_image"] = write_captioned_image(row, image_path)
            except Exception as exc:
                row["captioned_image_error"] = repr(exc)

            row["elapsed_seconds"] = round(time.time() - image_started, 2)
            rows.append(row)
            jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
            jsonl.flush()
            print(
                f"[{index}/{len(samples)}] {image_name}: "
                f"tiler+grounding={row['grounding_status']} "
                f"caption={row['caption_status']} vqa={row['vqa_status']} "
                f"elapsed={row['elapsed_seconds']}s",
                flush=True,
            )

    fieldnames = sorted({key for row in rows for key in row})
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    passed_grounding = sum(row["grounding_status"] == "pass" for row in rows)
    passed_caption = sum(row["caption_status"] == "pass" for row in rows)
    passed_vqa = sum(row["vqa_status"] == "pass" for row in rows)
    total_elapsed = round(time.time() - started, 2)

    with open(summary_path, "w", encoding="utf-8") as summary:
        summary.write("# Integrated GeoNLI Test Summary\n\n")
        summary.write(f"- Images tested: {len(rows)}\n")
        summary.write(f"- Tiler + grounding passed: {passed_grounding}/{len(rows)}\n")
        summary.write(f"- Captioning passed: {passed_caption}/{len(rows)}\n")
        summary.write(f"- VQA/reasoning passed: {passed_vqa}/{len(rows)}\n")
        summary.write(f"- Total elapsed seconds: {total_elapsed}\n")
        summary.write(f"- CSV results: `{os.path.basename(csv_path)}`\n")
        summary.write(f"- JSONL results: `{os.path.basename(jsonl_path)}`\n")
        summary.write(f"- Captioned image folder: `{os.path.basename(CAPTIONED_IMAGE_DIR)}`\n\n")
        summary.write("## Sample Outputs\n\n")
        for row in rows[:10]:
            summary.write(f"### {row['index']}. {row['image_id']}\n\n")
            if row.get("captioned_image"):
                summary.write(f"![Captioned image]({row['captioned_image']})\n\n")
            summary.write(f"- Grounding: {row['grounding_status']}\n")
            summary.write(f"- Caption output: {row.get('caption_output', '')}\n")
            summary.write(f"- VQA question: {row['vqa_question']}\n")
            summary.write(f"- VQA output: {row.get('vqa_output', '')}\n")
            summary.write(f"- VQA expected: {row['vqa_expected']}\n\n")

    print(f"Wrote {summary_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {jsonl_path}")


if __name__ == "__main__":
    main()
