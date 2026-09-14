import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VLM_ROOT = PROJECT_ROOT / "vlm-pipeline"
OUTPUT_DIR = Path(__file__).resolve().parent
CAPTIONED_IMAGE_DIR = OUTPUT_DIR / "captioned_images"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(VLM_ROOT) not in sys.path:
    sys.path.insert(0, str(VLM_ROOT))

import cv2
import torch

from detector import YOLODetector
from grounding import Grounder
from tiler import generate_tiles, merge_detections, remap_to_global
from test_inference import (
    DEFAULT_MODEL,
    DEFAULT_CAPTION_PROMPT,
    DEFAULT_VQA_PROMPT,
    build_runtime_config,
    load_model_and_processor,
    run_inference,
)


def wrap_text(text, max_chars=75):
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


def draw_annotations(image, yolo_detections, ground_detections):
    annotated = image.copy()

    # Draw YOLO detections (Green OBB)
    for det in yolo_detections:
        rect = ((det["cx"], det["cy"]), (det["w"], det["h"]), det["angle_deg"])
        box = cv2.boxPoints(rect).astype(int)
        cv2.polylines(annotated, [box], True, (0, 230, 115), 2, cv2.LINE_AA)
        label = f"{det['class_name']} {det['score']:.2f}"
        cv2.putText(
            annotated,
            label,
            (max(2, int(det["cx"] - det["w"] / 4)), max(16, int(det["cy"] - det["h"] / 4))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 230, 115),
            1,
            cv2.LINE_AA,
        )

    # Draw Grounding DINO detections (Red Bounding Box)
    for det in ground_detections:
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (50, 80, 255), 2, cv2.LINE_AA)
        label = f"G:{det['label']} {det['score']:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 4, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (50, 80, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated


def write_captioned_image(row, annotated_image):
    lines = [
        f"Image: {row['image_id']} | Res: {row.get('image_size', 'N/A')}",
        f"YOLO Objects: {row.get('final_detection_count', 0)} ({row.get('detection_classes', 'None')})",
    ]
    if row.get("query"):
        lines.append(f"Grounding Query: '{row['query']}' -> Found: {row.get('grounding_count', 0)}")
    if row.get("caption_output"):
        lines.append("Generated Caption:")
        lines.extend(wrap_text(row["caption_output"], 85))
    if row.get("vqa_question"):
        lines.append(f"VQA Question: {row['vqa_question']}")
        lines.append(f"VQA Answer:   {row.get('vqa_output', 'N/A')}")
    lines.append(
        f"Latency: YOLO={row.get('yolo_ms', 0)}ms, Grounding={row.get('grounding_ms', 0)}ms, VLM={row.get('vlm_ms', 0)}ms | Total={row.get('total_ms', 0)}ms"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    line_height = 20
    padding = 12
    panel_height = padding * 2 + line_height * len(lines)

    canvas = cv2.copyMakeBorder(
        annotated_image,
        0,
        panel_height,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(245, 245, 245),
    )

    y = annotated_image.shape[0] + padding + 12
    for line in lines:
        color = (20, 20, 20)
        if line.startswith("VQA Answer:"):
            color = (180, 40, 20)
        elif line.startswith("YOLO Objects:") or line.startswith("Grounding Query:"):
            color = (20, 100, 20)
        elif line.startswith("Latency:"):
            color = (80, 80, 80)

        cv2.putText(canvas, line, (padding, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += line_height

    CAPTIONED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    out_filename = f"{row['index']:03d}_{Path(row['image_id']).stem}_annotated.png"
    out_path = CAPTIONED_IMAGE_DIR / out_filename
    cv2.imwrite(str(out_path), canvas)
    return out_path.relative_to(OUTPUT_DIR).as_posix()


def get_test_samples():
    """Returns curated showcase satellite images + real dataset validation images."""
    samples = []

    # 1. Curated benchmark images
    sample_dir = VLM_ROOT / "data" / "test_samples"
    if (sample_dir / "airport.jpg").exists():
        samples.append({
            "image_id": "airport.jpg",
            "path": str(sample_dir / "airport.jpg"),
            "category": "Aviation & Infrastructure",
            "query": "airplane, jet bridge, runway",
            "vqa_question": "What type of facility is shown and how many large aircraft are visible?",
            "vqa_expected": "Commercial Airport / multiple aircraft on apron",
        })
    if (sample_dir / "urban_city.jpg").exists():
        samples.append({
            "image_id": "urban_city.jpg",
            "path": str(sample_dir / "urban_city.jpg"),
            "category": "Dense Urban & Transportation",
            "query": "building, road, intersection",
            "vqa_question": "Describe the urban density and the road network pattern.",
            "vqa_expected": "High density residential/commercial grid with road network",
        })
    if (sample_dir / "agricultural_fields.jpg").exists():
        samples.append({
            "image_id": "agricultural_fields.jpg",
            "path": str(sample_dir / "agricultural_fields.jpg"),
            "category": "Agriculture & Land Use",
            "query": "field, greenhouse, vegetation",
            "vqa_question": "What agricultural patterns and vegetation types are visible?",
            "vqa_expected": "Cultivated farmland plots and vegetation",
        })

    # 2. Synthetic Calibration Tile
    synth_path = VLM_ROOT / "data" / "sample_tile.png"
    if synth_path.exists():
        samples.append({
            "image_id": "sample_tile.png",
            "path": str(synth_path),
            "category": "Synthetic Calibration",
            "query": "building, road",
            "vqa_question": "Describe the colored geometric regions in this calibration tile.",
            "vqa_expected": "Geometric shapes, intersecting grid lines, and colored zones",
        })

    # 3. Real VRSBench Dataset Validation Images
    val_images_dir = VLM_ROOT / "data" / "train_images" / "Images_val"
    vqa_file = VLM_ROOT / "data" / "train_vqa_subset.json"
    cap_file = VLM_ROOT / "data" / "train_caption_subset.json"

    if val_images_dir.exists() and vqa_file.exists():
        with open(vqa_file, "r", encoding="utf-8") as f:
            vqas = json.load(f)
        vqa_map = {item["image_id"]: item for item in vqas}

        available = sorted([f.name for f in val_images_dir.iterdir() if f.name in vqa_map])
        for img_name in available[:6]:
            item = vqa_map[img_name]
            samples.append({
                "image_id": img_name,
                "path": str(val_images_dir / img_name),
                "category": "VRSBench Satellite Benchmark",
                "query": "vehicle, building, bridge, ship",
                "vqa_question": item["question"],
                "vqa_expected": item.get("ground_truth", ""),
            })

    return samples


def main():
    parser = argparse.ArgumentParser(description="Run Final GeoNLI End-to-End Suite")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--caption-tokens", type=int, default=128)
    parser.add_argument("--vqa-tokens", type=int, default=64)
    parser.add_argument("--skip-grounding", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CAPTIONED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    samples = get_test_samples()
    print(f"\n{'='*70}")
    print(f"GeoNLI Comprehensive Final Test Suite")
    print(f"Total test samples: {len(samples)}")
    print(f"Device: {'cuda (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'cpu'}")
    print(f"{'='*70}\n")

    # Initialize YOLO detector
    print("[1/3] Loading YOLO11-OBB Detector...")
    detector = YOLODetector(str(PROJECT_ROOT / "yolo11n-obb.pt"))

    # Initialize Grounding DINO
    grounder = None
    if not args.skip_grounding:
        print("[2/3] Loading Grounding DINO Tiny...")
        try:
            grounder = Grounder(box_threshold=0.35, text_threshold=0.25)
        except Exception as e:
            print(f"Grounding DINO warning: {e}")

    # Initialize 4-bit Qwen2-VL
    print("[3/3] Loading Qwen2-VL-2B (4-bit GPU Accelerated)...")
    model_args = SimpleNamespace(
        model=DEFAULT_MODEL,
        cache_dir=str(VLM_ROOT / "models"),
        load_in_4bit=True,
        load_in_8bit=False,
        no_quantization=False,
    )
    vlm_config = build_runtime_config(model_args)
    vlm_model, vlm_processor = load_model_and_processor(vlm_config)

    rows = []
    overall_start = time.time()

    jsonl_path = OUTPUT_DIR / "final_results.jsonl"
    csv_path = OUTPUT_DIR / "final_results.csv"
    report_path = OUTPUT_DIR / "FINAL_TEST_REPORT.md"

    with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for idx, sample in enumerate(samples, start=1):
            img_path = sample["path"]
            img_id = sample["image_id"]
            category = sample["category"]
            query = sample["query"]
            vqa_q = sample["vqa_question"]
            vqa_exp = sample.get("vqa_expected", "")

            print(f"\n--- [{idx}/{len(samples)}] Processing {img_id} ({category}) ---")
            image = cv2.imread(img_path)
            if image is None:
                print(f"Error: Could not read {img_path}")
                continue

            h, w = image.shape[:2]
            sample_start = time.time()

            # 1. Tiling + YOLO
            t0 = time.time()
            tiles = generate_tiles(image, tile_h=args.tile_size, tile_w=args.tile_size, overlap=args.overlap)
            all_dets = []
            for tile in tiles:
                tile_dets = detector.detect(tile, conf=args.confidence)
                all_dets.extend(remap_to_global(tile_dets, tile))
            final_yolo = merge_detections(all_dets)
            yolo_ms = int((time.time() - t0) * 1000)
            classes_found = sorted({d["class_name"] for d in final_yolo})

            # 2. Grounding DINO
            t0 = time.time()
            grounded_dets = []
            if grounder and query:
                try:
                    grounded_dets = grounder.ground(image, query)
                except Exception as e:
                    print(f"Grounding error: {e}")
            grounding_ms = int((time.time() - t0) * 1000)

            # 3. Qwen2-VL Captioning + VQA
            t0 = time.time()
            caption_out = ""
            vqa_out = ""
            try:
                caption_out = run_inference(
                    vlm_model,
                    vlm_processor,
                    img_path,
                    DEFAULT_CAPTION_PROMPT,
                    max_new_tokens=args.caption_tokens,
                )
            except Exception as e:
                caption_out = f"Caption error: {e}"

            try:
                vqa_prompt = DEFAULT_VQA_PROMPT.format(question=vqa_q)
                vqa_out = run_inference(
                    vlm_model,
                    vlm_processor,
                    img_path,
                    vqa_prompt,
                    max_new_tokens=args.vqa_tokens,
                )
            except Exception as e:
                vqa_out = f"VQA error: {e}"
            vlm_ms = int((time.time() - t0) * 1000)

            total_ms = int((time.time() - sample_start) * 1000)

            # 4. Generate Annotated Image
            annotated = draw_annotations(image, final_yolo, grounded_dets)

            row = {
                "index": idx,
                "image_id": img_id,
                "category": category,
                "image_size": f"{w}x{h}",
                "tiles": len(tiles),
                "raw_yolo_detections": len(all_dets),
                "final_detection_count": len(final_yolo),
                "detection_classes": ", ".join(classes_found) if classes_found else "none",
                "query": query,
                "grounding_count": len(grounded_dets),
                "caption_output": caption_out,
                "vqa_question": vqa_q,
                "vqa_output": vqa_out,
                "vqa_expected": vqa_exp,
                "yolo_ms": yolo_ms,
                "grounding_ms": grounding_ms,
                "vlm_ms": vlm_ms,
                "total_ms": total_ms,
                "detections": final_yolo,
                "grounded": grounded_dets,
            }

            annotated_rel_path = write_captioned_image(row, annotated)
            row["annotated_image"] = annotated_rel_path

            # Print quick summary
            print(f" -> YOLO: {len(final_yolo)} objs ({row['detection_classes']}) in {yolo_ms}ms")
            print(f" -> Grounding: {len(grounded_dets)} objs in {grounding_ms}ms")
            print(f" -> VLM: Caption + VQA completed in {vlm_ms}ms")
            print(f" -> Total latency: {total_ms}ms ({total_ms/1000:.2f}s)")
            print(f" -> VQA Ans: {vqa_out}")

            rows.append(row)
            jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            jsonl_file.flush()

    # Write CSV
    csv_fields = [
        "index",
        "image_id",
        "category",
        "image_size",
        "tiles",
        "final_detection_count",
        "detection_classes",
        "grounding_count",
        "vqa_question",
        "vqa_output",
        "vqa_expected",
        "yolo_ms",
        "grounding_ms",
        "vlm_ms",
        "total_ms",
        "annotated_image",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Generate Markdown Report
    total_time = round(time.time() - overall_start, 2)
    avg_yolo = round(sum(r["yolo_ms"] for r in rows) / len(rows), 1) if rows else 0
    avg_grounding = round(sum(r["grounding_ms"] for r in rows) / len(rows), 1) if rows else 0
    avg_vlm = round(sum(r["vlm_ms"] for r in rows) / len(rows), 1) if rows else 0
    avg_total = round(sum(r["total_ms"] for r in rows) / len(rows), 1) if rows else 0

    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_alloc = round(torch.cuda.memory_allocated() / (1024**3), 2) if torch.cuda.is_available() else 0

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# GeoNLI Final Comprehensive Testing & Benchmark Report\n\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Images Evaluated:** {len(rows)}\n")
        f.write(f"- **Execution Hardware:** {cuda_name}\n")
        f.write(f"- **VLM Quantization:** 4-bit NF4 (`BitsAndBytesConfig`)\n")
        f.write(f"- **Peak GPU VRAM Allocated:** {vram_alloc} GB\n")
        f.write(f"- **Total Benchmark Duration:** {total_time} seconds\n")
        f.write(f"- **Average End-to-End Latency per Image:** {avg_total/1000:.2f} s\n\n")

        f.write("## Average Latency Breakdown\n\n")
        f.write("| Component | Sub-task | Average Latency | Acceleration Mode |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write(f"| **Tiling + YOLO11-OBB** | Multi-tile Object Proposals | `{avg_yolo} ms` | PyTorch / CUDA |\n")
        f.write(f"| **Grounding DINO** | Text-Guided Open Grounding | `{avg_grounding} ms` | PyTorch / CUDA |\n")
        f.write(f"| **Qwen2-VL-2B VLM** | Captioning + VQA Reasoning | `{avg_vlm} ms` | 4-bit NF4 GPU |\n")
        f.write(f"| **Full End-to-End Pipeline** | All Models + Annotation | `{avg_total} ms` ({avg_total/1000:.2f}s) | Fully Integrated |\n\n")

        f.write("## Detailed Test Results\n\n")
        f.write("| # | Image | Category | YOLO Detections | Grounded | VQA Question & Answer | Total Latency |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in rows:
            vqa_snippet = f"**Q:** {r['vqa_question']}<br>**A:** {r['vqa_output']}"
            f.write(
                f"| {r['index']} | `{r['image_id']}` | {r['category']} | {r['final_detection_count']} ({r['detection_classes']}) | {r['grounding_count']} | {vqa_snippet} | {r['total_ms']/1000:.2f}s |\n"
            )

        f.write("\n## Sample Visual Walkthrough\n\n")
        for r in rows:
            f.write(f"### Sample {r['index']}: {r['image_id']} ({r['category']})\n\n")
            f.write(f"![{r['image_id']}]({r['annotated_image']})\n\n")
            f.write(f"- **Image Size**: `{r['image_size']}` ({r['tiles']} tiles)\n")
            f.write(f"- **YOLO Detections**: {r['final_detection_count']} proposals ({r['detection_classes']})\n")
            f.write(f"- **Grounding Query**: *\"{r['query']}\"* -> `{r['grounding_count']}` matches\n")
            f.write(f"- **Generated Caption**: {r['caption_output']}\n")
            f.write(f"- **VQA Question**: {r['vqa_question']}\n")
            f.write(f"- **VQA Answer**: `{r['vqa_output']}` *(Expected: {r['vqa_expected']})*\n")
            f.write(f"- **Latency Breakdown**: YOLO `{r['yolo_ms']}ms` | Grounding `{r['grounding_ms']}ms` | VLM `{r['vlm_ms']}ms` | Total: `{r['total_ms']}ms`\n\n")
            f.write("---\n\n")

    print(f"\n{'='*70}")
    print(f"Final testing completed in {total_time}s!")
    print(f"Summary Report : {report_path}")
    print(f"CSV Results    : {csv_path}")
    print(f"JSONL Results  : {jsonl_path}")
    print(f"Captioned Imgs : {CAPTIONED_IMAGE_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
