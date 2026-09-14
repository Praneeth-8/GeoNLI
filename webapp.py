import io
import json
import math
import os
import posixpath
import shutil
import sys
import time
import traceback
import uuid
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
RUNS_ROOT = FRONTEND_ROOT / "runs"
UPLOADS_ROOT = FRONTEND_ROOT / "uploads"
SAMPLES_ROOT = PROJECT_ROOT / "vlm-pipeline" / "data" / "test_samples"

os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "Ultralytics"))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
VLM_PATH = PROJECT_ROOT / "vlm-pipeline"
if str(VLM_PATH) not in sys.path:
    sys.path.insert(0, str(VLM_PATH))

from detector import YOLODetector
from grounding import Grounder
from tiler import generate_tiles, merge_detections, remap_to_global
from test_inference import (
    DEFAULT_CAPTION_PROMPT,
    DEFAULT_VQA_PROMPT,
    build_runtime_config,
    load_model_and_processor,
    run_inference,
)


class PipelineRuntime:
    def __init__(self):
        self.detector = None
        self.grounder_cache = {}
        self.vlm_bundle = None
        self.vlm_load_error = None

    def get_detector(self, model_path="yolo11n-obb.pt"):
        if self.detector is None:
            self.detector = YOLODetector(model_path)
        return self.detector

    def get_grounder(self, model_id="IDEA-Research/grounding-dino-tiny", box_threshold=0.35, text_threshold=0.25):
        key = (model_id, float(box_threshold), float(text_threshold))
        if key not in self.grounder_cache:
            self.grounder_cache[key] = Grounder(
                model_id=model_id,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
        return self.grounder_cache[key]

    def get_vlm_bundle(self):
        if self.vlm_bundle is None:
            class Args:
                model = "Qwen/Qwen2-VL-2B-Instruct"
                cache_dir = str(VLM_PATH / "models")
                load_in_4bit = True
                load_in_8bit = False
                no_quantization = False

            try:
                config = build_runtime_config(Args)
                self.vlm_bundle = load_model_and_processor(config)
                self.vlm_load_error = None
            except Exception as e:
                self.vlm_load_error = str(e)
                raise RuntimeError(f"Could not initialize Vision-Language model: {e}")
        return self.vlm_bundle


runtime = PipelineRuntime()


def ensure_dirs():
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "Ultralytics").mkdir(parents=True, exist_ok=True)


def to_builtin(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def normalize_records(records):
    normalized = []
    for record in records:
        normalized.append({key: to_builtin(value) for key, value in record.items()})
    return normalized


# Distinct palette for classes
CLASS_COLORS = [
    (0, 220, 115),   # Emerald Green
    (255, 170, 0),   # Bright Amber
    (0, 180, 255),   # Cyan
    (255, 75, 130),  # Pink/Rose
    (180, 100, 255), # Purple
    (255, 230, 0),   # Yellow
    (0, 240, 255),   # Neon Sky
    (255, 110, 0),   # Orange
]


def draw_label_badge(image, text, x, y, color=(0, 220, 115), text_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    
    x = max(2, min(int(x), image.shape[1] - tw - 6))
    y = max(th + 6, min(int(y), image.shape[0] - 4))
    
    cv2.rectangle(image, (x - 2, y - th - 4), (x + tw + 4, y + baseline), color, -1)
    cv2.putText(image, text, (x + 1, y - 2), font, scale, text_color, thickness, cv2.LINE_AA)


def draw_yolo_detections(image, detections):
    for det in detections:
        cid = det.get("class_id", 0)
        color = CLASS_COLORS[cid % len(CLASS_COLORS)]
        rect = ((det["cx"], det["cy"]), (det["w"], det["h"]), det["angle_deg"])
        box = cv2.boxPoints(rect).astype(int)
        
        cv2.polylines(image, [box], True, color, 2, cv2.LINE_AA)
        
        label = f'{det["class_name"]} {det.get("score", 0):.2f}'
        draw_label_badge(image, label, det["cx"] - det["w"] / 4, det["cy"] - det["h"] / 4, color=color)


def draw_grounding_detections(image, detections):
    color = (255, 50, 50) # Bright Blue/Red in BGR
    for det in detections:
        x1 = int(det["x1"])
        y1 = int(det["y1"])
        x2 = int(det["x2"])
        y2 = int(det["y2"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (50, 120, 255), 2, cv2.LINE_AA)
        label = f'G: {det["label"]} {det.get("score", 0):.2f}'
        draw_label_badge(image, label, x1, max(y1, 14), color=(50, 120, 255), text_color=(255, 255, 255))


def save_upload(file_item):
    suffix = Path(file_item.filename or "upload.png").suffix or ".png"
    stem = Path(file_item.filename or "upload").stem
    safe_stem = "".join(char if char.isalnum() or char in "-_" else "_" for char in stem)
    filename = f"{safe_stem}_{uuid.uuid4().hex[:8]}{suffix}"
    destination = UPLOADS_ROOT / filename
    with destination.open("wb") as target:
        shutil.copyfileobj(file_item.file, target)
    return destination


def parse_multipart_form(handler):
    content_length = int(handler.headers.get("Content-Length", "0"))
    body = handler.rfile.read(content_length)
    content_type = handler.headers.get("Content-Type", "")
    
    headers = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
    message = BytesParser(policy=default).parsebytes(headers + body)

    fields = {}
    files = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue

        filename = part.get_filename()
        if filename:
            payload = part.get_payload(decode=True)
            if payload is None:
                payload = part.get_content()
                if isinstance(payload, str):
                    payload = payload.encode("utf-8")
            files[name] = type(
                "UploadedFile",
                (),
                {
                    "filename": filename,
                    "file": io.BytesIO(payload or b""),
                },
            )()
        else:
            val = part.get_content()
            if isinstance(val, bytes):
                val = val.decode("utf-8", errors="replace")
            fields[name] = str(val).strip()

    return fields, files


def relative_frontend_path(path):
    return posixpath.join(*path.relative_to(FRONTEND_ROOT).parts)


def get_sample_catalog():
    samples = []
    sample_files = [
        ("airport.jpg", "Commercial Airport Runway & Aircraft", "aircraft, runway, plane", "How many airplanes and jet bridges are visible?"),
        ("urban_city.jpg", "Dense Urban Settlement & Roads", "building, road, vehicle", "Describe the urban density and road layout."),
        ("agricultural_fields.jpg", "Agricultural Farmland & Greenhouses", "field, road, tree", "What type of agricultural patterns and vegetation are present?"),
    ]
    
    for filename, title, query, vqa in sample_files:
        path = SAMPLES_ROOT / filename
        if path.exists():
            samples.append({
                "id": filename,
                "title": title,
                "filename": filename,
                "default_query": query,
                "default_vqa": vqa,
                "url": f"/api/sample/{filename}",
            })
            
    # Also check synthetic sample
    synth_path = PROJECT_ROOT / "vlm-pipeline" / "data" / "sample_tile.png"
    if synth_path.exists():
        samples.append({
            "id": "sample_tile.png",
            "title": "Synthetic Sensor Calibration Tile",
            "filename": "sample_tile.png",
            "default_query": "building, road",
            "default_vqa": "Describe the scene layout and color regions.",
            "url": "/api/sample/sample_tile.png",
        })
        
    return samples


def run_pipeline(
    image_path,
    query="",
    vqa_question="",
    tile_size=512,
    overlap=64,
    confidence=0.25,
    run_yolo=True,
    run_grounding=True,
    run_captioning=True,
    run_vqa=True,
):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = image.shape[:2]
    timings = {}
    
    # 1. Tiling & YOLO Detection
    detections = []
    raw_count = 0
    tiles_count = 0
    
    if run_yolo:
        t0 = time.time()
        detector = runtime.get_detector()
        tiles = generate_tiles(image, tile_h=tile_size, tile_w=tile_size, overlap=overlap)
        tiles_count = len(tiles)
        all_detections = []

        for tile in tiles:
            tile_dets = detector.detect(tile, conf=confidence)
            mapped = remap_to_global(tile_dets, tile)
            all_detections.extend(mapped)

        detections = merge_detections(all_detections)
        raw_count = len(all_detections)
        timings["yolo_ms"] = int((time.time() - t0) * 1000)

    # 2. Grounding DINO
    grounded = []
    if run_grounding and query and query.strip():
        t0 = time.time()
        try:
            grounder = runtime.get_grounder("IDEA-Research/grounding-dino-tiny", 0.35, 0.25)
            grounded = grounder.ground(image, query)
        except Exception as e:
            print(f"Grounding DINO warning: {e}")
            grounded = []
        timings["grounding_ms"] = int((time.time() - t0) * 1000)

    # 3. Annotate
    annotated = image.copy()
    if run_yolo and detections:
        draw_yolo_detections(annotated, detections)
    if run_grounding and grounded:
        draw_grounding_detections(annotated, grounded)

    run_id = uuid.uuid4().hex[:10]
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "annotated.png"
    cv2.imwrite(str(output_path), annotated)

    # 4. Vision Language Model (Qwen2-VL)
    caption = None
    vqa_answer = None

    if run_captioning or (run_vqa and vqa_question):
        t0 = time.time()
        try:
            model, processor = runtime.get_vlm_bundle()
            if run_captioning:
                caption = run_inference(model, processor, str(image_path), DEFAULT_CAPTION_PROMPT, max_new_tokens=256)
            if run_vqa and vqa_question:
                vqa_answer = run_inference(
                    model,
                    processor,
                    str(image_path),
                    DEFAULT_VQA_PROMPT.format(question=vqa_question),
                    max_new_tokens=128,
                )
        except Exception as exc:
            print(f"VLM inference warning: {exc}")
            if run_captioning and not caption:
                caption = f"VLM Inference Note: {exc}"
            if run_vqa and vqa_question and not vqa_answer:
                vqa_answer = f"VLM Inference Note: {exc}"
        timings["vlm_ms"] = int((time.time() - t0) * 1000)

    return {
        "annotated_path": output_path,
        "width": w,
        "height": h,
        "tiles": tiles_count,
        "raw_detection_count": raw_count,
        "final_detection_count": len(detections),
        "grounding_count": len(grounded),
        "detections": normalize_records(detections),
        "grounded": normalize_records(grounded),
        "caption": caption,
        "vqa_answer": vqa_answer,
        "timings": timings,
    }


class GeoNLIRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_ROOT), **kwargs)

    def send_json(self, status_code, data):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in {"/", "/index.html"}:
            self.path = "/index.html"
            return super().do_GET()

        if path == "/api/health":
            import torch
            vram_free, vram_total = (0, 0)
            if torch.cuda.is_available():
                try:
                    f, t = torch.cuda.mem_get_info()
                    vram_free, vram_total = f / (1024**2), t / (1024**2)
                except Exception:
                    pass
            self.send_json(HTTPStatus.OK, {
                "status": "online",
                "cuda_available": torch.cuda.is_available(),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "vram_free_mb": round(vram_free, 1),
                "vram_total_mb": round(vram_total, 1),
                "models": {
                    "yolo_obb": "YOLO11n-OBB",
                    "grounding_dino": "IDEA-Research/grounding-dino-tiny",
                    "vlm": "Qwen/Qwen2-VL-2B-Instruct",
                }
            })
            return

        if path == "/api/samples":
            self.send_json(HTTPStatus.OK, {"samples": get_sample_catalog()})
            return

        if path.startswith("/api/sample/"):
            filename = posixpath.basename(unquote(path))
            target_path = SAMPLES_ROOT / filename
            if not target_path.exists():
                target_path = PROJECT_ROOT / "vlm-pipeline" / "data" / filename
            if target_path.exists() and target_path.is_file():
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg" if filename.endswith((".jpg", ".jpeg")) else "image/png")
                self.send_header("Content-Length", str(target_path.stat().st_size))
                self.end_headers()
                with target_path.open("rb") as f:
                    shutil.copyfileobj(f, self.wfile)
                return
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Sample not found")
                return

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/analyze":
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
            return

        try:
            content_type = self.headers.get("Content-Type", "")
            target_image_path = None
            
            if "multipart/form-data" in content_type:
                fields, files = parse_multipart_form(self)
                image_item = files.get("image")
                sample_id = fields.get("sample_id", "").strip()

                if image_item and getattr(image_item, "filename", ""):
                    target_image_path = save_upload(image_item)
                elif sample_id:
                    sample_path = SAMPLES_ROOT / sample_id
                    if not sample_path.exists():
                        sample_path = PROJECT_ROOT / "vlm-pipeline" / "data" / sample_id
                    if sample_path.exists():
                        dest = UPLOADS_ROOT / f"sample_{sample_id}"
                        shutil.copy2(sample_path, dest)
                        target_image_path = dest
                    else:
                        raise ValueError(f"Sample image '{sample_id}' not found.")
                else:
                    raise ValueError("Please select a sample image or upload a satellite image.")

                query = fields.get("query", "").strip()
                vqa_question = fields.get("vqa_question", "").strip()
                tile_size = int(fields.get("tile_size", "512"))
                overlap = int(fields.get("overlap", "64"))
                confidence = float(fields.get("confidence", "0.25"))
                run_yolo = fields.get("run_yolo", "true").lower() == "true"
                run_grounding = fields.get("run_grounding", "true").lower() == "true"
                run_captioning = fields.get("run_captioning", "true").lower() == "true"
                run_vqa = fields.get("run_vqa", "true").lower() == "true"

            elif "application/json" in content_type:
                content_length = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(content_length).decode("utf-8"))
                sample_id = data.get("sample_id", "")
                if sample_id:
                    sample_path = SAMPLES_ROOT / sample_id
                    if not sample_path.exists():
                        sample_path = PROJECT_ROOT / "vlm-pipeline" / "data" / sample_id
                    if sample_path.exists():
                        dest = UPLOADS_ROOT / f"sample_{sample_id}"
                        shutil.copy2(sample_path, dest)
                        target_image_path = dest
                    else:
                        raise ValueError(f"Sample image '{sample_id}' not found.")
                else:
                    raise ValueError("No image or sample_id specified in JSON request.")

                query = data.get("query", "").strip()
                vqa_question = data.get("vqa_question", "").strip()
                tile_size = int(data.get("tile_size", 512))
                overlap = int(data.get("overlap", 64))
                confidence = float(data.get("confidence", 0.25))
                run_yolo = bool(data.get("run_yolo", True))
                run_grounding = bool(data.get("run_grounding", True))
                run_captioning = bool(data.get("run_captioning", True))
                run_vqa = bool(data.get("run_vqa", True))
            else:
                raise ValueError(f"Unsupported content type: {content_type}")

            result = run_pipeline(
                target_image_path,
                query=query,
                vqa_question=vqa_question,
                tile_size=tile_size,
                overlap=overlap,
                confidence=confidence,
                run_yolo=run_yolo,
                run_grounding=run_grounding,
                run_captioning=run_captioning,
                run_vqa=run_vqa,
            )

            response = {
                "status": "success",
                "input_image": relative_frontend_path(target_image_path),
                "annotated_image": relative_frontend_path(result["annotated_path"]),
                "query": query,
                "vqa_question": vqa_question,
                "image_width": result["width"],
                "image_height": result["height"],
                "tiles": result["tiles"],
                "raw_detection_count": result["raw_detection_count"],
                "final_detection_count": result["final_detection_count"],
                "grounding_count": result["grounding_count"],
                "detections": result["detections"],
                "grounded": result["grounded"],
                "caption": result["caption"],
                "vqa_answer": result["vqa_answer"],
                "timings": result["timings"],
            }
            self.send_json(HTTPStatus.OK, response)

        except Exception as exc:
            traceback.print_exc()
            self.send_json(HTTPStatus.BAD_REQUEST, {
                "status": "error",
                "error": str(exc),
                "details": traceback.format_exc(limit=4),
            })


def main():
    ensure_dirs()
    port = 8081
    server = ThreadingHTTPServer(("0.0.0.0", port), GeoNLIRequestHandler)
    print(f"==================================================")
    print(f"GeoNLI Studio Web Server running at:")
    print(f"http://127.0.0.1:{port}")
    print(f"==================================================")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()


if __name__ == "__main__":
    main()
