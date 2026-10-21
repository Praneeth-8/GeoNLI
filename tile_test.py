
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from shapely.geometry import Polygon

from tiler import generate_tiles, remap_to_global, merge_detections

# Configuration
MODEL_PATH = "yolo11n-obb.pt"

DATASET_ROOT = Path("DATASET_PATH")
IMAGE_DIR = DATASET_ROOT / "images" / "train"
LABEL_DIR = DATASET_ROOT / "labels" / "train"

TILE_SIZE = 640
OVERLAP = 64
CONF = 0.25
IOU_THRESHOLD = 0.50


def box_polygon(cx, cy, w, h, angle_deg):
    from shapely.affinity import rotate, translate

    box = Polygon([
        (-w/2, -h/2), (w/2, -h/2),
        (w/2, h/2), (-w/2, h/2)
    ])
    return translate(
        rotate(box, angle_deg, origin=(0, 0)),
        xoff=cx, yoff=cy
    )


def polygon_iou(a, b):
    pa, pb = Polygon(a), Polygon(b)
    if not pa.is_valid or not pb.is_valid:
        return 0.0
    union = pa.union(pb).area
    return pa.intersection(pb).area / union if union else 0.0


def read_gt(path, width, height):
    output = []
    if not path.exists():
        raise FileNotFoundError(f"Missing annotation: {path}")

    for line in path.read_text().splitlines():
        values = line.split()
        if len(values) != 9:
            continue

        cls = int(values[0])
        coords = np.array(list(map(float, values[1:])))
        coords[0::2] *= width
        coords[1::2] *= height

        output.append({
            "class_id": cls,
            "poly": coords.reshape(4, 2)
        })

    return output


def predict_full(model, image):
    result = model.predict(
        image, imgsz=TILE_SIZE, conf=CONF, verbose=False
    )[0]
    output = []

    if result.obb is None:
        return output

    boxes = result.obb.xywhr.cpu().numpy()
    scores = result.obb.conf.cpu().numpy()
    classes = result.obb.cls.cpu().numpy().astype(int)

    for (cx, cy, w, h, angle), score, cls in zip(
        boxes, scores, classes
    ):
        output.append({
            "class_id": int(cls),
            "score": float(score),
            "poly": np.asarray(box_polygon(
                cx, cy, w, h, np.degrees(angle)
            ).exterior.coords[:-1])
        })

    return output


def predict_tiled(model, image):
    raw = []

    for tile in generate_tiles(
        image, TILE_SIZE, TILE_SIZE, OVERLAP
    ):
        result = model.predict(
            tile.image, imgsz=TILE_SIZE, conf=CONF, verbose=False
        )[0]

        if result.obb is None:
            continue

        boxes = result.obb.xywhr.cpu().numpy()
        scores = result.obb.conf.cpu().numpy()
        classes = result.obb.cls.cpu().numpy().astype(int)

        local = []
        for (cx, cy, w, h, angle), score, cls in zip(
            boxes, scores, classes
        ):
            local.append({
                "cx": float(cx), "cy": float(cy),
                "w": float(w), "h": float(h),
                "angle_deg": float(np.degrees(angle)),
                "score": float(score),
                "class_id": int(cls),
                "class_name": str(cls)
            })

        raw.extend(remap_to_global(local, tile))

    merged = merge_detections(raw)
    output = []

    for d in merged:
        poly = box_polygon(
            d["cx"], d["cy"], d["w"], d["h"], d["angle_deg"]
        )
        output.append({
            "class_id": d["class_id"],
            "score": d["score"],
            "poly": np.asarray(poly.exterior.coords[:-1])
        })

    return output, len(raw)


def count_matches(gt, pred):
    matched = set()
    tp = fp = 0

    for d in sorted(pred, key=lambda x: x["score"], reverse=True):
        candidates = [
            (polygon_iou(d["poly"], g["poly"]), i)
            for i, g in enumerate(gt)
            if i not in matched and g["class_id"] == d["class_id"]
        ]

        best_iou, best_idx = max(candidates, default=(0.0, -1))

        if best_iou >= IOU_THRESHOLD:
            tp += 1
            matched.add(best_idx)
        else:
            fp += 1

    return tp, fp, len(gt) - len(matched)


def main():
    if not IMAGE_DIR.exists() or not LABEL_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_ROOT}. "
            "Run the DOTA128 download/validation step first."
        )

    model = YOLO(MODEL_PATH)
    paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif"}
    )

    if not paths:
        raise RuntimeError(f"No images found in {IMAGE_DIR}")

    rows = []

    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue

        h, w = image.shape[:2]
        gt = read_gt(LABEL_DIR / f"{path.stem}.txt", w, h)

        start = time.perf_counter()
        full = predict_full(model, image)
        full_time = time.perf_counter() - start

        start = time.perf_counter()
        tiled, raw_count = predict_tiled(model, image)
        tiled_time = time.perf_counter() - start

        full_tp, full_fp, full_fn = count_matches(gt, full)
        tile_tp, tile_fp, tile_fn = count_matches(gt, tiled)

        rows.append({
            "image": path.name,
            "ground_truth": len(gt),
            "full_tp": full_tp, "full_fp": full_fp,
            "full_fn": full_fn,
            "tiled_tp": tile_tp, "tiled_fp": tile_fp,
            "tiled_fn": tile_fn,
            "raw_tile_detections": raw_count,
            "merged_detections": len(tiled),
            "full_seconds": full_time,
            "tiled_seconds": tiled_time
        })

        print(f"Evaluated {path.name}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No images were successfully evaluated.")

    df.to_csv("tiler_metrics.csv", index=False)

    for name in ("full", "tiled"):
        tp = int(df[f"{name}_tp"].sum())
        fp = int(df[f"{name}_fp"].sum())
        fn = int(df[f"{name}_fn"].sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        print(f"\n{name.upper()} RESULTS")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1-score:  {f1:.4f}")
        print(f"TP={tp}, FP={fp}, FN={fn}")
        print(
            "Mean seconds/image:",
            round(df[f"{name}_seconds"].mean(), 3)
        )

    print("\nPer-image metrics saved to tiler_metrics.csv")
    print("IoU threshold:", IOU_THRESHOLD)
    print("Tile overlap:", OVERLAP)


if __name__ == "__main__":
    main()
