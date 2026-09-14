import argparse
import os

import cv2

from tiler import (
    generate_tiles,
    remap_to_global,
    merge_detections
)

from detector import YOLODetector
from grounding import Grounder


def parse_args():
    parser = argparse.ArgumentParser(description="Run tiling and YOLO OBB grounding.")
    parser.add_argument(
        "--image",
        default=os.path.join("vlm-pipeline", "data", "sample_tile.png"),
        help="Input image path.",
    )
    parser.add_argument("--model", default="yolo11n-obb.pt", help="YOLO OBB model path or name.")
    parser.add_argument("--tile-size", type=int, default=512, help="Square tile size.")
    parser.add_argument("--overlap", type=int, default=64, help="Tile overlap in pixels.")
    parser.add_argument(
        "--query",
        help="Optional text query for Grounding DINO, for example 'house' or 'solar panel'.",
    )
    parser.add_argument(
        "--grounding-model",
        default="IDEA-Research/grounding-dino-tiny",
        help="Grounding DINO model id to use when --query is provided.",
    )
    parser.add_argument(
        "--grounding-box-threshold",
        type=float,
        default=0.35,
        help="Grounding DINO box confidence threshold.",
    )
    parser.add_argument(
        "--grounding-text-threshold",
        type=float,
        default=0.25,
        help="Grounding DINO text matching threshold.",
    )
    parser.add_argument("--output", default="detections.png", help="Annotated output image path.")
    parser.add_argument("--show", action="store_true", help="Display the annotated image window.")
    return parser.parse_args()


def draw_detections(image, detections):
    for det in detections:
        rect = (
            (det["cx"], det["cy"]),
            (det["w"], det["h"]),
            det["angle_deg"],
        )
        box = cv2.boxPoints(rect).astype(int)

        cv2.polylines(image, [box], True, (0, 255, 0), 2)
        cv2.putText(
            image,
            det["class_name"],
            (int(det["cx"]), int(det["cy"])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )


def draw_grounded_detections(image, detections):
    for det in detections:
        x1 = int(det["x1"])
        y1 = int(det["y1"])
        x2 = int(det["x2"])
        y2 = int(det["y2"])
        label = f'{det["label"]} {det["score"]:.2f}'

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            image,
            label,
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )


def main():
    args = parse_args()
    image = cv2.imread(args.image)

    if image is None:
        raise FileNotFoundError(f"Couldn't open {args.image}")

    annotated_image = image.copy()

    tiles = generate_tiles(
        img=image,
        tile_h=args.tile_size,
        tile_w=args.tile_size,
        overlap=args.overlap,
    )
    print(f"Generated {len(tiles)} tiles")

    detector = YOLODetector(args.model)
    all_detections = []

    for tile in tiles:
        detections = remap_to_global(detector.detect(tile), tile)
        all_detections.extend(detections)

    final_detections = merge_detections(all_detections)

    print(f"Raw detections   : {len(all_detections)}")
    print(f"Final detections : {len(final_detections)}")

    draw_detections(annotated_image, final_detections)

    if args.query:
        grounder = Grounder(
            model_id=args.grounding_model,
            box_threshold=args.grounding_box_threshold,
            text_threshold=args.grounding_text_threshold,
        )
        grounded_detections = grounder.ground(image, args.query)
        draw_grounded_detections(annotated_image, grounded_detections)
        print(f"Grounded detections ({args.query}): {len(grounded_detections)}")

    cv2.imwrite(args.output, annotated_image)
    print(f"Wrote {args.output}")

    if args.show:
        window_title = "YOLO + Grounding DINO" if args.query else "Detections"
        cv2.imshow(window_title, annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
