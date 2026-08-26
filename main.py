import cv2

from tiler import (
    generate_tiles,
    remap_to_global,
    merge_detections
)

from detector import YOLODetector
from grounding import Grounder


IMAGE_PATH = "test_image.png"
QUERY = "House"


image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Couldn't open {IMAGE_PATH}"
    )


# --------------------------------------------------------------------
# Generate Tiles
# --------------------------------------------------------------------

tiles = generate_tiles(
    img=image,
    tile_h=512,
    tile_w=512,
    overlap=64
)

print(f"Generated {len(tiles)} tiles")


# --------------------------------------------------------------------
# Load YOLO Detector
# --------------------------------------------------------------------

detector = YOLODetector()


# # --------------------------------------------------------------------
# # Detect Objects
# # --------------------------------------------------------------------

all_detections = []

for tile in tiles:

    detections = detector.detect(tile)

    detections = remap_to_global(
        detections,
        tile
    )

    all_detections.extend(detections)


# --------------------------------------------------------------------
# Merge Duplicate Detections
# --------------------------------------------------------------------

final_detections = merge_detections(
    all_detections
)

print(f"Raw detections   : {len(all_detections)}")
print(f"Final detections : {len(final_detections)}")


# --------------------------------------------------------------------
# Grounding DINO
# # --------------------------------------------------------------------

grounder = Grounder()

grounded_detections = grounder.ground(
    image,
    QUERY
)

print(
    f"Grounded detections: "
    f"{len(grounded_detections)}"
)


# --------------------------------------------------------------------
# Draw YOLO OBB Detections
# --------------------------------------------------------------------

for det in final_detections:

    rect = (
        (det["cx"], det["cy"]),
        (det["w"], det["h"]),
        det["angle_deg"]
    )

    box = cv2.boxPoints(rect)
    box = box.astype(int)

    cv2.polylines(
        image,
        [box],
        True,
        (0, 255, 0),
        2
    )

    label = det["class_name"]

    cv2.putText(
        image,
        label,
        (
            int(det["cx"]),
            int(det["cy"])
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )


# --------------------------------------------------------------------
# Draw Grounding DINO Results
# --------------------------------------------------------------------

for det in grounded_detections:

    x1 = int(det["x1"])
    y1 = int(det["y1"])
    x2 = int(det["x2"])
    y2 = int(det["y2"])

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        2
    )

    label = (
        f"{det['label']} "
        f"{det['score']:.2f}"
    )

    cv2.putText(
        image,
        label,
        (x1, max(y1 - 5, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2
    )


# --------------------------------------------------------------------
# Show Result
# --------------------------------------------------------------------

cv2.imshow(
    "YOLO + Grounding DINO",
    image
)

cv2.waitKey(0)

cv2.destroyAllWindows()


# --------------------------------------------------------------------
# Save Result
# --------------------------------------------------------------------

cv2.imwrite(
    "grounding_results.png",
    image
)