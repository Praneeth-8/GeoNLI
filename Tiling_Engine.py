import cv2
import math
from ultralytics import YOLO
from shapely.geometry import Polygon
from shapely.affinity import rotate as shapely_rotate
IMAGE_PATH="" 


img = cv2.imread(IMAGE_PATH) # numpy array of the image
def generate_tiles(img,tile_h,tile_w,overlap=0):
    img_height=img.shape[0]
    img_width=img.shape[1]
    stride_h, stride_w = tile_h - overlap,tile_w - overlap # overlap in pixels
    tiles=[]

    y=0

    while y < img_height:
        x = 0
        while x < img_width:
            y_end, x_end = min(y+tile_h,img_height),min(x+tile_w,img_width) # for bounds
            tile = img[y:y_end, x:x_end]
            tiles.append({"tile": tile, "x":x, "y":y})
            if x_end == img_width:
                break
            x+=stride_w
        if y_end == img_height:
            break
        y += stride_h

    return tiles

model = YOLO("yolo11n-obb.pt")

def yolo_detect_fn(tile_img):
    results = model.predict(tile_img, verbose=False)
    result = results[0]

    detections = []
    if result.obb is None or len(result.obb) == 0:
        return detections

    xywhr = result.obb.xywhr.cpu().numpy()   # angle comes back in RADIANS
    confs = result.obb.conf.cpu().numpy()
    classes = result.obb.cls.cpu().numpy().astype(int)

    for (cx, cy, w, h, angle_rad), conf, cls_id in zip(xywhr, confs, classes):
        detections.append({
            "cx": float(cx), "cy": float(cy),
            "w": float(w), "h": float(h),
            "angle_deg": math.degrees(angle_rad),
            "score": float(conf),
            "class_id": int(cls_id),
        })
    return detections

def remap_to_global(detections, tile, edge_margin=5):
    tile_h, tile_w = tile["tile"].shape[0], tile["tile"].shape[1]
    global_dets = []

    for d in detections:
        x1, x2 = d["cx"] - d["w"] / 2, d["cx"] + d["w"] / 2
        y1, y2 = d["cy"] - d["h"] / 2, d["cy"] + d["h"] / 2

        touches_edge = (
            x1 <= edge_margin or x2 >= tile_w - edge_margin or
            y1 <= edge_margin or y2 >= tile_h - edge_margin
        )

        global_dets.append({
            "cx": d["cx"] + tile["x"],
            "cy": d["cy"] + tile["y"],
            "w": d["w"], "h": d["h"],
            "angle_deg": d["angle_deg"],
            "score": d["score"],
            "class_id": d["class_id"],
            "touches_edge": touches_edge,
        })
    return global_dets

def obb_to_polygon(cx, cy, w, h, angle_deg):
    """Turn a (cx, cy, w, h, angle) box into a shapely polygon so we can
    compute real rotated intersection/union, not axis-aligned IoU."""
    hw, hh = w / 2, h / 2
    box = Polygon([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])
    box = shapely_rotate(box, angle_deg, origin=(0, 0))
    return Polygon([(x + cx, y + cy) for x, y in box.exterior.coords])


def rotated_iou(a, b):
    pa = obb_to_polygon(a["cx"], a["cy"], a["w"], a["h"], a["angle_deg"])
    pb = obb_to_polygon(b["cx"], b["cy"], b["w"], b["h"], b["angle_deg"])
    if not pa.is_valid or not pb.is_valid:
        return 0.0
    inter = pa.intersection(pb).area
    union = pa.union(pb).area
    return inter / union if union > 0 else 0.0


def merge_detections(detections, iou_threshold=0.5, edge_center_dist=40):
    """
    detections: flat list of global-coordinate detection dicts
        (cx, cy, w, h, angle_deg, score, class_id, touches_edge)
        -- output of remap_to_global() across all tiles, concatenated.

    Returns: one detection per real object.
    """
    groups = {}
    for d in detections:
        groups.setdefault(d["class_id"], []).append(d)

    final = []
    for class_id, group in groups.items():
        group = sorted(group, key=lambda d: d["score"], reverse=True)
        suppressed = [False] * len(group)

        # Pass 1: standard rotated NMS -- catches near-identical duplicate
        # boxes from overlapping tiles that both saw the whole object.
        for i, di in enumerate(group):
            if suppressed[i]:
                continue
            for j in range(i + 1, len(group)):
                if suppressed[j]:
                    continue
                if rotated_iou(di, group[j]) > iou_threshold:
                    suppressed[j] = True

        # Pass 2: edge-fragment cleanup -- catches truncated detections
        # near a tile boundary that IoU alone wouldn't recognize as
        # duplicates of a nearby, more complete detection.
        kept = [i for i in range(len(group)) if not suppressed[i]]
        for i in kept:
            di = group[i]
            if not di.get("touches_edge"):
                continue
            for j in kept:
                if i == j or suppressed[j]:
                    continue
                dj = group[j]
                if dj.get("touches_edge"):
                    continue
                dist = math.hypot(di["cx"] - dj["cx"], di["cy"] - dj["cy"])
                if dist < edge_center_dist:
                    suppressed[i] = True
                    break

        final.extend(group[i] for i in range(len(group)) if not suppressed[i])

    return final