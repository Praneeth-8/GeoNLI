import cv2
import math
from ultralytics import YOLO
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