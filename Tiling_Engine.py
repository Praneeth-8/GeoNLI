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