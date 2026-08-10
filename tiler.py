import math
from dataclasses import dataclass

from shapely.geometry import Polygon
from shapely.affinity import rotate as shapely_rotate


@dataclass
class Tile:
    """
    Represents one tile extracted from the original image.
    """

    image: any
    x: int
    y: int
    width: int
    height: int


# --------------------------------------------------------------------
# Tile Generation
# --------------------------------------------------------------------

def generate_tiles(img, tile_h, tile_w, overlap=0):

    img_height, img_width = img.shape[:2]

    stride_h = tile_h - overlap
    stride_w = tile_w - overlap

    tiles = []

    y = 0

    while y < img_height:

        x = 0

        while x < img_width:

            y_end = min(y + tile_h, img_height)
            x_end = min(x + tile_w, img_width)

            tile = Tile(
                image=img[y:y_end, x:x_end],
                x=x,
                y=y,
                width=x_end - x,
                height=y_end - y
            )

            tiles.append(tile)

            if x_end == img_width:
                break

            x += stride_w

        if y_end == img_height:
            break

        y += stride_h

    return tiles


# --------------------------------------------------------------------
# Coordinate Mapping
# --------------------------------------------------------------------

def remap_to_global(detections, tile, edge_margin=5):

    tile_h = tile.height
    tile_w = tile.width

    global_dets = []

    for d in detections:

        x1 = d["cx"] - d["w"] / 2 #calculate detection bounds 
                                #c varialbles are from center
        x2 = d["cx"] + d["w"] / 2

        y1 = d["cy"] - d["h"] / 2
        y2 = d["cy"] + d["h"] / 2

        touches_edge = (

            x1 <= edge_margin or
            x2 >= tile_w - edge_margin or

            y1 <= edge_margin or
            y2 >= tile_h - edge_margin

        ) # edge margin is later required for merger so that it can merge detection without any overlaps

        global_dets.append({

            "cx": d["cx"] + tile.x,
            "cy": d["cy"] + tile.y,

            "w": d["w"],
            "h": d["h"],

            "angle_deg": d["angle_deg"],

            "score": d["score"],

            "class_id": d["class_id"],
            "class_name": d["class_name"],

            "touches_edge": touches_edge

        })

    return global_dets


# --------------------------------------------------------------------
# Rotated IoU
# --------------------------------------------------------------------

def obb_to_polygon(cx, cy, w, h, angle_deg):

    hw = w / 2
    hh = h / 2
    # This turns the OBB into a shapely polygon
    # so we can have a IOU for two boxes
    box = Polygon([
        (-hw, -hh),
        ( hw, -hh),
        ( hw,  hh),
        (-hw,  hh)
    ])

    box = shapely_rotate(box, angle_deg, origin=(0, 0))

    return Polygon([
        (x + cx, y + cy)
        for x, y in box.exterior.coords
    ])


def rotated_iou(a, b):

    pa = obb_to_polygon(
        a["cx"], a["cy"],
        a["w"], a["h"],
        a["angle_deg"]
    )

    pb = obb_to_polygon(
        b["cx"], b["cy"],
        b["w"], b["h"],
        b["angle_deg"]
    )

    if not pa.is_valid or not pb.is_valid:
        return 0.0

    inter = pa.intersection(pb).area
    union = pa.union(pb).area

    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------
# Detection Merger
# --------------------------------------------------------------------

def merge_detections(
    detections,
    iou_threshold=0.7,
    edge_center_dist=40
):

    groups = {}

    for d in detections:
        groups.setdefault(d["class_id"], []).append(d) #This groups the same object type together so we don't have to compare two entirely different objects.

    final = []

    for class_id, group in groups.items():

        group = sorted(
            group,
            key=lambda d: d["score"],
            reverse=True
        ) # sorting to keep the highest confidence one

        suppressed = [False] * len(group) #this is later used to decide which detections to keep if true : suppress that detection.

        # Standard Rotated NMS
        for i, di in enumerate(group): #enumerate through the group 

            if suppressed[i]: # if the detection has already been suppressed then continue
                continue

            for j in range(i + 1, len(group)): #else this part compares it with the other detections and suppress if above iou threshold

                if suppressed[j]:
                    continue

                if rotated_iou(di, group[j]) > iou_threshold:
                    suppressed[j] = True

        # Edge cleanup
        kept = [
            i for i in range(len(group))
            if not suppressed[i]
        ]#keep the indexes of all detections which are kept

        for i in kept:

            di = group[i]

            if not di["touches_edge"]: #
                continue

            for j in kept:

                if i == j :
                    continue

                dj = group[j]

                if dj["touches_edge"]:
                    continue

                dist = math.hypot(

                    di["cx"] - dj["cx"],
                    di["cy"] - dj["cy"]

                )

                if dist < edge_center_dist:

                    suppressed[i] = True
                    break

        final.extend(

            group[i]

            for i in range(len(group))

            if not suppressed[i]

        )

    return final