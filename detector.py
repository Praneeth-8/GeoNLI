import math
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(PROJECT_ROOT, "Ultralytics"))

from ultralytics import YOLO


class YOLODetector:
    """
    Wrapper around the YOLO11 OBB detector.

    Responsibilities:
    - Load the model.
    - Run inference on a single tile with configurable confidence.
    - Return detections in a standard format.
    """

    def __init__(self, model_path="yolo11n-obb.pt"):
        self.model = YOLO(model_path)
        self.class_names = self.model.names

    def detect(self, tile, conf=0.25):
        """
        Parameters
        ----------
        tile : Tile
            Tile object from tiling.py
        conf : float
            Confidence threshold (default 0.25)

        Returns
        -------
        list[dict]
            List of detections in tile-local coordinates.
        """
        results = self.model.predict(
            tile.image,
            conf=conf,
            verbose=False,
        )

        result = results[0]
        detections = []

        if result.obb is None or len(result.obb) == 0:
            return detections

        xywhr = result.obb.xywhr.cpu().numpy()
        scores = result.obb.conf.cpu().numpy()
        classes = result.obb.cls.cpu().numpy().astype(int)

        for (cx, cy, w, h, angle_rad), score, class_id in zip(
            xywhr,
            scores,
            classes,
        ):
            detections.append({
                "cx": round(float(cx), 2),
                "cy": round(float(cy), 2),
                "w": round(float(w), 2),
                "h": round(float(h), 2),
                "angle_deg": round(math.degrees(angle_rad), 2),
                "score": round(float(score), 4),
                "class_id": int(class_id),
                "class_name": self.class_names.get(class_id, str(class_id)),
            })

        return detections
