import math
from ultralytics import YOLO


class YOLODetector:
    """
    Wrapper around the YOLO11 OBB detector.

    Responsibilities:
    - Load the model.
    - Run inference on a single tile.
    - Return detections in a standard format."""

    def __init__(self, model_path="yolo11n-obb.pt"):
        self.model = YOLO(model_path)
        self.class_names = self.model.names

    def detect(self, tile):
        """
        Parameters
        ----------
        tile : Tile
            Tile object from tiling.py

        Returns
        -------
        list[dict]
            List of detections in tile-local coordinates.
        """

        results = self.model.predict(
            tile.image,
            verbose=False
        )

        result = results[0]

        detections = []

        # No detections
        if result.obb is None or len(result.obb) == 0:
            return detections

        xywhr = result.obb.xywhr.cpu().numpy()
        scores = result.obb.conf.cpu().numpy()
        classes = result.obb.cls.cpu().numpy().astype(int)

        for (cx, cy, w, h, angle_rad), score, class_id in zip(
            xywhr,
            scores,
            classes
        ):

            detections.append({

                "cx": float(cx),
                "cy": float(cy),

                "w": float(w),
                "h": float(h),

                "angle_deg": math.degrees(angle_rad),

                "score": float(score),

                "class_id": class_id,
                "class_name": self.class_names[class_id]

            })

        return detections