import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


class Grounder:
    def __init__(
        self,
        model_id="IDEA-Research/grounding-dino-tiny",
        device=None,
        box_threshold=0.35,
        text_threshold=0.25,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
            self.device
        )
        self.model.eval()

    def ground(self, image, query):
        if not query or not str(query).strip():
            return []

        clean_query = str(query).strip().lower()
        if not clean_query.endswith("."):
            clean_query = clean_query + "."

        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        elif isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb)
            else:
                pil_image = Image.fromarray(image).convert("RGB")
        else:
            raise ValueError("Unsupported image format for Grounder")

        inputs = self.processor(images=pil_image, text=[clean_query], return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[pil_image.size[::-1]],
        )

        detections = []
        if not results:
            return detections

        result = results[0]
        labels = result.get("text_labels", result.get("labels", []))
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist() if hasattr(box, "tolist") else box
            clean_label = str(label).strip()
            detections.append(
                {
                    "x1": round(float(x1), 2),
                    "y1": round(float(y1), 2),
                    "x2": round(round(float(x2), 2)),
                    "y2": round(float(y2), 2),
                    "w": round(float(x2 - x1), 2),
                    "h": round(float(y2 - y1), 2),
                    "score": round(float(score), 4),
                    "label": clean_label,
                }
            )

        return detections

