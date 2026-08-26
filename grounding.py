import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection
)


class Grounder:

    def __init__(
        self,
        model_id="IDEA-Research/grounding-dino-tiny",
        device=None,
        box_threshold=0.35,
        text_threshold=0.25
    ):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        self.processor = AutoProcessor.from_pretrained(
            model_id
        )

        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(self.device)

        self.model.eval()

    def ground(self, image, query):

        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            image = Image.fromarray(image).convert("RGB")

        inputs = self.processor(
            images=image,
            text=[[query]],
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]]
        )

        result = results[0]

        detections = []

        for box, score, label in zip(
            result["boxes"],
            result["scores"],
            result["text_labels"]
        ):

            x1, y1, x2, y2 = box.tolist()

            detections.append({
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "score": float(score),
                "label": label
            })

        return detections