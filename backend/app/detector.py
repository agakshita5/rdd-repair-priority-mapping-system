"""Detection module: YOLOv8 inference + annotated image output.

If no weights file is found (or RDD_FORCE_MOCK=1) it runs in *mock* mode and
returns deterministic fake detections, so the API and frontend can be built
before the trained model is ready. /health reports which mode is active.
"""
import hashlib
import random
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from . import config

# Map whatever class names the weights use to RDD2022 codes
CLASS_ALIASES = {
    "d00": "D00", "longitudinal crack": "D00",
    "d10": "D10", "transverse crack": "D10",
    "d20": "D20", "alligator crack": "D20",
    "d40": "D40", "pothole": "D40", "potholes": "D40",
}

BOX_COLORS = {"D00": "#2f80ed", "D10": "#f2c94c", "D20": "#f2994a", "D40": "#eb5757"}


def normalize_class(name: str) -> str | None:
    key = name.strip().lower().replace("_", " ").replace("-", " ")
    return CLASS_ALIASES.get(key)


@dataclass
class Detection:
    damage_type: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(x2 - x1, 0) * max(y2 - y1, 0)


class Detector:
    def __init__(self, weights_path: Path = config.WEIGHTS_PATH, force_mock: bool = config.FORCE_MOCK):
        self.model = None
        self.mode = "mock"
        if not force_mock and Path(weights_path).exists():
            from ultralytics import YOLO  # imported lazily: heavy dependency

            self.model = YOLO(str(weights_path))
            self.mode = "yolo"
            # Warm-up: the first prediction is slow (model setup), keep it out of user requests
            self.model.predict(Image.new("RGB", (640, 640)), verbose=False)

    def predict(self, image_path: Path) -> tuple[list[Detection], int, int, float]:
        """Return (detections, image_width, image_height, inference_ms)."""
        with Image.open(image_path) as im:
            width, height = im.size
        start = time.perf_counter()
        if self.model is not None:
            detections = self._predict_yolo(image_path)
        else:
            detections = self._predict_mock(image_path, width, height)
        inference_ms = (time.perf_counter() - start) * 1000
        return detections, width, height, inference_ms

    def _predict_yolo(self, image_path: Path) -> list[Detection]:
        result = self.model.predict(source=str(image_path), conf=config.CONF_THRESHOLD, verbose=False)[0]
        detections = []
        boxes = result.boxes
        for xyxy, conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
            code = normalize_class(self.model.names[int(cls)])
            if code is None:
                continue
            detections.append(Detection(code, float(conf), tuple(float(v) for v in xyxy)))
        return detections

    @staticmethod
    def _predict_mock(image_path: Path, width: int, height: int) -> list[Detection]:
        seed = hashlib.md5(Path(image_path).read_bytes()).hexdigest()
        rng = random.Random(seed)
        detections = []
        for _ in range(rng.randint(1, 3)):
            bw = width * rng.uniform(0.08, 0.3)
            bh = height * rng.uniform(0.05, 0.25)
            x1 = rng.uniform(0, width - bw)
            y1 = rng.uniform(height * 0.4, height - bh)  # damage sits in the lower half (road)
            code = rng.choice(list(config.DAMAGE_WEIGHTS))
            detections.append(Detection(code, round(rng.uniform(0.4, 0.95), 3), (x1, y1, x1 + bw, y1 + bh)))
        return detections


def annotate(image_path: Path, detections: list[Detection], out_path: Path) -> None:
    """Save a copy of the image with labelled boxes drawn on it."""
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        draw = ImageDraw.Draw(im)
        line = max(2, im.width // 300)
        for d in detections:
            color = BOX_COLORS[d.damage_type]
            draw.rectangle(d.box, outline=color, width=line)
            label = f"{d.damage_type} {d.confidence:.2f}"
            x1, y1 = d.box[0], max(d.box[1] - 14, 0)
            tw = draw.textlength(label)
            draw.rectangle((x1, y1, x1 + tw + 6, y1 + 14), fill=color)
            draw.text((x1 + 3, y1 + 1), label, fill="black")
        im.save(out_path, "JPEG", quality=90)
