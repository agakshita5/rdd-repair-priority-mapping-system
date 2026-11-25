"""API request/response models."""
from typing import Literal

from pydantic import BaseModel

Priority = Literal["High", "Medium", "Low"]
Status = Literal["Reported", "Scheduled", "Repaired"]
DamageType = Literal["D00", "D10", "D20", "D40"]


class DetectionOut(BaseModel):
    damage_type: DamageType
    label: str
    confidence: float
    box: list[float]  # [x1, y1, x2, y2] pixels
    score: float


class LocationOut(BaseModel):
    id: int
    rank: int | None = None
    lat: float
    lon: float
    total_score: float
    priority: Priority
    status: Status
    n_detections: int
    n_images: int
    damage_types: list[DamageType]
    max_confidence: float | None
    simulated: bool
    image_url: str | None
    updated_at: str


class DetectResponse(BaseModel):
    image_id: int
    lat: float
    lon: float
    gps_source: Literal["exif", "manual"]
    source: str
    image_width: int
    image_height: int
    detections: list[DetectionOut]
    image_score: float
    annotated_url: str
    inference_ms: float
    model_mode: Literal["yolo", "mock"]
    location: LocationOut | None


class ImageOut(BaseModel):
    id: int
    image_url: str
    annotated_url: str | None
    source: str
    gps_source: str
    created_at: str


class LocationDetail(LocationOut):
    detections: list[DetectionOut]
    images: list[ImageOut]


class StatusUpdate(BaseModel):
    status: Status


class Health(BaseModel):
    status: str
    model_mode: Literal["yolo", "mock"]
