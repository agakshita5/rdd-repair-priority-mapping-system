"""Scoring module: repair-priority score.

S = W_type x (A_box / A_image) x C
"""
from . import config


def detection_score(damage_type: str, box_area: float, image_area: float, confidence: float) -> float:
    if damage_type not in config.DAMAGE_WEIGHTS:
        raise ValueError(f"Unknown damage type: {damage_type}")
    if image_area <= 0:
        raise ValueError("Image area must be positive")
    area_ratio = min(max(box_area / image_area, 0.0), 1.0)
    confidence = min(max(confidence, 0.0), 1.0)
    return config.DAMAGE_WEIGHTS[damage_type] * area_ratio * confidence


def priority_level(location_score: float) -> str:
    if location_score >= config.PRIORITY_HIGH:
        return "High"
    if location_score >= config.PRIORITY_MEDIUM:
        return "Medium"
    return "Low"
