"""Clustering module: group detections within CLUSTER_RADIUS_M into one location."""
import math
from typing import Iterable

EARTH_RADIUS_M = 6_371_000


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def bounding_box(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    """Cheap pre-filter box (min_lat, max_lat, min_lon, max_lon) around a point."""
    dlat = radius_m / 111_320
    dlon = radius_m / (111_320 * max(math.cos(math.radians(lat)), 1e-6))
    return lat - dlat, lat + dlat, lon - dlon, lon + dlon


def find_nearest(
    lat: float, lon: float, candidates: Iterable[tuple[int, float, float]], radius_m: float
) -> int | None:
    """Return the id of the nearest candidate (id, lat, lon) within radius_m, else None."""
    best_id, best_d = None, radius_m
    for cid, clat, clon in candidates:
        d = haversine_m(lat, lon, clat, clon)
        if d <= best_d:
            best_id, best_d = cid, d
    return best_id
