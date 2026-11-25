"""Pipeline: detect -> score -> cluster -> store, for one uploaded image."""
import sqlite3
from pathlib import Path

from . import cluster, config, db, scoring
from .detector import Detection, Detector, annotate


def _find_or_create_location(conn: sqlite3.Connection, lat: float, lon: float, simulated: bool) -> int:
    min_lat, max_lat, min_lon, max_lon = cluster.bounding_box(lat, lon, config.CLUSTER_RADIUS_M)
    # Repaired locations are history: new damage there starts a new location
    rows = conn.execute(
        "SELECT id, lat, lon FROM locations WHERE status != 'Repaired' "
        "AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?",
        (min_lat, max_lat, min_lon, max_lon),
    ).fetchall()
    match = cluster.find_nearest(lat, lon, ((r["id"], r["lat"], r["lon"]) for r in rows), config.CLUSTER_RADIUS_M)
    if match is not None:
        return match
    ts = db.now()
    cur = conn.execute(
        "INSERT INTO locations (lat, lon, simulated, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (lat, lon, int(simulated), ts, ts),
    )
    return cur.lastrowid


def _refresh_location(conn: sqlite3.Connection, location_id: int) -> None:
    """Recompute centroid, total score, counts and priority from stored rows."""
    agg = conn.execute(
        "SELECT COALESCE(SUM(score), 0) AS total, COUNT(*) AS n FROM detections WHERE location_id = ?",
        (location_id,),
    ).fetchone()
    pos = conn.execute(
        "SELECT AVG(lat) AS lat, AVG(lon) AS lon, COUNT(*) AS n FROM images WHERE location_id = ?",
        (location_id,),
    ).fetchone()
    conn.execute(
        "UPDATE locations SET lat = ?, lon = ?, total_score = ?, priority = ?, "
        "n_detections = ?, n_images = ?, updated_at = ? WHERE id = ?",
        (
            pos["lat"], pos["lon"], agg["total"], scoring.priority_level(agg["total"]),
            agg["n"], pos["n"], db.now(), location_id,
        ),
    )


def process_image(
    conn: sqlite3.Connection,
    detector: Detector,
    image_path: Path,
    lat: float,
    lon: float,
    gps_source: str,
    source: str,
) -> dict:
    detections, width, height, inference_ms = detector.predict(image_path)

    annotated_name = f"{image_path.stem}_annotated.jpg"
    annotate(image_path, detections, config.ANNOTATED_DIR / annotated_name)

    scored: list[tuple[Detection, float]] = [
        (d, scoring.detection_score(d.damage_type, d.area, width * height, d.confidence)) for d in detections
    ]

    location_id = None
    if scored:
        location_id = _find_or_create_location(conn, lat, lon, simulated=(source == "simulated"))

    ts = db.now()
    image_id = conn.execute(
        "INSERT INTO images (location_id, filename, annotated_filename, lat, lon, gps_source, source, "
        "width, height, inference_ms, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (location_id, image_path.name, annotated_name, lat, lon, gps_source, source,
         width, height, inference_ms, ts),
    ).lastrowid

    for d, score in scored:
        conn.execute(
            "INSERT INTO detections (image_id, location_id, damage_type, confidence, x1, y1, x2, y2, "
            "score, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, location_id, d.damage_type, d.confidence, *d.box, score, ts),
        )

    if location_id is not None:
        _refresh_location(conn, location_id)

    return {
        "image_id": image_id,
        "location_id": location_id,
        "width": width,
        "height": height,
        "inference_ms": inference_ms,
        "annotated_filename": annotated_name,
        "detections": scored,
    }
