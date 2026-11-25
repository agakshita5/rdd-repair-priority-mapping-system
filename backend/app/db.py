"""Database module: SQLite storage for locations, images and detections (FR6).

PostGIS is planned for the final phase; SQLite + haversine is used for the MVP.
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS locations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    lat           REAL    NOT NULL,
    lon           REAL    NOT NULL,
    total_score   REAL    NOT NULL DEFAULT 0,
    priority      TEXT    NOT NULL DEFAULT 'Low'
                  CHECK (priority IN ('High', 'Medium', 'Low')),
    status        TEXT    NOT NULL DEFAULT 'Reported'
                  CHECK (status IN ('Reported', 'Scheduled', 'Repaired')),
    n_detections  INTEGER NOT NULL DEFAULT 0,
    n_images      INTEGER NOT NULL DEFAULT 0,
    simulated     INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS images (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    location_id        INTEGER REFERENCES locations(id),
    filename           TEXT    NOT NULL,
    annotated_filename TEXT,
    lat                REAL    NOT NULL,
    lon                REAL    NOT NULL,
    gps_source         TEXT    NOT NULL,   -- exif | manual
    source             TEXT    NOT NULL,   -- upload | mapillary | simulated
    width              INTEGER,
    height             INTEGER,
    inference_ms       REAL,
    created_at         TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS detections (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id     INTEGER NOT NULL REFERENCES images(id),
    location_id  INTEGER NOT NULL REFERENCES locations(id),
    damage_type  TEXT    NOT NULL,
    confidence   REAL    NOT NULL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    score        REAL    NOT NULL,
    created_at   TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_locations_latlon ON locations(lat, lon);
CREATE INDEX IF NOT EXISTS idx_images_location ON images(location_id);
CREATE INDEX IF NOT EXISTS idx_detections_location ON detections(location_id);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_conn():
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.executescript(SCHEMA)


_LOCATION_SELECT = """
SELECT l.*,
       (SELECT GROUP_CONCAT(DISTINCT d.damage_type) FROM detections d
         WHERE d.location_id = l.id) AS damage_types,
       (SELECT MAX(d.confidence) FROM detections d
         WHERE d.location_id = l.id) AS max_confidence,
       (SELECT i.annotated_filename FROM images i
         WHERE i.location_id = l.id ORDER BY i.id DESC LIMIT 1) AS latest_annotated
FROM locations l
"""


def get_location(conn: sqlite3.Connection, location_id: int) -> sqlite3.Row | None:
    return conn.execute(_LOCATION_SELECT + " WHERE l.id = ?", (location_id,)).fetchone()


def list_locations(
    conn: sqlite3.Connection,
    damage_type: str | None = None,
    priority: str | None = None,
    status: str | None = None,
    limit: int = 500,
) -> list[sqlite3.Row]:
    sql, params = _LOCATION_SELECT + " WHERE 1 = 1", []
    if priority:
        sql += " AND l.priority = ?"
        params.append(priority)
    if status:
        sql += " AND l.status = ?"
        params.append(status)
    if damage_type:
        sql += " AND EXISTS (SELECT 1 FROM detections d WHERE d.location_id = l.id AND d.damage_type = ?)"
        params.append(damage_type)
    sql += " ORDER BY l.total_score DESC, l.id ASC LIMIT ?"
    params.append(limit)
    return conn.execute(sql, params).fetchall()


def location_detections(conn: sqlite3.Connection, location_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM detections WHERE location_id = ? ORDER BY score DESC", (location_id,)
    ).fetchall()


def location_images(conn: sqlite3.Connection, location_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM images WHERE location_id = ? ORDER BY id DESC", (location_id,)
    ).fetchall()


def update_status(conn: sqlite3.Connection, location_id: int, status: str) -> bool:
    cur = conn.execute(
        "UPDATE locations SET status = ?, updated_at = ? WHERE id = ?", (status, now(), location_id)
    )
    return cur.rowcount > 0
