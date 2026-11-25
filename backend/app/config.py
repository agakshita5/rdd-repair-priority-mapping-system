"""Central settings. Override any value with an environment variable."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Storage (SQLite DB + uploaded / annotated images)
STORAGE_DIR = Path(os.getenv("RDD_STORAGE_DIR", BASE_DIR / "storage"))
DB_PATH = STORAGE_DIR / "rdd.db"
UPLOAD_DIR = STORAGE_DIR / "uploads"
ANNOTATED_DIR = STORAGE_DIR / "annotated"

# Model
WEIGHTS_PATH = Path(os.getenv("RDD_WEIGHTS", BASE_DIR / "weights" / "best.pt"))
FORCE_MOCK = os.getenv("RDD_FORCE_MOCK", "0") == "1"  # fake detections, no model needed
CONF_THRESHOLD = float(os.getenv("RDD_CONF", "0.25"))

# Upload limits
MAX_UPLOAD_MB = 10
ALLOWED_FORMATS = {"JPEG", "PNG"}

# Damage classes (RDD2022 codes) and repair-priority weights
CLASS_LABELS = {
    "D00": "Longitudinal crack",
    "D10": "Transverse crack",
    "D20": "Alligator crack",
    "D40": "Pothole",
}
DAMAGE_WEIGHTS = {"D00": 0.4, "D10": 0.5, "D20": 0.8, "D40": 1.0}

# Clustering: detections within this radius belong to the same location
CLUSTER_RADIUS_M = 15.0

# Location-score cut-offs for priority levels (initial values, tune on real data)
PRIORITY_HIGH = 0.10
PRIORITY_MEDIUM = 0.03

STATUSES = ("Reported", "Scheduled", "Repaired")
SOURCES = ("upload", "mapillary", "simulated")

CORS_ORIGINS = os.getenv(
    "RDD_CORS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")
