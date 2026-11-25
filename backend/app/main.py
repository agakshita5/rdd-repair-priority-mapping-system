"""FastAPI app: routes only. Logic lives in pipeline / scoring / cluster / gps / db."""
import io
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from . import config, db, gps, pipeline, schemas
from .detector import Detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    config.ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    db.init_db()
    app.state.detector = Detector()
    yield


app = FastAPI(title="Road Damage Detection API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"]
)
config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=config.STORAGE_DIR), name="files")


def _url(folder: str, name: str | None) -> str | None:
    return f"/files/{folder}/{name}" if name else None


def _location_out(row, rank: int | None = None) -> schemas.LocationOut:
    return schemas.LocationOut(
        id=row["id"],
        rank=rank,
        lat=row["lat"],
        lon=row["lon"],
        total_score=round(row["total_score"], 5),
        priority=row["priority"],
        status=row["status"],
        n_detections=row["n_detections"],
        n_images=row["n_images"],
        damage_types=sorted(row["damage_types"].split(",")) if row["damage_types"] else [],
        max_confidence=round(row["max_confidence"], 4) if row["max_confidence"] is not None else None,
        simulated=bool(row["simulated"]),
        image_url=_url("annotated", row["latest_annotated"]),
        updated_at=row["updated_at"],
    )


def _detection_out(damage_type: str, confidence: float, box, score: float) -> schemas.DetectionOut:
    return schemas.DetectionOut(
        damage_type=damage_type,
        label=config.CLASS_LABELS[damage_type],
        confidence=round(confidence, 4),
        box=[round(v, 1) for v in box],
        score=round(score, 5),
    )


@app.get("/health", response_model=schemas.Health)
def health():
    return {"status": "ok", "model_mode": app.state.detector.mode}


@app.post("/detect", response_model=schemas.DetectResponse)
async def detect(
    file: UploadFile = File(..., description="Road image (JPEG or PNG)"),
    lat: float | None = Form(None, description="Used only if the image has no EXIF GPS"),
    lon: float | None = Form(None),
    source: str = Form("upload", description="upload | mapillary | simulated"),
):
    if source not in config.SOURCES:
        raise HTTPException(422, f"source must be one of {config.SOURCES}")

    data = await file.read()
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Image larger than {config.MAX_UPLOAD_MB} MB")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "File is not a valid image")
    if fmt not in config.ALLOWED_FORMATS:
        raise HTTPException(415, "Only JPEG and PNG images are supported")

    # FR2: EXIF GPS first, then the location picked on the map
    coords = gps.read_gps(io.BytesIO(data))
    gps_source = "exif"
    if coords is None:
        if lat is None or lon is None:
            raise HTTPException(422, "location_required: image has no GPS data, send lat and lon")
        if not gps.is_valid_coord(lat, lon):
            raise HTTPException(422, "lat/lon out of range")
        coords, gps_source = (lat, lon), "manual"

    # Keep the original pixels: re-encoding a JPEG is lossy and changes the model's output.
    # Only images that need rotation (EXIF orientation) or are PNG get rewritten, losslessly as PNG.
    with Image.open(io.BytesIO(data)) as im:
        needs_rotation = im.getexif().get(0x0112, 1) != 1
        if fmt == "JPEG" and not needs_rotation:
            image_path = config.UPLOAD_DIR / f"{uuid.uuid4().hex}.jpg"
            image_path.write_bytes(data)
        else:
            image_path = config.UPLOAD_DIR / f"{uuid.uuid4().hex}.png"
            ImageOps.exif_transpose(im).convert("RGB").save(image_path, "PNG")

    with db.get_conn() as conn:
        result = pipeline.process_image(
            conn, app.state.detector, image_path, coords[0], coords[1], gps_source, source
        )
        location_row = db.get_location(conn, result["location_id"]) if result["location_id"] else None

    detections = [_detection_out(d.damage_type, d.confidence, d.box, s) for d, s in result["detections"]]
    return schemas.DetectResponse(
        image_id=result["image_id"],
        lat=coords[0],
        lon=coords[1],
        gps_source=gps_source,
        source=source,
        image_width=result["width"],
        image_height=result["height"],
        detections=detections,
        image_score=round(sum(d.score for d in detections), 5),
        annotated_url=_url("annotated", result["annotated_filename"]),
        inference_ms=round(result["inference_ms"], 1),
        model_mode=app.state.detector.mode,
        location=_location_out(location_row) if location_row else None,
    )


@app.get("/locations", response_model=list[schemas.LocationOut])
def get_locations(
    damage_type: schemas.DamageType | None = None,
    priority: schemas.Priority | None = None,
    status: schemas.Status | None = None,
    limit: int = Query(500, ge=1, le=5000),
):
    """Ranked repair list: highest total score first (FR8)."""
    with db.get_conn() as conn:
        rows = db.list_locations(conn, damage_type, priority, status, limit)
    return [_location_out(r, rank=i + 1) for i, r in enumerate(rows)]


@app.get("/locations/{location_id}", response_model=schemas.LocationDetail)
def get_location(location_id: int):
    with db.get_conn() as conn:
        row = db.get_location(conn, location_id)
        if row is None:
            raise HTTPException(404, "Location not found")
        dets = db.location_detections(conn, location_id)
        imgs = db.location_images(conn, location_id)
    return schemas.LocationDetail(
        **_location_out(row).model_dump(),
        detections=[
            _detection_out(d["damage_type"], d["confidence"], (d["x1"], d["y1"], d["x2"], d["y2"]), d["score"])
            for d in dets
        ],
        images=[
            schemas.ImageOut(
                id=i["id"],
                image_url=_url("uploads", i["filename"]),
                annotated_url=_url("annotated", i["annotated_filename"]),
                source=i["source"],
                gps_source=i["gps_source"],
                created_at=i["created_at"],
            )
            for i in imgs
        ],
    )


@app.patch("/locations/{location_id}/status", response_model=schemas.LocationOut)
def update_status(location_id: int, body: schemas.StatusUpdate):
    """FR9: Reported -> Scheduled -> Repaired. (Login required in final phase, NFR3.)"""
    with db.get_conn() as conn:
        if not db.update_status(conn, location_id, body.status):
            raise HTTPException(404, "Location not found")
        row = db.get_location(conn, location_id)
    return _location_out(row)
