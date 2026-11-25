# Backend: Road Damage Detection API

FastAPI + SQLite. Takes a road image and its location, runs YOLOv8, scores each
detection, groups nearby detections into locations, and serves a ranked repair list.

## Run

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
uvicorn app.main:app --reload            # http://127.0.0.1:8000/docs
pytest -q                                # run tests
```

**Model:** put the trained weights at `weights/best.pt` (shared over Google Drive, not git).
If the file is missing, the API runs in **mock mode**: fake but deterministic detections,
so the frontend can be built before the model is ready. `GET /health` shows the mode.
To skip the large `ultralytics` install while in mock mode, install everything in
`requirements.txt` except it.

## Folder layout

```
app/
  main.py       routes (HTTP only)
  config.py     settings: class weights, cluster radius, priority cut-offs
  gps.py        EXIF GPS reader                       (FR2)
  detector.py   YOLOv8 inference + annotated image    (FR3)
  scoring.py    S = W_type x A_box/A_image x C        (FR4)
  cluster.py    haversine, 15 m grouping              (FR5)
  pipeline.py   detect -> score -> cluster -> store
  db.py         SQLite schema + queries               (FR6)
  schemas.py    request/response models (API contract)
scripts/
  fetch_mapillary.py  real geotagged Indian street images -> /detect
  seed_simulated.py   RDD2022 India test images at simulated city locations
tests/          pytest: scoring, GPS, clustering, API
weights/        best.pt goes here (git-ignored)
storage/        rdd.db + uploaded/annotated images (git-ignored)
```

## API contract (for the frontend)

Base URL `http://127.0.0.1:8000`. CORS allows `http://localhost:5173` (Vite).
Image URLs in responses are relative: prefix them with the base URL.

### `POST /detect` (multipart form)
| field | required | notes |
|---|---|---|
| `file` | yes | JPEG or PNG, at most 10 MB |
| `lat`, `lon` | only if the image has no EXIF GPS | from the map pin-drop |
| `source` | no | `upload` (default) \| `mapillary` \| `simulated` |

Errors: `422` with `location_required` → ask the user to drop a pin; `415` → not a JPEG/PNG; `413` → file too large.

```json
{
  "image_id": 7, "lat": 19.076, "lon": 72.8777,
  "gps_source": "exif", "source": "upload",
  "image_width": 1280, "image_height": 720,
  "detections": [
    {"damage_type": "D40", "label": "Pothole", "confidence": 0.81,
     "box": [412.0, 380.5, 640.2, 510.0], "score": 0.02615}
  ],
  "image_score": 0.02615,
  "annotated_url": "/files/annotated/<id>_annotated.jpg",
  "inference_ms": 142.3, "model_mode": "yolo",
  "location": { ...same shape as GET /locations item... }
}
```
`location` is `null` if no damage was found.

### `GET /locations?damage_type=D40&priority=High&status=Reported&limit=500`
Ranked repair list, highest score first. All filters are optional.
```json
[{
  "id": 3, "rank": 1, "lat": 19.0761, "lon": 72.8778,
  "total_score": 0.1342, "priority": "High", "status": "Reported",
  "n_detections": 4, "n_images": 2, "damage_types": ["D20", "D40"],
  "simulated": false,
  "image_url": "/files/annotated/<id>_annotated.jpg",
  "updated_at": "2026-11-22T06:28:47+00:00"
}]
```
Marker colours: High = red, Medium = orange, Low = yellow. Show a "Simulated location" badge when `simulated` is true.

### `GET /locations/{id}`
Same as above, plus `detections[]` and `images[]` (`image_url`, `annotated_url`, `source`, `gps_source`, `created_at`).

### `PATCH /locations/{id}/status` (JSON)
```json
{"status": "Scheduled"}
```
`status` is `Reported` | `Scheduled` | `Repaired`. Returns the updated location. A repaired location is not reused; new damage at the same spot creates a new location.

### `GET /health`
```json
{"status": "ok", "model_mode": "mock"}
```

## Demo data

```bash
# Real geotagged Indian street images (free token: mapillary.com -> Developers)
export MAPILLARY_TOKEN='MLY|xxxx'
python scripts/fetch_mapillary.py --per-city 40

# Backup: RDD2022 India test images at simulated city locations
python scripts/seed_simulated.py path/to/RDD2022/India/test/images --n 80
```

## Scoring

| Class | Code | Weight |
|---|---|---|
| Pothole | D40 | 1.0 |
| Alligator crack | D20 | 0.8 |
| Transverse crack | D10 | 0.5 |
| Longitudinal crack | D00 | 0.4 |

Detection score `S = W × (box area / image area) × confidence`. A location's score is the sum of its
detections' scores (detections within 15 m are grouped). Priority: High ≥ 0.10, Medium ≥ 0.03, else Low.
These cut-offs are initial values in `app/config.py` and should be tuned on real data.

## MVP simplifications (final phase)
- SQLite + haversine instead of PostGIS
- No login on status changes yet (NFR3)
- No face or number-plate blurring yet (NFR5)
