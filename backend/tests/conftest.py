import io
import os
import tempfile

# Must be set before the app is imported: isolated storage + fake detector
os.environ["RDD_STORAGE_DIR"] = tempfile.mkdtemp(prefix="rdd_test_")
os.environ["RDD_FORCE_MOCK"] = "1"

import piexif  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app import config  # noqa: E402
from app.main import app  # noqa: E402


def _to_dms(value: float):
    value = abs(value)
    d = int(value)
    m = int((value - d) * 60)
    s = round(((value - d) * 60 - m) * 60 * 10000)
    return ((d, 1), (m, 1), (s, 10000))


def make_image(lat: float | None = None, lon: float | None = None, fmt: str = "JPEG", seed: int = 0) -> bytes:
    """Create a small test image, optionally with EXIF GPS tags."""
    img = Image.new("RGB", (640, 480), (90 + seed % 50, 90, 90))
    buf = io.BytesIO()
    if lat is not None and fmt == "JPEG":
        gps = {
            piexif.GPSIFD.GPSLatitudeRef: b"N" if lat >= 0 else b"S",
            piexif.GPSIFD.GPSLatitude: _to_dms(lat),
            piexif.GPSIFD.GPSLongitudeRef: b"E" if lon >= 0 else b"W",
            piexif.GPSIFD.GPSLongitude: _to_dms(lon),
        }
        img.save(buf, fmt, exif=piexif.dump({"GPS": gps}))
    else:
        img.save(buf, fmt)
    return buf.getvalue()


@pytest.fixture
def client():
    if config.DB_PATH.exists():
        config.DB_PATH.unlink()  # fresh database for every test
    with TestClient(app) as c:
        yield c


def upload(client, data: bytes, **form):
    return client.post("/detect", files={"file": ("road.jpg", data, "image/jpeg")}, data=form)
