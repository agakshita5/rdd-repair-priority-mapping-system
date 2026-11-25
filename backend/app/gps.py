"""GPS module: read latitude/longitude from image EXIF data"""
from typing import BinaryIO
from PIL import Image

GPS_IFD = 0x8825 # 34853
LAT_REF, LAT, LON_REF, LON = 1, 2, 3, 4


def _ref(value, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        value = value.decode(errors="ignore")
    return str(value).strip("\x00 ").upper() or default


def _to_degrees(dms, ref: str) -> float:
    d, m, s = (float(x) for x in dms)
    value = d + m / 60 + s / 3600
    return -value if ref in ("S", "W") else value


def is_valid_coord(lat: float, lon: float) -> bool:
    return -90 <= lat <= 90 and -180 <= lon <= 180


def read_gps(source: str | BinaryIO) -> tuple[float, float] | None:
    """Return (lat, lon) from EXIF, or None if the image has no usable GPS data."""
    try:
        with Image.open(source) as img:
            gps = img.getexif().get_ifd(GPS_IFD)
    except Exception:
        return None
    if not gps or LAT not in gps or LON not in gps:
        return None
    try:
        lat = _to_degrees(gps[LAT], _ref(gps.get(LAT_REF), "N"))
        lon = _to_degrees(gps[LON], _ref(gps.get(LON_REF), "E"))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if not is_valid_coord(lat, lon) or (lat == 0 and lon == 0):
        return None
    return lat, lon
