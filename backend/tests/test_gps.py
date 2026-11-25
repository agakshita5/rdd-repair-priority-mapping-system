import io

import pytest

from app.gps import read_gps
from tests.conftest import make_image


def test_reads_exif_gps():
    lat, lon = read_gps(io.BytesIO(make_image(12.9716, 77.5946)))
    assert lat == pytest.approx(12.9716, abs=1e-5)
    assert lon == pytest.approx(77.5946, abs=1e-5)


def test_southern_western_hemisphere_is_negative():
    lat, lon = read_gps(io.BytesIO(make_image(-33.8688, -70.6693)))
    assert lat == pytest.approx(-33.8688, abs=1e-5)
    assert lon == pytest.approx(-70.6693, abs=1e-5)


def test_jpeg_without_gps_returns_none():
    assert read_gps(io.BytesIO(make_image())) is None


def test_png_returns_none():
    assert read_gps(io.BytesIO(make_image(fmt="PNG"))) is None


def test_non_image_returns_none():
    assert read_gps(io.BytesIO(b"not an image")) is None
