import pytest

from app.cluster import find_nearest, haversine_m

M_PER_DEG_LAT = 111_195  # metres per degree of latitude


def test_haversine_one_degree_latitude():
    assert haversine_m(0, 0, 1, 0) == pytest.approx(M_PER_DEG_LAT, rel=1e-3)


def test_haversine_zero_distance():
    assert haversine_m(19.07, 72.87, 19.07, 72.87) == 0


def test_point_within_radius_is_matched():
    candidates = [(1, 12.9716, 77.5946)]
    lat = 12.9716 + 5 / M_PER_DEG_LAT  # 5 m north
    assert find_nearest(lat, 77.5946, candidates, 15) == 1


def test_point_outside_radius_is_not_matched():
    candidates = [(1, 12.9716, 77.5946)]
    lat = 12.9716 + 50 / M_PER_DEG_LAT  # 50 m north
    assert find_nearest(lat, 77.5946, candidates, 15) is None


def test_nearest_candidate_wins():
    base = 12.9716
    candidates = [(1, base + 12 / M_PER_DEG_LAT, 77.5946), (2, base + 3 / M_PER_DEG_LAT, 77.5946)]
    assert find_nearest(base, 77.5946, candidates, 15) == 2
