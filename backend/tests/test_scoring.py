import pytest

from app import config
from app.scoring import detection_score, priority_level


def test_formula_matches_definition():
    # W(pothole)=1.0, box covers 10% of image, confidence 0.5 -> 0.05
    assert detection_score("D40", 100, 1000, 0.5) == pytest.approx(0.05)


def test_pothole_outranks_crack_of_same_size():
    pothole = detection_score("D40", 100, 1000, 0.8)
    alligator = detection_score("D20", 100, 1000, 0.8)
    transverse = detection_score("D10", 100, 1000, 0.8)
    longitudinal = detection_score("D00", 100, 1000, 0.8)
    assert pothole > alligator > transverse > longitudinal


def test_lower_confidence_lowers_score():
    assert detection_score("D40", 100, 1000, 0.9) > detection_score("D40", 100, 1000, 0.3)


def test_inputs_are_clamped():
    assert detection_score("D40", 5000, 1000, 1.5) == pytest.approx(1.0)


def test_unknown_class_rejected():
    with pytest.raises(ValueError):
        detection_score("D99", 100, 1000, 0.5)


def test_zero_image_area_rejected():
    with pytest.raises(ValueError):
        detection_score("D40", 100, 0, 0.5)


@pytest.mark.parametrize(
    "score,expected",
    [
        (config.PRIORITY_HIGH, "High"),
        (config.PRIORITY_HIGH + 0.5, "High"),
        (config.PRIORITY_MEDIUM, "Medium"),
        (config.PRIORITY_MEDIUM - 0.001, "Low"),
        (0.0, "Low"),
    ],
)
def test_priority_levels(score, expected):
    assert priority_level(score) == expected
