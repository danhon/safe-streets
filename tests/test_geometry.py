import numpy as np
import pytest

from safe_streets.geometry import fit_homography, image_shift, in_polygon, to_world

# A plausible image->world homography (perspective, metres).
H_TRUE = np.array([[0.02, 0.004, -12.0], [0.0005, 0.06, -25.0], [0.00001, 0.0009, 1.0]])


def world_to_image(X, Y):
    p = np.linalg.inv(H_TRUE) @ np.array([X, Y, 1.0])
    return p[0] / p[2], p[1] / p[2]


def test_homography_roundtrip():
    world = [(0, 0), (10, 0), (20, 0), (0, 6), (10, 6), (20, 6)]
    image = [world_to_image(*w) for w in world]
    H, err = fit_homography(image, world)
    assert err < 1e-3
    x, y = world_to_image(7.5, 3.0)
    X, Y = to_world(H, x, y)
    assert abs(X - 7.5) < 1e-3 and abs(Y - 3.0) < 1e-3


def test_fewer_than_4_points_raises():
    with pytest.raises(ValueError):
        fit_homography([(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (0, 1)])


def test_collinear_points_raise():
    with pytest.raises(ValueError):
        fit_homography([(0, 0), (1, 0), (2, 0), (3, 0)], [(0, 0), (1, 0), (2, 0), (3, 0)])


def test_in_polygon():
    poly = [(0, 0), (10, 0), (10, 5), (0, 5)]
    assert in_polygon(poly, 5, 2)
    assert not in_polygon(poly, 11, 2)


def test_image_shift_detects_translation():
    rng = np.random.default_rng(0)
    base = (rng.random((360, 640)) * 255).astype(np.uint8)
    import cv2

    base = cv2.GaussianBlur(base, (3, 3), 0)
    shifted = np.roll(base, 12, axis=1)
    assert image_shift(base, base) < 1
    s = image_shift(base, shifted)
    assert s is not None and 11 <= s <= 13


def test_image_shift_ignores_changed_regions():
    """Moved shadows and passing cars change parts of the frame; only a
    shift the unchanged parts agree on counts (2026-11-02: a median of all
    matches read 9.8 px on a still camera in late-afternoon light)."""
    import cv2

    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur((rng.random((360, 640)) * 255).astype(np.uint8), (3, 3), 0)
    other = cv2.GaussianBlur((rng.random((360, 640)) * 255).astype(np.uint8), (3, 3), 0)
    cur = base.copy()
    cur[:, 100:] = other[:, 100:]  # 85% of the frame is different
    s = image_shift(base, cur)
    assert s is not None and s < 1.5
    assert 11 <= image_shift(base, np.roll(base, 12, axis=1)) <= 13
