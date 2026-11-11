"""The vendored ByteTrack against the trackers it must match: the original
(ifzhang/ByteTrack, MIT) for the Kalman step, and the tracker speedcam ran
for the box overlap (the Mac's frame-by-frame replay, 2026-11-10: with both,
0 of 11,462 frames differ, boxes within 0.0001 px)."""

import numpy as np
import pytest

from safe_streets.bytetrack import KalmanFilter, ious


def test_iou_counts_box_edges_as_coordinates_not_pixels():
    """No +1 per side (cython_bbox's): at RTMDet's 0.25 the low-score boxes
    sit near the 0.8 fused cost, where that +1 changed 243 frames' matches."""
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[5, 0, 15, 10]], np.float32)
    assert ious(a, b)[0, 0] == pytest.approx(50 / 150, abs=1e-6)
    assert ious(a, a)[0, 0] == pytest.approx(1.0, abs=1e-6)
    assert ious(a, np.array([[20, 20, 30, 30]]))[0, 0] == 0.0


def test_multi_predict_takes_the_noise_from_the_height_before_the_step():
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([100.0, 200.0, 0.5, 80.0]))
    mean[7] = 40.0  # the height grows 40 px a frame
    m, c = kf.multi_predict(mean[None], cov[None])
    F = kf._motion_mat
    assert np.allclose(m[0], F @ mean)
    assert np.allclose(c[0], F @ cov @ F.T + kf._motion_cov(80.0))
