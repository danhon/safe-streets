"""Synthetic headlight frames and calibration 2, shared by the night tests."""

import numpy as np

from safe_streets.lights import Blob

CAL2_ROAD = [(870, 457), (1920, 465), (1920, 615), (870, 600)]
CAL2_H = np.array([[0.011061946868896487, 0, -9.95575218200684],
                   [0, 0.13513513565063487, -72.29729757308965], [0, 0, 1]])
M_PER_PX = 0.011061946868896487


def _path(x_at, t1, fps=20.0, y=470.0):
    """Lights at x_at(t) (one x or a list) in frames from 0 to t1 at fps,
    each kept while it's in the frame."""
    out = []
    for i in range(int(t1 * fps) + 1):
        t = round(i / fps, 3)
        xs = [x for x in np.atleast_1d(x_at(t)) if 0 <= x <= 1920]
        out.append((t, [Blob(float(x), y, 200, int(x) - 10, int(x) + 10) for x in xs]))
    return out
