"""Calibration. The tested way is drive-bys at known speeds: drive past at a
steady speed you know, twice or more, pick each drive's pass, enter its true
speed, and the along-road scale is fitted from them (scale_from_drives).

The other way, clicking measured ground points on a snapshot and typing their
positions in metres to fit a homography, is experimental: it has never been
tested on a real install."""

import re
from dataclasses import dataclass

import numpy as np

from .speed import MPS_TO_MPH, Sample, fit_speed

# The routes are the operator's (operator/views.py).

MAX_REPROJ_M = 0.5
SNAP_RE = re.compile(r"^snap-\d+\.jpg$")


@dataclass(frozen=True)
class DriveFit:
    px_per_m: float  # along the road, at the drives' wheel line
    readings_mph: list[float]  # each drive at that scale
    spread_mph: float  # the largest |reading - true speed|

    @property
    def H(self) -> np.ndarray:
        """A scale-only image -> road map: px / px_per_m."""
        return np.diag([1 / self.px_per_m, 1 / self.px_per_m, 1.0])


def _px_per_s(samples: list[Sample]) -> float:
    """Theil-Sen slope of wheel x against time, px/s (absolute)."""
    ts, xs = np.array([s.t for s in samples]), np.array([s.x for s in samples])
    i, j = np.triu_indices(len(ts), 1)
    dt = ts[j] - ts[i]
    ok = dt > 0
    return float(abs(np.median((xs[j] - xs[i])[ok] / dt[ok])))


def scale_from_drives(drives: list[tuple[list[Sample], float]]) -> DriveFit:
    """The along-road scale from drive-bys: each (cleaned track, true speed
    in m/s). The scale is the mean of each drive's own (its px/s over its
    true speed). Needs two drives or more: one can't show its own error."""
    if len(drives) < 2:
        raise ValueError("a calibration needs two drive-bys or more")
    own = [_px_per_s(s) / v for s, v in drives]
    px_per_m = float(np.mean(own))
    H = np.diag([1 / px_per_m, 1 / px_per_m, 1.0])
    readings = [fit_speed(s, H).mph for s, _ in drives]
    spread = max(abs(r - v * MPS_TO_MPH) for r, (_, v) in zip(readings, drives))
    return DriveFit(px_per_m, readings, spread)
