"""Speed from a track: wheel positions on the road plane against pts time."""

from dataclasses import dataclass

import numpy as np

from .db import is_hidden  # noqa: F401  (re-exported: one rule for what's hidden)
from .geometry import in_polygon, to_world

MPS_TO_MPH = 2.23694


@dataclass(frozen=True)
class Sample:
    t: float  # seconds from clip start, from pts
    x: float  # image px: bottom-centre of the box
    y: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1, full-frame px
    conf: float | None = None  # YOLO's confidence in this frame's box (None at night)
    cls: str | None = None  # YOLO's class for this frame's box


@dataclass(frozen=True)
class SpeedFit:
    mph: float
    plus_minus: float
    metres: float
    seconds: float
    direction: int
    n: int
    sigma_m: float = 0.0  # robust sigma of the residuals, metres


def fit_speed(samples: list[Sample], H: np.ndarray) -> SpeedFit:
    t = np.array([s.t for s in samples])
    X = np.array([to_world(H, s.x, s.y)[0] for s in samples])
    n = len(t)
    # Theil–Sen: median of pairwise slopes, pairs at least 0.2 s apart.
    i, j = np.triu_indices(n, k=1)
    dt = t[j] - t[i]
    keep = dt >= 0.2
    if not keep.any():
        keep = dt > 0
    slope = float(np.median((X[j] - X[i])[keep] / dt[keep])) if keep.any() else 0.0
    intercept = float(np.median(X - slope * t))
    resid = X - (slope * t + intercept)
    sigma = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
    spread = float(np.sqrt(((t - t.mean()) ** 2).sum()))
    se = sigma / spread if spread > 0 else float("inf")
    return SpeedFit(
        mph=abs(slope) * MPS_TO_MPH,
        plus_minus=2 * se * MPS_TO_MPH,
        metres=float(X.max() - X.min()),
        seconds=float(t.max() - t.min()),
        direction=1 if slope >= 0 else -1,
        n=n,
        sigma_m=sigma,
    )


def gate_flags(fit: SpeedFit, samples: list[Sample], cal, settings, fps: float) -> list[str]:
    flags = []
    if fit.metres < settings.min_track_m or fit.seconds < settings.min_track_s:
        flags.append("short")
    if fit.plus_minus > settings.max_uncertainty_mph:
        flags.append("uncertain")
    if fit.mph < settings.parked_mph:
        flags.append("parked")
    xs = [p[0] for p in cal.road]
    lo, hi = min(xs), max(xs)
    # Positions are the leading edge minus half a car, so a car that reached
    # an end of the outline reads half a car-width short of it.
    half = float(np.median([s.box[2] - s.box[0] for s in samples])) / 2
    margin = 0.15 * (hi - lo) + half

    def near_end(x):
        return x <= lo + margin or x >= hi - margin

    if not (near_end(samples[0].x) and near_end(samples[-1].x)):
        flags.append("partial")
    if fps < settings.min_fps:
        flags.append("low_fps")
    # The scale was checked only in the trusted region (by default a band
    # around the calibration drives' wheel line). Elsewhere, nearer or further
    # from the camera, a metre spans a different number of pixels.
    if cal.trusted is not None:
        mx, my = float(np.median([s.x for s in samples])), float(np.median([s.y for s in samples]))
        if not in_polygon(cal.trusted, mx, my):
            flags.append("outside_trusted")
    return flags


def whole(mph: float) -> int:
    """The speed as the table rounds it."""
    return int(f"{mph:.0f}")


def band(mph: float | None, limit: float) -> str:
    if mph is None or mph <= limit:
        return ""
    over = mph - limit
    if over <= 5:
        return "over-0"
    if over <= 10:
        return "over-5"
    if over <= 15:
        return "over-10"
    return "over-15"


FLAG_TEXT = {
    "short": "short track", "uncertain": "wide ±", "partial": "only partly seen",
    "low_fps": "low frame rate", "low_res": "low-res clip",
    "uncalibrated": "not calibrated", "camera_moved": "camera moved",
    "coarse": "coarse detection: may read ~10% slow",
    "headlights": "night: tracked by its headlights",
    "outside_trusted": "outside where the calibration was checked",
}
# "partial" isn't here: parked cars and the tree hide part of the road on most
# passes, and the 14.5 mph calibration pass (true 15) was partial. It's kept on
# the pass, not shown as a caveat.
ROUGH = {"short", "uncertain", "low_fps", "low_res", "coarse", "outside_trusted", "headlights"}


def confidence(flags: list[str], plus_minus) -> tuple[str, str]:
    """good: nothing wrong and ± ≤ 2 mph; fair: nothing wrong, wider ±;
    rough: a caveat about the track or clip. Returns (class, label)."""
    rough = [FLAG_TEXT.get(f, f) for f in flags if f in ROUGH]
    if rough:
        return "rough", "rough estimate: " + ", ".join(rough)
    if plus_minus is not None and plus_minus <= 2:
        return "good", "good estimate"
    return "fair", "fair estimate"
