"""Image ↔ road-plane geometry, and the has-the-camera-moved check."""

import json
from dataclasses import dataclass, field, replace

import cv2
import numpy as np


# The working frame: every clip is decoded to this size, and a calibration's
# points (stored in its snapshot's pixels) are scaled into it.
WORK = (1920, 1080)

Pts = list[tuple[float, float]]


@dataclass(frozen=True)
class Calibration:
    id: int
    image_pts: Pts
    world_pts: Pts  # metres; X along the road, Y across
    H: np.ndarray  # 3x3, image -> world
    road: Pts  # where to look: the image polygon of the road
    reference_path: str
    status: str  # provisional | active | stale
    created_at: str = ""
    lane: Pts | None = None  # the in-scope lane: a pass's wheels must be inside it (default: the road)
    trusted: Pts | None = None  # where the scale was checked; a pass outside it is rough
    night_lane: tuple[float, float] | None = None  # image y of the near lane's far edge, left and right
    occluders: list[float] = field(default_factory=list)  # image x of things that split a light's track
    mask: list[Pts] = field(default_factory=list)  # polygons never counted as motion (an on-screen clock)
    size: tuple[int, int] = WORK  # the snapshot's, which the points are in

    def __post_init__(self):
        if self.lane is None:
            object.__setattr__(self, "lane", list(self.road))


def to_working(cal: Calibration) -> Calibration:
    """The calibration in the working frame's pixels (WORK)."""
    sx, sy = WORK[0] / cal.size[0], WORK[1] / cal.size[1]
    if (sx, sy) == (1.0, 1.0):
        return cal

    def pts(ps):
        return None if ps is None else [(x * sx, y * sy) for x, y in ps]

    S = np.diag([sx, sy, 1.0])
    return replace(cal, image_pts=pts(cal.image_pts), H=cal.H @ np.linalg.inv(S), road=pts(cal.road),
                   lane=pts(cal.lane), trusted=pts(cal.trusted),
                   night_lane=None if cal.night_lane is None else (cal.night_lane[0] * sy, cal.night_lane[1] * sy),
                   occluders=[x * sx for x in cal.occluders], mask=[pts(m) for m in cal.mask], size=WORK)


def fit_homography(image_pts, world_pts) -> tuple[np.ndarray, float]:
    """Least-squares homography from ≥4 ground points. Returns (H, mean
    re-projection error in metres)."""
    if len(image_pts) < 4 or len(image_pts) != len(world_pts):
        raise ValueError("need at least 4 matching points")
    src = np.asarray(image_pts, dtype=np.float64)
    dst = np.asarray(world_pts, dtype=np.float64)
    for pts in (src, dst):
        centred = pts - pts.mean(axis=0)
        if np.linalg.matrix_rank(centred, tol=1e-6) < 2:
            raise ValueError("points are collinear")
    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise ValueError("homography fit failed")
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    return H, float(np.linalg.norm(proj - dst, axis=1).mean())


def to_world(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
    p = H @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def to_image(H: np.ndarray, X: float, Y: float) -> tuple[float, float]:
    p = np.linalg.inv(H) @ np.array([X, Y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def in_polygon(poly, x: float, y: float) -> bool:
    return cv2.pointPolygonTest(np.asarray(poly, dtype=np.float32), (float(x), float(y)), False) >= 0


def image_shift(ref_gray: np.ndarray, cur_gray: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    """How far, in px, the camera's view has moved: the translation of a
    RANSAC fit (shift, rotation, scale) to ORB keypoints matched between the
    reference and current frames. None if too few matches agree to say.

    RANSAC, not the median of all matches: moved shadows, cars and a
    different time of day give many wrong matches, and on 2026-11-02 their
    median read 9.8 px on a camera that hadn't moved (RANSAC: 0.6 px)."""
    orb = cv2.ORB_create(nfeatures=2000)
    k1, d1 = orb.detectAndCompute(ref_gray, mask)
    k2, d2 = orb.detectAndCompute(cur_gray, mask)
    if d1 is None or d2 is None:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    if len(matches) < 20:
        return None
    src = np.float32([k1[m.queryIdx].pt for m in matches])
    dst = np.float32([k2[m.trainIdx].pt for m in matches])
    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None or inliers is None or int(inliers.sum()) < 20:
        return None
    return float(np.hypot(M[0, 2], M[1, 2]))


def calibration_from_row(row) -> Calibration:
    """A stored calibration, in the working frame's pixels."""
    keys = row.keys()

    def opt(name):
        v = row[name] if name in keys else None
        return json.loads(v) if v else None

    def tuples(ps):
        return None if ps is None else [tuple(p) for p in ps]

    nl, size = opt("night_lane"), opt("size")
    return to_working(Calibration(
        id=row["id"],
        image_pts=tuples(json.loads(row["image_pts"])),
        world_pts=tuples(json.loads(row["world_pts"])),
        H=np.asarray(json.loads(row["H"]), dtype=np.float64),
        road=tuples(json.loads(row["road"])),
        reference_path=row["reference_path"],
        status=row["status"],
        created_at=row["created_at"],
        lane=tuples(opt("lane")),
        trusted=tuples(opt("trusted")),
        night_lane=tuple(nl) if nl else None,
        occluders=opt("occluders") or [],
        mask=[tuples(m) for m in (opt("mask") or [])],
        size=tuple(size) if size else WORK,
    ))
