"""A synthetic street scene with a vehicle at a known speed: clips for tests
and benches with no footage in them .

The road is a flat plane seen side-on in perspective from a known
homography. A boxy vehicle (or, at night, a pair of saturated headlights)
moves along the near lane at a constant speed, and each frame is written with
exactly the presentation time it's given, so tests can check the pipeline's
speed against the truth at uneven frame rates.

  python -m safe_streets.synth demo OUT_DIR    (a segment to run the app on, 20 mph)

The sprite is a drawing, not a photograph: a detector may not call it a car.
Tests that need detection use the scene's true boxes (`Scene.box`).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

TIME_BASE = Fraction(1, 90000)
BASE = (1920, 1080)
# The road in the 1920x1080 frame: world (X along the road, Y across, metres)
# to image px. Near kerb (Y=0) low in the frame, far kerb (Y=7) higher and
# narrower, so the view is side-on with some perspective.
WORLD = np.float32([[0, 0], [24, 0], [24, 7], [0, 7]])
IMAGE = np.float32([[60, 650], [1860, 650], [1600, 470], [320, 470]])
LANE_Y = 1.8  # the near lane's wheel line, metres from the near kerb
CAR_LEN, HEIGHT_RATIO = 4.5, 0.36  # metres; image height over image length
START_X, END_X = -6.0, 30.0  # the car's centre runs between these


@dataclass(frozen=True)
class Scene:
    """What a rendered clip shows. Image coordinates are the clip's pixels."""

    H: np.ndarray  # image px -> road metres (X along, Y across)
    road: list[tuple[float, float]]
    lane_line: tuple[float, float]  # night: image y at x=0 and at the right edge
    truth_mps: float
    direction: int
    size: tuple[int, int]
    t0: float  # when the car's centre is at START_X (or END_X going left)
    _Hinv: np.ndarray = field(repr=False, default=None)

    def centre_x(self, t: float) -> float:
        d = self.truth_mps * (t - self.t0)
        return START_X + d if self.direction > 0 else END_X - d

    def _img(self, X: float, Y: float) -> tuple[float, float]:
        p = self._Hinv @ np.array([X, Y, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def wheel_point(self, t: float) -> tuple[float, float]:
        """The car's true ground point (its centre on the lane), image px."""
        return self._img(self.centre_x(t), LANE_Y)

    def box(self, t: float) -> tuple[float, float, float, float]:
        """The car's true box (x0, y0, x1, y1), image px, unclipped."""
        c = self.centre_x(t)
        xa, ya = self._img(c - CAR_LEN / 2, LANE_Y)
        xb, yb = self._img(c + CAR_LEN / 2, LANE_Y)
        x0, x1 = min(xa, xb), max(xa, xb)
        y1 = (ya + yb) / 2
        return x0, y1 - HEIGHT_RATIO * (x1 - x0), x1, y1

    def in_view(self, t: float) -> bool:
        x0, _, x1, _ = self.box(t)
        return x0 >= 0 and x1 <= self.size[0]


def scene_for(size: tuple[int, int], speed_mps: float, direction: int, t0: float) -> Scene:
    s = np.float32([size[0] / BASE[0], size[1] / BASE[1]])
    img = IMAGE * s
    H = cv2.getPerspectiveTransform(img, WORLD).astype(np.float64)
    road = [tuple(map(float, p)) for p in img]
    lane = (float(470 * s[1] + 0.55 * 180 * s[1]),) * 2
    return Scene(H=H, road=road, lane_line=lane, truth_mps=speed_mps, direction=direction,
                 size=size, t0=t0, _Hinv=np.linalg.inv(H))


def uneven_times(n: int, mean_fps: float, jitter: float, seed: int, start: float = 0.0) -> list[float]:
    """n strictly increasing frame times, gaps of 1/mean_fps scaled by a
    factor in [1 - jitter, 1 + jitter], each on the 1/90000 s clock."""
    rng = np.random.default_rng(seed)
    gaps = (1 / mean_fps) * rng.uniform(1 - jitter, 1 + jitter, n)
    ticks = np.round((start + np.concatenate([[0.0], np.cumsum(gaps[:-1])])) * 90000).astype(np.int64)
    for i in range(1, n):
        ticks[i] = max(ticks[i], ticks[i - 1] + 1)
    return [int(k) / 90000 for k in ticks]


def _background(sc: Scene, night: bool, seed: int) -> np.ndarray:
    w, h = sc.size
    rng = np.random.default_rng(seed)
    if night:
        img = np.full((h, w), 18, np.uint8)
        cv2.fillPoly(img, [np.int32(sc.road)], 34)
        return img
    img = np.full((h, w, 3), (150, 165, 140), np.uint8)  # verge, BGR
    cv2.fillPoly(img, [np.int32(sc.road)], (88, 88, 92))
    near = np.int32([sc.road[0], sc.road[1]])
    far = np.int32([sc.road[3], sc.road[2]])
    cv2.polylines(img, [near, far], False, (200, 200, 200), max(2, w // 640))
    for _ in range(40):  # static texture above the road, for the camera-moved check
        x, y = int(rng.uniform(0, w)), int(rng.uniform(0, sc.road[3][1] * 0.95))
        a, b = int(rng.uniform(w / 80, w / 20)), int(rng.uniform(h / 60, h / 15))
        cv2.rectangle(img, (x, y), (x + a, y + b), tuple(int(c) for c in rng.integers(40, 220, 3)), -1)
    return img


def _draw_photo(frame: np.ndarray, sc: Scene, t: float) -> None:
    """A side-view car a real detector takes for one (RTMDet-tiny: car at
    ~0.7): a shaded body, cabin, glass, lights, rimmed wheels and a shadow,
    filling the true box."""
    bx0, by0, bx1, by1 = sc.box(t)
    L, Hh = bx1 - bx0, by1 - by0
    colour = (60, 40, 150)

    def P(fx, fy):  # fx 0 rear .. 1 front, fy 0 ground .. 1 roof
        fx = fx if sc.direction > 0 else 1 - fx
        return (int(round(bx0 + fx * L)), int(round(by1 - fy * Hh)))

    cv2.ellipse(frame, (int(bx0 + L / 2), int(by1)), (int(L * 0.52), max(1, int(Hh * 0.10))), 0, 0, 360,
                (40, 40, 42), -1)
    body = np.int32([P(0.0, 0.18), P(0.01, 0.50), P(0.06, 0.56), P(0.22, 0.60), P(0.30, 0.95), P(0.36, 1.0),
                     P(0.66, 1.0), P(0.78, 0.66), P(0.95, 0.58), P(1.0, 0.45), P(1.0, 0.20), P(0.97, 0.12),
                     P(0.03, 0.12)])
    cv2.fillPoly(frame, [body], colour, cv2.LINE_AA)
    low = np.int32([P(0.0, 0.12), P(1.0, 0.12), P(1.0, 0.32), P(0.0, 0.32)])
    shade = frame.copy()
    cv2.fillPoly(shade, [low], tuple(int(c * 0.6) for c in colour), cv2.LINE_AA)
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillPoly(mask, [body], 255)
    frame[mask > 0] = (0.5 * frame[mask > 0] + 0.5 * shade[mask > 0]).astype(np.uint8)
    glass = (70, 60, 50)
    cv2.fillPoly(frame, [np.int32([P(0.25, 0.62), P(0.32, 0.92), P(0.48, 0.94), P(0.48, 0.62)])], glass, cv2.LINE_AA)
    cv2.fillPoly(frame, [np.int32([P(0.51, 0.62), P(0.51, 0.94), P(0.64, 0.94), P(0.75, 0.64)])], glass, cv2.LINE_AA)
    cv2.fillPoly(frame, [np.int32([P(0.96, 0.45), P(1.0, 0.45), P(1.0, 0.52), P(0.95, 0.54)])], (200, 230, 240),
                 cv2.LINE_AA)
    cv2.fillPoly(frame, [np.int32([P(0.0, 0.42), P(0.03, 0.42), P(0.03, 0.50), P(0.0, 0.50)])], (30, 30, 200),
                 cv2.LINE_AA)
    r = max(2, int(Hh * 0.17))
    for fx in (0.18, 0.80):
        cx = P(fx, 0.0)[0]
        c = (cx, int(round(by1)) - r)
        cv2.circle(frame, c, r, (25, 25, 25), -1, cv2.LINE_AA)
        cv2.circle(frame, c, max(1, int(r * 0.6)), (170, 170, 175), -1, cv2.LINE_AA)
        cv2.circle(frame, c, max(1, int(r * 0.2)), (90, 90, 95), -1, cv2.LINE_AA)


def _draw(frame: np.ndarray, sc: Scene, t: float, night: bool, photo: bool = False) -> None:
    if photo and not night:
        return _draw_photo(frame, sc, t)
    x0, y0, x1, y1 = (int(round(v)) for v in sc.box(t))
    L = x1 - x0
    if night:
        front = x1 if sc.direction > 0 else x0
        r = max(3, L // 30)
        for dy in (int(L * 0.10), int(L * 0.06)):
            cv2.circle(frame, (front - sc.direction * r, y1 - dy), r, 255, -1)
        return
    cv2.rectangle(frame, (x0, y0 + (y1 - y0) // 3), (x1, y1 - (y1 - y0) // 8), (60, 40, 150), -1)
    cv2.rectangle(frame, (x0 + L // 5, y0), (x1 - L // 5, y0 + (y1 - y0) // 3), (60, 40, 150), -1)
    cv2.rectangle(frame, (x0 + L // 4, y0 + (y1 - y0) // 12), (x1 - L // 4, y0 + (y1 - y0) // 3),
                  (210, 200, 170), -1)
    r = max(3, (y1 - y0) // 5)
    for cx in (x0 + L // 5, x1 - L // 5):
        cv2.circle(frame, (cx, y1 - r // 2), r, (20, 20, 20), -1)


def render_clip(path: Path, *, speed_mps: float, frame_times: list[float], direction: int = 1,
                size: tuple[int, int] = BASE, night: bool = False, bframes: bool = False,
                seed: int = 0, gop: int | None = None, photo: bool = False) -> Scene:
    """Write an H.264 MP4 at `path` whose frames carry exactly `frame_times`
    (seconds, on the 1/90000 s clock). The car's centre crosses the frame
    over the clip's middle. Night clips are grey (R = G = B), full range.
    photo: draw a car a real detector recognises (the opt-in real-model
    tests) instead of the flat one."""
    mid = (frame_times[0] + frame_times[-1]) / 2
    t0 = mid - (12.0 - START_X) / speed_mps  # centre at X=12 (mid-road) at mid
    sc = scene_for(size, speed_mps, direction, t0)
    bg = _background(sc, night, seed)
    w, h = size
    with av.open(str(path), "w", options={"video_track_timescale": "90000"}) as out:
        st = out.add_stream("libx264", rate=25)
        st.width, st.height = w, h
        st.time_base = TIME_BASE
        st.codec_context.time_base = TIME_BASE
        st.pix_fmt = "yuvj420p" if night else "yuv420p"
        st.options = {"preset": "ultrafast", "crf": "16", "bf": "2" if bframes else "0",
                      "x264-params": ("bframes=2:b-pyramid=0" if bframes else "bframes=0")
                      + (f":keyint={gop}:min-keyint={gop}:scenecut=0" if gop else "")}
        if night:
            st.codec_context.color_range = 2  # JPEG / full range
        for t in frame_times:
            img = bg.copy()
            _draw(img, sc, t, night, photo)
            if night:
                vf = av.VideoFrame.from_ndarray(img, format="gray").reformat(format="yuvj420p")
            else:
                vf = av.VideoFrame.from_ndarray(img, format="bgr24").reformat(format="yuv420p")
            vf.pts = int(round(t * 90000))
            vf.time_base = TIME_BASE
            for pkt in st.encode(vf):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)
    return sc


def read_pts(path: Path) -> list[float]:
    """Frame times as the pipeline reads them (OpenCV, CAP_PROP_POS_MSEC)."""
    cap = cv2.VideoCapture(str(path))
    out = []
    try:
        while cap.grab():
            out.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    finally:
        cap.release()
    return out


def demo(out: Path, now=None) -> Path:
    """A folder the app can run on: one 20 s segment, started 5 minutes
    before `now`, with a car crossing at 20 mph drawn so a real detector sees
    it, named by its start in TZ (UTC if unset) as SOURCE_PATTERN's default,
    and scene.json: the homography (H, image px to road metres) and the road
    outline to calibrate with."""
    import json
    import os
    from datetime import datetime, timedelta, timezone
    from zoneinfo import ZoneInfo

    now = now or datetime.now(timezone.utc)
    start = (now - timedelta(minutes=5)).astimezone(ZoneInfo(os.environ.get("TZ") or "UTC"))
    out.mkdir(parents=True, exist_ok=True)
    seg = out / start.strftime("%Y%m%d-%H%M%S.mp4")
    sc = render_clip(seg, speed_mps=8.9408, frame_times=uneven_times(300, 15, 0.3, seed=1), photo=True)
    (out / "scene.json").write_text(json.dumps({"H": sc.H.tolist(), "road": [list(p) for p in sc.road],
                                                "truth_mph": sc.truth_mps / 0.44704}))
    os.utime(seg, (now.timestamp() - 60, now.timestamp() - 60))
    return seg


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] != "demo":
        sys.exit(__doc__)
    print(demo(Path(sys.argv[2])))


if __name__ == "__main__":
    main()
