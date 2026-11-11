"""Find moments when something moves on the road, from a low-res export of
the continuous recording.

Used by the backfill (backfill.py) for stretches when Protect's motion events
missed cars: the motion zone didn't cover the whole road until the afternoon
of 2026-11-02. It exports the Low channel (640×360) in chunks and runs
background subtraction inside the road outline.
Measured: 5 minutes of footage is 14 MB, exports in 1.4 s, and scans in about
7 s of CPU.
"""

from pathlib import Path

import cv2
import numpy as np

LOW_W, LOW_H = 640, 360
FULL_W = 1920


def merge_windows(wins, gap: float, max_len: float):
    out: list[tuple[float, float]] = []
    for a, b in sorted(wins):
        if out and a - out[-1][1] <= gap and b - out[-1][0] <= max_len:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def find_windows(clip: Path, road_full, *, warmup_s: float = 10.0, thresh: float = 0.04, min_on_s: float = 0.3,
                 pad_s: float = 4.0, merge_gap_s: float = 2.0, max_len_s: float = 60.0,
                 mask: list | None = None) -> list[tuple[float, float]]:
    masked = mask
    """Seconds (from the clip's start) of padded windows with road motion.
    mask: polygons (1080p px) never counted as motion.
    Motion before warmup_s is ignored: the background model is still learning,
    and the previous chunk's overlap already covered that time."""
    cap = cv2.VideoCapture(str(clip))
    ok, first = cap.read()
    if not ok:
        return []
    scale = first.shape[1] / FULL_W
    road = (np.asarray(road_full, np.float32) * scale)
    h, w = first.shape[:2]
    x0, y0 = max(0, int(road[:, 0].min())), max(0, int(road[:, 1].min()))
    x1, y1 = min(w, int(road[:, 0].max()) + 1), min(h, int(road[:, 1].max()) + 1)
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(mask, [(road - [x0, y0]).astype(np.int32)], 255)
    for poly in masked or []:  # never motion (an on-screen clock)
        cv2.fillPoly(mask, [(np.asarray(poly, np.float32) * scale - [x0, y0]).astype(np.int32)], 0)
    area = max(1, int((mask > 0).sum()))
    mog = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)
    mog.apply(first[y0:y1, x0:x1])

    raw: list[tuple[float, float]] = []
    on = None
    last_t = 0.0
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        i += 1
        if i % 2:  # every other frame is plenty for something crossing in ≥1 s
            continue
        ok, f = cap.retrieve()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        last_t = t
        fg = mog.apply(f[y0:y1, x0:x1])
        moving = ((fg == 255) & (mask > 0)).sum() / area > thresh
        if moving and on is None:
            on = t
        elif not moving and on is not None:
            if t - on >= min_on_s and t >= warmup_s:
                raw.append((max(on, warmup_s), t))
            on = None
    cap.release()
    if on is not None and last_t - on >= min_on_s and last_t >= warmup_s:
        raw.append((max(on, warmup_s), last_t))
    padded = [(max(0.0, a - pad_s), min(last_t, b + pad_s)) for a, b in raw]
    return merge_windows(padded, merge_gap_s, max_len_s)
