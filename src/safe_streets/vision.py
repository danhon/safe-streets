"""The detector (detector.py; RTMDet-tiny by default) on the road strip: a cheap
2 fps gate, then ByteTrack on every frame, keeping each frame's detection box.

Timestamps are each frame's pts (CAP_PROP_POS_MSEC), never an assumed fps:
the porch clips vary from 8.7 to 20 fps (measured 2026-11-02).
"""

import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .geometry import in_polygon
from .speed import Sample

log = logging.getLogger(__name__)

# COCO's bicycle (1) by day too (plan 2026-11-09-two-wheelers): without it a
# bike was seen only when YOLO took it for a motorcycle (pass 2384, a cargo
# bike). The night tracker (lights.py) is untouched.
VEHICLES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
TWO_WHEEL_IDS = (1, 3)
FOUR_WHEEL_IDS = (2, 5, 7)
EDGE_PX = 4
GATE_FPS = 2.0
# Tracking runs inference only from SPAN_PAD_S before the gate first saw a
# moving vehicle to SPAN_PAD_S after it last did. The gate samples every
# 0.5 s, and a car is partly in view for about a second before YOLO finds
# it. Per-frame tracking costs ~65 ms of CPU around the GPU inference
# (measured 2026-11-02), and a car is in view for 1–3 s of a ~28 s clip.
# It's track()'s default; the pipeline passes TRACK_PAD_BEFORE_S and
# TRACK_PAD_AFTER_S (config.py), which default to it.
SPAN_PAD_S = 2.5
STATIC_PX = 15  # a detection this close to one in most frames is parked
# The motion pre-gate (Vision.motion_gate; GPU-impact spike, bench.py): a
# sample is worth YOLO only if more than MOTION_THR grey levels changed since
# the frame before across MOTION_RUN_PX columns (1080p px) of the road
# outline, looking at a row and column every MOTION_STRIDE 1080p px: every
# 4th at full size, every one in a 640×360 export (at every 4th, its car
# showed 36 px of motion, not ~200, and the gate started a second late).
MOTION_THR, MOTION_RUN_PX, MOTION_STRIDE = 25, 60, 4
# The motion gate takes the headlight tracker's blobs from its own decode
# (motion_gate(lights=True); plan step 5) only where the luma plane is
# exactly what light_frames sees: a full-range 1920×1080 clip (grey is then
# luma, 1.000·Y; finding 7) every frame of which is grey, every chroma
# sample exactly neutral, as on 40 of 40 infrared clips measured. A colour
# frame's BGR clips at a headlight, so its grey isn't the luma there.
# Calibration is in High-channel pixels. Some exports come back 640×360
# (seen 2026-11-02); those frames are scaled up so the geometry still holds.
FRAME_W, FRAME_H = 1920, 1080
WIDE_PX = 400



def _iou(a: np.ndarray, b: np.ndarray) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _inside(a: np.ndarray, b: np.ndarray) -> float:
    """The share of box a's area inside box b."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return w * h / area if area > 0 else 0.0


def two_wheeler_keep(xyxy: np.ndarray, cls: np.ndarray, conf: np.ndarray, *, iou: float = 0.5,
                     inside: float = 0.8) -> np.ndarray:
    """Which of one frame's boxes to track. YOLO's NMS is per class, so one
    rider can come back as a bicycle and a motorcycle: of such a pair (IoU at
    least `iou`) only the more confident is kept. A bicycle box at least
    `inside` within a car, bus or truck box is a bike on its rack: dropped.
    Motorcycles aren't racked, so the rack rule is bicycles only."""
    n = len(cls)
    keep = np.ones(n, dtype=bool)
    ids = cls.astype(int)
    for i in range(n):
        if ids[i] == 1 and any(ids[j] in FOUR_WHEEL_IDS and _inside(xyxy[i], xyxy[j]) >= inside
                               for j in range(n)):
            keep[i] = False
    for i in range(n):
        for j in range(i + 1, n):
            if {ids[i], ids[j]} == set(TWO_WHEEL_IDS) and keep[i] and keep[j] and _iou(xyxy[i], xyxy[j]) >= iou:
                keep[j if conf[i] >= conf[j] else i] = False
    return keep

@dataclass
class TrackResult:
    tracks: dict[int, list[Sample]]
    fps: float
    frame_at: Callable[[float], np.ndarray]
    classes: dict[int, str]
    low_res: bool
    # Seconds: the last frame's pts plus one frame. Protect starts an export
    # at the keyframe before the requested start and ends it at the requested
    # end, so the clip really began at end - duration (measured 2026-11-02).
    duration: float | None = None
    # How many times YOLO ran (both crops' calls are summed by the pipeline).
    inferences: int = 0


@dataclass
class Gate:
    """The gate's answer: when a vehicle was moving (first and last sampled
    t), or None; and every frame's pts, so track() needn't decode the whole
    clip again just to learn the clip's fps and duration. moving: each
    sampled t with a moving vehicle (span is the first and last);
    inferences: how many times YOLO ran."""
    span: tuple[float, float] | None
    times: list[float]
    moving: list[float] = field(default_factory=list)
    inferences: int = 0
    # The headlight tracker's (t, blobs) per frame, when the motion gate was
    # asked for them and could take them from its own decode; else None.
    lights: list | None = None


def native_box(crop, shape) -> tuple[tuple[int, int, int, int], float, float]:
    """The 1080p crop box in this frame's own pixels, and the x, y scale.
    Saves upscaling every 640×360 frame to 1920×1080 just to look at it."""
    h, w = shape[:2]
    sx, sy = w / FRAME_W, h / FRAME_H
    x0, y0, x1, y1 = crop
    return (int(x0 * sx), int(y0 * sy), int(round(x1 * sx)), int(round(y1 * sy))), sx, sy


def duration_from(times: list[float]) -> float | None:
    """The clip's length: the last pts plus the median frame gap."""
    if not times:
        return None
    gaps = np.diff(times)
    return times[-1] + (float(np.median(gaps)) if len(gaps) else 0.0)


def fps_from(times: list[float]) -> float:
    return (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 0.0


def crop_for(road) -> tuple[int, int, int, int]:
    """The road polygon's bounding box, extended upward by 1.5× its height
    (car bodies stand above their wheels) and downward by 0.3× (a near-curb
    car's box bottom reaches below its wheels' line), clipped to the frame.
    The x range stays the outline's: YOLO's input is a fixed 640 px, so a
    wider crop gives coarser boxes and slower speeds (a 15 mph drive-by
    read 14.6 on this crop and 13.2 on the wide one, 2026-11-02)."""
    xs = [p[0] for p in road]
    ys = [p[1] for p in road]
    x0, x1, y0, y1 = int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys))
    h = y1 - y0
    return x0, max(0, int(y0 - 1.5 * h)), x1, min(FRAME_H, int(y1 + 0.3 * h))


def wide_crop_for(road) -> tuple[int, int, int, int]:
    """crop_for plus a car length (WIDE_PX) each side. A near car is ~500 px
    long, and with only part of it in the crop YOLO often misses it, above
    all at night, when it's blurred and lit by its own headlights. Used for
    the gate and as the fallback track; speeds from it read ~10% slow."""
    x0, y0, x1, y1 = crop_for(road)
    return max(0, x0 - WIDE_PX), y0, min(FRAME_W, x1 + WIDE_PX), y1


def keep_sample(box, crop) -> bool:
    """False for a box cut off by the crop's left or right edge: its
    bottom-centre would be dragged and fake a speed."""
    return box[0] > crop[0] + EDGE_PX and box[2] < crop[2] - EDGE_PX


def split_track(samples: list[Sample], max_jump_px: float = 150, max_width_change: float = 0.25):
    """ByteTrack sometimes hands a parked car's ID to a car emerging from
    behind it (seen on the 11:21:14 calibration pass). Split a track wherever
    the box jumps or changes width sharply from one sample to the next."""
    parts: list[list[Sample]] = []
    for s in samples:
        if parts:
            p = parts[-1][-1]
            pw, w = p.box[2] - p.box[0], s.box[2] - s.box[0]
            jump = abs((s.box[0] + s.box[2]) / 2 - (p.box[0] + p.box[2]) / 2)
            if jump > max_jump_px or abs(w - pw) > max_width_change * max(pw, 1):
                parts.append([s])
                continue
            parts[-1].append(s)
        else:
            parts.append([s])
    return parts


def clean_track(samples: list[Sample], crop) -> list[Sample]:
    """Positions for the one car this track is really following.

    1. Keep the run of steady movement with the largest displacement.
       ByteTrack can keep one ID across a car parking, sitting parked, and a
       different car driving past (the 11:21:14 calibration pass); the passing
       car is the run that goes furthest.
    2. Follow that car's leading edge (its front, in the direction it moves),
       half a car-width back, so positions read as its centre. The leading edge
       is visible from when a car enters until it reaches the far end, and a
       parked car hiding the back doesn't move it. On the 15 mph drive-bys the
       front edges gave 604 and 608 px/s, 0.7% apart; centres of partly hidden
       boxes were off by up to 40%. Samples whose leading edge is at the crop's
       edge are dropped."""
    run = _furthest_moving_run(samples, lambda s: (s.box[0] + s.box[2]) / 2)
    if len(run) < 3:
        return []
    direction = 1 if (run[-1].box[0] + run[-1].box[2]) >= (run[0].box[0] + run[0].box[2]) else -1
    full = [s.box[2] - s.box[0] for s in run if keep_sample(s.box, crop)]
    widths = full if full else [s.box[2] - s.box[0] for s in run]
    half = float(np.percentile(widths, 80)) / 2
    out = []
    for s in run:
        x0, _, x1, _ = s.box
        if direction > 0 and x1 < crop[2] - EDGE_PX:
            out.append(Sample(t=s.t, x=x1 - half, y=s.y, box=s.box, conf=s.conf, cls=s.cls))
        elif direction < 0 and x0 > crop[0] + EDGE_PX:
            out.append(Sample(t=s.t, x=x0 + half, y=s.y, box=s.box, conf=s.conf, cls=s.cls))
    out = _furthest_moving_run(out, lambda s: s.x)
    return out if len(out) >= 3 else []


def _furthest_moving_run(samples: list[Sample], pos, min_px_per_s: float = 100.0,
                         max_px_per_s: float = 3000.0) -> list[Sample]:
    """The run of consecutive samples that keeps moving (each step between
    min_px_per_s, about 2.5 mph, and max_px_per_s, beyond which it's an ID
    jump) and covers the most ground."""
    if len(samples) < 2:
        return samples
    runs, start = [], 0
    for i in range(1, len(samples) + 1):
        ok = False
        if i < len(samples):
            dt = samples[i].t - samples[i - 1].t
            v = abs(pos(samples[i]) - pos(samples[i - 1])) / dt if dt > 0 else 0.0
            ok = min_px_per_s <= v <= max_px_per_s
        if not ok:
            runs.append((start, i))
            start = i
    a, b = max(runs, key=lambda r: abs(pos(samples[r[1] - 1]) - pos(samples[r[0]])))
    return samples[a:b]


def moving_times(per_frame) -> list[float]:
    """per_frame: [(t, [(x, y) wheel points in the road])] from the gate's
    sampled frames. A point seen (within STATIC_PX) in most frames is parked;
    returns each t with a point that isn't."""
    if not per_frame:
        return []

    def seen_in(p):
        return sum(any(np.hypot(p[0] - q[0], p[1] - q[1]) < STATIC_PX for q in f) for _, f in per_frame)

    need = max(2, int(0.6 * len(per_frame)))
    return [t for t, f in per_frame if any(seen_in(p) < need for p in f)]


def episodes(moving: list[float], pad: tuple[float, float]) -> list[tuple[float, float]]:
    """EPISODES=on (GPU-impact spike): the union of [t - pad[0], t + pad[1]]
    over the gate's moving sample times, in order; touching ones merge."""
    out: list[list[float]] = []
    for t in sorted(set(moving)):
        a, b = t - pad[0], t + pad[1]
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def moving_span(per_frame) -> tuple[float, float] | None:
    """The first and last of moving_times, or None."""
    ts = moving_times(per_frame)
    return (min(ts), max(ts)) if ts else None


def _to_work(frame):
    """The frame at the working size, FRAME_W×FRAME_H (any other size is
    resized), and whether it was smaller (its passes are low_res)."""
    h, w = frame.shape[:2]
    if (w, h) == (FRAME_W, FRAME_H):
        return frame, False
    smaller = w < FRAME_W or h < FRAME_H
    interp = cv2.INTER_LINEAR if smaller else cv2.INTER_AREA
    return cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=interp), smaller


def _frames(clip: Path):
    """Yields (t_seconds, frame, scaled) using each frame's pts. Frames are
    always FRAME_W×FRAME_H; scaled is True when the clip was smaller."""
    cap = cv2.VideoCapture(str(clip))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame, scaled = _to_work(frame)
            yield t, frame, scaled
    finally:
        cap.release()


def _span_frames(clip: Path, lo: float, hi: float):
    """_frames, for lo ≤ t ≤ hi only: earlier frames are grab()bed (decoded,
    no colour conversion), and decoding stops after hi."""
    cap = cv2.VideoCapture(str(clip))
    try:
        while cap.grab():
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t > hi:
                break
            if t < lo:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            frame, scaled = _to_work(frame)
            yield t, frame, scaled
    finally:
        cap.release()


def frame_at_factory(clip: Path) -> Callable[[float], np.ndarray | None]:
    return lambda t: frame_at_seek(clip, t)


def frame_at_seek(clip: Path, t: float, back_s: float = 1.0) -> np.ndarray | None:
    """The screenshot frame: the first with pts ≥ t − 1 ms, else the last;
    always FRAME_W×FRAME_H. Seeks instead of decoding from the start. OpenCV
    turns a POS_MSEC seek into a frame number using the average fps, which
    is wrong for variable-rate clips, so it seeks back_s early, checks where
    it landed, and if that overshot t, decodes from the start instead. Only
    the chosen frame is retrieved. If t is past the last frame, that last
    frame is found again by its pts."""
    frame, last = _seek_grab(clip, t, back_s)
    if frame is None and last is not None:  # ran off the end: want the last frame
        frame, _ = _seek_grab(clip, last, back_s)
    return frame


def _seek_grab(clip: Path, t: float, back_s: float):
    """(the frame at t, or None if the clip ends first; the last pts seen)."""
    pos = cv2.CAP_PROP_POS_MSEC
    cap = cv2.VideoCapture(str(clip))
    last = None
    try:
        if t > back_s:
            cap.set(pos, (t - back_s) * 1000.0)
            if not cap.grab() or cap.get(pos) / 1000.0 >= t - 1e-3:
                # Overshot (or failed): the right frame may be behind us.
                cap.release()
                cap = cv2.VideoCapture(str(clip))
            else:
                last = cap.get(pos) / 1000.0
        while cap.grab():
            last = cap.get(pos) / 1000.0
            if last >= t - 1e-3:
                ok, frame = cap.retrieve()
                if ok and (frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H):
                    frame = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_LINEAR)
                return (frame if ok else None), last
        return None, last
    finally:
        cap.release()


def gpu_throttle(level: str) -> None:
    """Compile every GPU model with GPU_QUEUE_THROTTLE=level ("" leaves
    OpenVINO alone). Without it, the thread waiting on the HD 530 busy-spins:
    a 640x640 inference cost 45 ms of CPU for 46 ms of wall time, and 8 ms
    with LOW, for +3 ms of wall time and identical outputs (14 vs 50 ms
    through Ultralytics; bench/gpu_wait.py, 2026-11-02). Ultralytics builds
    its own ov.Core and config, so the hint is added in compile_model.
    Idempotent; an explicit GPU_QUEUE_THROTTLE in a config wins."""
    if not level:
        return
    import openvino as ov

    orig = ov.Core.compile_model
    if getattr(orig, "_speedcam_throttle", None) == level:
        return

    def compile_model(self, model, device_name=None, config=None, **kw):
        config = dict(config or {})
        if str(device_name).upper().startswith("GPU"):
            config.setdefault("GPU_QUEUE_THROTTLE", level)
        return orig(self, model, device_name, config, **kw)

    compile_model._speedcam_throttle = level
    compile_model._speedcam_cache = getattr(orig, "_speedcam_cache", None)
    ov.Core.compile_model = compile_model


def compile_cache(cache_dir) -> None:
    """Compile every model (GPU or CPU) with OpenVINO's CACHE_DIR, so a
    restart loads the compiled model from disk instead of compiling it again:
    the first event after a restart took ~24 s instead of ~2 (2026-11-07),
    and freezes 3 and 4 came at or soon after fresh GPU contexts (the
    GPU-impact spike; a speculative link). OpenVINO keys the cache by the
    model, device, config and driver, so a stale entry is never used.
    None or "" leaves compiling alone. Idempotent; an explicit CACHE_DIR in
    a config wins."""
    if not cache_dir:
        return
    import openvino as ov

    cache_dir = str(cache_dir)
    orig = ov.Core.compile_model
    if getattr(orig, "_speedcam_cache", None) == cache_dir:
        return
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    def compile_model(self, model, device_name=None, config=None, **kw):
        config = dict(config or {})
        config.setdefault("CACHE_DIR", cache_dir)
        return orig(self, model, device_name, config, **kw)

    compile_model._speedcam_cache = cache_dir
    compile_model._speedcam_throttle = getattr(orig, "_speedcam_throttle", None)
    ov.Core.compile_model = compile_model


def _grey_full_range(f) -> bool:
    """A PyAV frame whose luma plane is light_frames' grey, bit for bit:
    8-bit 4:2:0, full range, 1920×1080, and every chroma sample 128, so
    OpenCV's BGR is R = G = B = Y."""
    if (f.width, f.height) != (FRAME_W, FRAME_H) or f.format.name not in ("yuvj420p", "yuv420p"):
        return False
    if getattr(f, "color_range", None) != 2:  # AVCOL_RANGE_JPEG
        return False
    for plane in f.planes[1:3]:
        c = np.frombuffer(plane, np.uint8).reshape(-1, plane.line_size)[: f.height // 2, : f.width // 2]
        if not c.min() == c.max() == 128:
            return False
    return True


def new_tracker():
    """ByteTrack (bytetrack.py, vendored, MIT), driven so each sample keeps
    its frame's detection box (the tracker's idx column), not the Kalman
    state, which lags a car still coming into view and then catches up: a
    20 mph drive-by read 24.5 from the Kalman state, 21.9 from its
    detections."""
    from .bytetrack import BYTETracker, Params

    return BYTETracker(Params())


class Vision:
    def __init__(self, model_dir: Path, device: str, conf: float, throttle: str = "LOW",
                 cache_dir: Path | None = None, detector=None):
        gpu_throttle(throttle)
        compile_cache(cache_dir)
        self.cache_dir = cache_dir
        self._first: set[str] = set()
        self.model_dir = Path(model_dir)
        self.device = device
        self.conf = conf
        self._models: dict[str, object] = {}
        self._lock = threading.Lock()
        self._device_in_use: str | None = None
        self._detector = detector

    def detector(self):
        """The Detector (detector.py), loaded on first use: the default is
        RTMDet-tiny from MODELS_DIR."""
        with self._lock:
            if self._detector is None:
                from .detector import RTMDet

                self._detector = RTMDet(self.model_dir, self.device, self.conf, classes=VEHICLES)
            return self._detector

    def _infer(self, kind: str, img):
        t0 = time.monotonic()
        d = self.detector().detect(img)
        self._first_inference(kind, t0)
        keep = np.isin(d.cls.astype(int), list(VEHICLES)) & (d.conf >= self.conf)
        return d[keep]

    def device_in_use(self) -> str:
        """/healthz's yolo_device. Asked of OpenVINO once (2026-11-08): a fresh
        ov.Core() lists devices in ~0.4 s of CPU on the event loop, Docker's
        healthcheck asks every 30 s, and the devices can't change while the
        process runs. A failed query isn't kept."""
        if self._device_in_use is None:
            try:
                import openvino as ov

                devices = ov.Core().available_devices
            except Exception as e:  # noqa: BLE001
                return f"unknown ({e})"
            gone = self.device.endswith("gpu") and "GPU" not in devices
            self._device_in_use = "cpu (no GPU visible)" if gone else self.device
        return self._device_in_use

    def _first_inference(self, kind: str, t0: float) -> None:
        """Log each model's first inference after a start: it includes
        loading (or compiling) the model, which the cache should shorten."""
        if kind not in self._first:
            self._first.add(kind)
            log.info("detector %s first inference ms=%d device=%s cache=%s", kind,
                     int((time.monotonic() - t0) * 1000), self.device, self.cache_dir or "off")

    def _detect(self, img):
        return self._infer("gate", img).xyxy

    def gate(self, clip: Path, crop, road, gate_fps: float = GATE_FPS) -> Gate:
        """When a vehicle was moving in the road, sampling gate_fps frames a
        second, and every frame's pts. Every frame is grab()bed (for its pts)
        but only the sampled ones are retrieve()d, and YOLO sees the crop at
        the clip's own resolution; points are mapped back to 1080p."""
        cap = cv2.VideoCapture(str(clip))
        per_frame: list[tuple[float, list[tuple[float, float]]]] = []
        times: list[float] = []
        next_t = 0.0
        try:
            while cap.grab():
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                times.append(t)
                if t < next_t:
                    continue
                next_t = t + 1 / gate_fps
                ok, frame = cap.retrieve()
                if not ok:
                    break
                (nx0, ny0, nx1, ny1), sx, sy = native_box(crop, frame.shape)
                pts = []
                for b in self._detect(frame[ny0:ny1, nx0:nx1]):
                    cx, cy = ((b[0] + b[2]) / 2 + nx0) / sx, (b[3] + ny0) / sy
                    if in_polygon(road, cx, cy):
                        pts.append((cx, cy))
                per_frame.append((t, pts))
        finally:
            cap.release()
        ts = moving_times(per_frame)
        return Gate((min(ts), max(ts)) if ts else None, times, ts, len(per_frame))

    def motion_gate(self, clip: Path, crop, road, gate_fps: float = GATE_FPS, lights: bool = False) -> Gate:
        """gate(), with a motion pre-gate (MOTION_GATE; the GPU-impact
        spike's bench.py motion_gate, ported as it was measured): YOLO runs
        on a sample only if the road outline's luma changed since the frame
        before (MOTION_THR, MOTION_RUN_PX), and a detection is moving only
        if that change overlaps its box by MOTION_RUN_PX too, so a parked car
        isn't, whatever passes beside it. Decodes with PyAV on one thread,
        reading the luma plane, not colour, except on the samples YOLO sees;
        t is each frame's pts, bit for bit as OpenCV gives it. An empty or broken clip
        gives what it decoded so far, like gate(). With lights, it also finds
        the headlight tracker's blobs in each frame's luma (Gate.lights), so
        the night path needn't decode the clip again, unless one of its
        frames isn't exactly grey, the clip isn't full-range 1920×1080
        (_grey_full_range), or decoding fails part-way."""
        import av

        from .lights import BAND, BlobFinder

        try:
            container = av.open(str(clip))
        except (av.error.FFmpegError, OSError):
            return Gate(None, [])
        times: list[float] = []
        moving: list[float] = []
        inferences = 0
        night: list | None = [] if lights else None
        finder = BlobFinder()
        try:
            if not container.streams.video:
                return Gate(None, [])
            stream = container.streams.video[0]
            stream.thread_type = "NONE"
            # OpenCV's arithmetic (cap_ffmpeg_impl: the pts, or the dts when
            # it's 0 or unset, less the stream's start, × time_base, in ms):
            # night_cars turns on a time's last bit.
            start = stream.start_time or 0
            r2d = stream.time_base.numerator / stream.time_base.denominator
            prev = mask = None
            next_t = 0.0
            try:
                for f in container.decode(stream):
                    pts = f.pts if f.pts not in (None, 0) else f.dts
                    if pts is None:
                        continue
                    t = float(pts - start) * r2d * 1000.0 / 1000.0
                    times.append(t)
                    plane = f.planes[0]
                    y = np.frombuffer(plane, np.uint8).reshape(-1, plane.line_size)[: f.height, : f.width]
                    if mask is None:
                        sx, sy = f.width / FRAME_W, f.height / FRAME_H
                        d = max(1, round(MOTION_STRIDE * sx))
                        ys = [p[1] for p in road]
                        r0, r1 = max(0, int((min(ys) - 10) * sy)), int((max(ys) + 10) * sy)
                        m = np.zeros((f.height, f.width), np.uint8)
                        cv2.fillPoly(m, [np.array([(x * sx, yy * sy) for x, yy in road], np.int32)], 1)
                        mask = m[r0:r1:d, ::d].astype(bool)
                    if night is not None:
                        if not _grey_full_range(f):
                            night = None  # the night path decodes the clip as today
                        else:
                            night.append((t, finder(y[BAND[0]:BAND[1]], BAND[0])))
                    band = y[r0:r1:d, ::d]
                    if t >= next_t and prev is not None:
                        next_t = t + 1 / gate_fps
                        changed = (np.abs(band.astype(np.int16) - prev.astype(np.int16)) > MOTION_THR) & mask
                        cols = changed.sum(axis=0) >= 2
                        if cols.sum() * d / sx >= MOTION_RUN_PX:
                            frame = f.to_ndarray(format="bgr24")
                            (nx0, ny0, nx1, ny1), _, _ = native_box(crop, frame.shape)
                            inferences += 1
                            for b in self._detect(frame[ny0:ny1, nx0:nx1]):
                                cx, cy = ((b[0] + b[2]) / 2 + nx0) / sx, (b[3] + ny0) / sy
                                c0, c1 = int((b[0] + nx0) / d), int(np.ceil((b[2] + nx0) / d))
                                if in_polygon(road, cx, cy) and cols[c0:c1].sum() * d / sx >= MOTION_RUN_PX:
                                    moving.append(t)
                                    break
                    prev = band.copy()
            except av.error.FFmpegError:
                log.warning("motion gate: %s stopped decoding after %d frames", clip.name, len(times))
                night = None  # OpenCV may decode past it: the night path decodes as today
        finally:
            container.close()
        return Gate((min(moving), max(moving)) if moving else None, times, moving, inferences, night)

    def track(self, clip: Path, crop, road, span: tuple[float, float] | None = None,
              times: list[float] | None = None, pad: tuple[float, float] = (SPAN_PAD_S, SPAN_PAD_S),
              intervals: list[tuple[float, float]] | None = None):
        """Samples per track id (inside the road outline, not edge-truncated),
        the measured fps, a full-frame getter, each track's class, and
        whether the clip was low resolution. With span (from the gate), YOLO
        runs only on frames from pad[0] before it to pad[1] after it. With
        intervals (episodes(), in order), YOLO runs only inside them instead,
        a fresh tracker for each, whose ids are offset by 1000 a time
        (each tracker numbers from 1). With times (every frame's pts, from
        the gate), frames before the first are only grab()bed, decoding stops
        after the last, and fps and duration come from times."""
        x0, y0, x1, y1 = crop
        tracks: dict[int, list[Sample]] = {}
        votes: dict[int, Counter] = {}
        given = bool(times)
        times = list(times) if given else []
        low_res = False
        if intervals:
            windows = [tuple(iv) for iv in intervals]
        elif span:
            windows = [(span[0] - pad[0], span[1] + pad[1])]
        else:
            windows = [(float("-inf"), float("inf"))]
        lo, hi = windows[0][0], windows[-1][1]
        k, tracker = -1, None
        inferences = 0
        for t, frame, scaled in (_span_frames(clip, lo, hi) if given else _frames(clip)):
            if not given:
                times.append(t)
            low_res = low_res or scaled
            j = next((i for i, (a, b) in enumerate(windows) if a <= t <= b), None)
            if j is None:
                continue
            if j != k:
                k, tracker = j, new_tracker()
            det = self._infer("track", frame[y0:y1, x0:x1])
            inferences += 1
            det = det[two_wheeler_keep(det.xyxy, det.cls, det.conf)]
            out = tracker.update(det.xyxy, det.conf, det.cls)
            if len(out) == 0:
                continue
            idx = out[:, 7].astype(int)
            for (bx0, by0, bx1, by1), tid, cls, conf in zip(det.xyxy[idx], out[:, 4], det.cls[idx], det.conf[idx]):
                tid = int(tid) + 1000 * k
                box = (bx0 + x0, by0 + y0, bx1 + x0, by1 + y0)
                cx, cy = (box[0] + box[2]) / 2, box[3]
                name = VEHICLES.get(int(cls), "car")
                tracks.setdefault(int(tid), []).append(Sample(t=t, x=float(cx), y=float(cy),
                                                              box=tuple(float(v) for v in box),
                                                              conf=float(conf), cls=name))
                votes.setdefault(int(tid), Counter())[name] += 1
        fps = fps_from(times)
        cleaned: dict[int, list[Sample]] = {}
        for tid, s in tracks.items():
            for k, part in enumerate(split_track(s)):
                kept = [q for q in clean_track(part, crop) if in_polygon(road, q.x, q.y)]
                if kept:
                    cleaned[tid * 100 + k] = kept
                    votes[tid * 100 + k] = votes[tid]
        tracks = cleaned
        classes = {tid: votes[tid].most_common(1)[0][0] for tid in tracks}
        return TrackResult(tracks, fps, frame_at_factory(clip), classes, low_res, duration_from(times), inferences)

    def night_lights(self, clip: Path, frames=None):
        """(headlight blobs per frame, low_res) for an infrared clip, else
        None. With frames (the motion gate's, from its own decode of a
        full-size clip), only the infrared check reads the clip. The import
        is local because lights imports _frames from here."""
        from .lights import is_infrared, light_frames

        if not is_infrared(clip):
            return None
        return (frames, False) if frames is not None else light_frames(clip)
