"""Night cars by their headlights: saturated blobs per frame, by pts.

At night the camera switches to infrared greyscale and YOLO misses dark cars,
but a car's headlights still saturate the sensor. These are the per-frame
blobs, the tracks that link them, and the rules that judge a track a car.
"""

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from .geometry import to_world
from .speed import MPS_TO_MPH, Sample, SpeedFit, fit_speed
from .vision import _frames, _furthest_moving_run

log = logging.getLogger(__name__)

BAND = (380, 720)  # rows searched, full-res y: the road and its kerbs
SAT = 245  # grey threshold; headlights clip to white
MIN_AREA, MAX_AREA = 80, 20000  # blob area, full-res px
# Infrared frames are exactly grey (mean |B-G| + |G-R| over the first five
# frames measured 0.0 on the night fixture and a full night clip, n0908);
# daytime measures 21.9 (the day fixture) and 26.1 (d1712).
IR_MAX_CHROMA = 3.0
IR_SAMPLES = 5
# Linking: a track takes the nearest blob within LINK_X px of where it
# predicts the light, widened by LINK_X_PER_S per second since its last
# point, and within LINK_Y px in y. A one-point track has no velocity yet, so
# it looks up to FIRST_LINK_SPEED px/s either way (150 px a frame at 20 fps).
LINK_X, LINK_X_PER_S, LINK_Y = 60.0, 150.0, 40.0
FIRST_LINK_SPEED = 3000.0
MIN_POINTS = 3
# Judging: a light has no wheel, so its samples sit on a fixed near-lane wheel
# line (with calibration 2's H, y doesn't change X) in a car-sized box.
WHEEL_Y = 585.0
BOX_HALF_W, BOX_Y0, BOX_Y1 = 300.0, 380.0, 650.0
MAX_STEP_BACK = 30.0  # px a car's light may fall back on its furthest point
# One car's lights may not overlap in time: the labelled set's front/rear
# pairs overlap 14-67% of the shorter run or follow up to 0.5 s apart.
COMPANION_GAP_S = 1.0

_OPEN = np.ones((3, 3), np.uint8)


@dataclass(frozen=True)
class Blob:
    """Full-frame px; x, y is the centroid, x0..x1 the horizontal extent."""
    x: float
    y: float
    area: int
    x0: int
    x1: int


Frames = list[tuple[float, list[Blob]]]


@dataclass
class LightTrack:
    """One light followed through the frames; points ordered by t."""
    points: list[tuple[float, Blob]] = field(default_factory=list)

    @property
    def t0(self) -> float:
        return self.points[0][0]

    @property
    def t1(self) -> float:
        return self.points[-1][0]

    @property
    def xs(self) -> list[float]:
        return [b.x for _, b in self.points]

    @property
    def median_y(self) -> float:
        return float(np.median([b.y for _, b in self.points]))

    def predict(self, t: float) -> float:
        """Constant velocity from the last two points, so a gap carries the
        prediction forward instead of leaving it where the light vanished."""
        t1, b1 = self.points[-1]
        if len(self.points) == 1 or self.points[-2][0] >= t1:   # no velocity (or repeated pts)
            return b1.x
        t0, b0 = self.points[-2]
        return b1.x + (b1.x - b0.x) / (t1 - t0) * (t - t1)

    def reach(self, t: float) -> float:
        gap = t - self.t1
        if len(self.points) == 1:
            return LINK_X + FIRST_LINK_SPEED * gap
        return LINK_X + LINK_X_PER_S * gap


def find_blobs(gray: np.ndarray, y_offset: int) -> list[Blob]:
    """Saturated blobs of MIN_AREA..MAX_AREA in a grey band whose top row is
    y_offset in the full frame. The upper bound drops a saturated road or a
    washed-out frame, which would otherwise be one enormous blob."""
    return _blobs_in(_opened(gray), y_offset)


def _opened(gray: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx((gray >= SAT).astype(np.uint8), cv2.MORPH_OPEN, _OPEN)


class BlobFinder:
    """find_blobs for one clip's frames in order, reusing the last frame's
    blobs while its opened mask is unchanged: most frames' lights are street
    and porch lights that don't move. A frame's blobs depend only on that
    mask, so the output is the same (GPU-impact spike, finding 7: 73% less
    blob-finding CPU on 40 night clips). reused counts the frames it saved."""

    def __init__(self):
        self.reused = 0
        self._mask: np.ndarray | None = None
        self._y_offset: int | None = None
        self._blobs: list[Blob] = []

    def __call__(self, gray: np.ndarray, y_offset: int) -> list[Blob]:
        mask = _opened(gray)
        if self._mask is not None and y_offset == self._y_offset and np.array_equal(mask, self._mask):
            self.reused += 1
        else:
            self._mask, self._y_offset, self._blobs = mask, y_offset, _blobs_in(mask, y_offset)
        return list(self._blobs)


def _blobs_in(mask: np.ndarray, y_offset: int) -> list[Blob]:
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not MIN_AREA <= area <= MAX_AREA:
            continue
        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        blobs.append(Blob(float(cents[i][0]), float(cents[i][1]) + y_offset, area,
                          x0, x0 + int(stats[i, cv2.CC_STAT_WIDTH])))
    return blobs


def link_tracks(frames: Frames, max_gap_s: float = 0.4) -> list[LightTrack]:
    """Greedy, frame by frame: each live track takes the nearest unclaimed
    blob in its reach, closest pairs first; unclaimed blobs start tracks. A
    track ends after max_gap_s with no blob. Tracks of MIN_POINTS or more."""
    live: list[LightTrack] = []
    done: list[LightTrack] = []
    for t, blobs in frames:
        still = []
        for tr in live:
            # 1e-6: a gap of exactly max_gap_s (8 frames at 20 fps) still bridges
            (done if t - tr.t1 > max_gap_s + 1e-6 else still).append(tr)
        live = still
        pairs = []
        for i, tr in enumerate(live):
            px, reach, y = tr.predict(t), tr.reach(t), tr.points[-1][1].y
            pairs += [(abs(b.x - px), i, j) for j, b in enumerate(blobs)
                      if abs(b.x - px) <= reach and abs(b.y - y) <= LINK_Y]
        pairs.sort()
        took_track, took_blob = set(), set()
        for _, i, j in pairs:
            if i in took_track or j in took_blob:
                continue
            live[i].points.append((t, blobs[j]))
            took_track.add(i)
            took_blob.add(j)
        live += [LightTrack([(t, b)]) for j, b in enumerate(blobs) if j not in took_blob]
    return sorted((tr for tr in done + live if len(tr.points) >= MIN_POINTS), key=lambda tr: tr.t0)


def light_frames(clip: Path) -> tuple[Frames, bool]:
    """(pts seconds, blobs) for every frame, and whether any frame was scaled.
    Reuses vision._frames, so timestamps are pts and 640x360 exports are
    scaled up to 1920x1080 (scaled: the pass gets low_res, as YOLO's do)."""
    y0, y1 = BAND
    out: Frames = []
    low_res = False
    finder = BlobFinder()
    for t, frame, scaled in _frames(clip):
        low_res = low_res or scaled
        gray = cv2.cvtColor(frame[y0:y1], cv2.COLOR_BGR2GRAY)
        out.append((t, finder(gray, y0)))
    return out, low_res


def is_infrared(clip: Path) -> bool:
    """True when the clip's first IR_SAMPLES frames, read in sequence, are
    greyscale. No seeking: each seek decodes from a keyframe, and five cost
    more than decoding the whole clip."""
    cap = cv2.VideoCapture(str(clip))
    try:
        chromas = []
        for _ in range(IR_SAMPLES):
            ok, f = cap.read()
            if not ok:
                break
            f = f[::4, ::4].astype(np.int16)   # every 16th pixel is plenty for a mean
            chromas.append(float((np.abs(f[..., 0] - f[..., 1]) + np.abs(f[..., 1] - f[..., 2])).mean()))
    finally:
        cap.release()
    return bool(chromas) and float(np.mean(chromas)) < IR_MAX_CHROMA


@dataclass(frozen=True)
class Verdict:
    """What the rules made of one track; every track gets one, for the log."""
    track: LightTrack
    direction: int
    is_car: bool
    car_fails: tuple[str, ...]
    show_speed: bool
    speed_fails: tuple[str, ...]
    fit: SpeedFit | None
    companion: LightTrack | None = None
    length_m: float | None = None
    samples: tuple[Sample, ...] = ()  # the moving run
    cover: float = 0.0  # fraction of the outline's x range the run covers


@dataclass(frozen=True)
class NightCar:
    samples: list[Sample]
    direction: int
    fit: SpeedFit | None  # set only when the speed is shown
    length_m: float | None


def _sample(t: float, b: Blob) -> Sample:
    return Sample(t=t, x=b.x, y=WHEEL_Y, box=(b.x - BOX_HALF_W, BOX_Y0, b.x + BOX_HALF_W, BOX_Y1))


def _cover(xs: list[float], lo: float, hi: float) -> float:
    if not xs:
        return 0.0
    a, b = np.clip([min(xs), max(xs)], lo, hi)
    return float(b - a) / (hi - lo)


def _judge(tr: LightTrack, lo: float, hi: float, H, settings, night_lane=None) -> Verdict:
    samples = [_sample(t, b) for t, b in tr.points]
    run = _furthest_moving_run(samples, lambda s: s.x)
    i0 = next(i for i, s in enumerate(samples) if s is run[0])
    blobs = [b for _, b in tr.points[i0:i0 + len(run)]]
    xs = [s.x for s in run]
    direction = 1 if xs[-1] >= xs[0] else -1

    car_fails = []
    mx, my = float(np.median(xs)), float(np.median([b.y for b in blobs]))
    if night_lane is None and settings.night_lane_y0 is not None and settings.night_lane_y1 is not None:
        night_lane = (settings.night_lane_y0, settings.night_lane_y1)
    if night_lane is not None:
        y0, y1 = night_lane
        if not my > y0 + (y1 - y0) * mx / 1920:
            car_fails.append("lane")
    cover = _cover(xs, lo, hi) if len(run) >= 2 else 0.0
    if cover < settings.night_car_cover:
        car_fails.append("cover")
    ahead = np.maximum.accumulate(np.array(xs) * direction)
    if (ahead - np.array(xs) * direction).max() > MAX_STEP_BACK:
        car_fails.append("monotone")
    # The moving run's floor is ~2.5 mph per step; a walker's torch clears it.
    dt = run[-1].t - run[0].t
    metres = abs(to_world(H, run[-1].x, WHEEL_Y)[0] - to_world(H, run[0].x, WHEEL_Y)[0])
    if dt <= 0 or metres / dt * MPS_TO_MPH < settings.night_car_min_mph:
        car_fails.append("min_speed")
    is_car = not car_fails

    fit, speed_fails = None, []
    if is_car:
        inside = [s for s in run if lo <= s.x <= hi]
        if cover < settings.night_speed_cover:
            speed_fails.append("speed_cover")
        if len(inside) >= 3:
            fit = fit_speed(inside, H)
            if fit.sigma_m > settings.night_max_resid_m:
                speed_fails.append("steady")
            if not settings.night_min_mph <= fit.mph <= settings.night_max_mph:
                speed_fails.append("range")
            if fit.plus_minus > settings.max_uncertainty_mph:
                speed_fails.append("plus_minus")
        else:
            speed_fails.append("steady")
    return Verdict(track=tr, direction=direction, is_car=is_car, car_fails=tuple(car_fails),
                   show_speed=is_car and not speed_fails, speed_fails=tuple(speed_fails),
                   fit=fit, samples=tuple(run), cover=cover)


def _separation_m(a: Verdict, b: Verdict, H) -> float:
    """Median |X| from b's samples to a's straight-line path at the same
    times, metres. a's path is extrapolated, because a front light often
    leaves the frame as the rear one enters (n1259 shares 3 frames)."""
    ta, xa = [s.t for s in a.samples], [s.x for s in a.samples]
    k, m = np.polyfit(ta, xa, 1)
    return float(np.median([abs(to_world(H, s.x, WHEEL_Y)[0] - to_world(H, k * s.t + m, WHEEL_Y)[0])
                            for s in b.samples]))


def _gap_s(a: Verdict, b: Verdict) -> float:
    """Seconds between two runs' time spans; negative when they overlap."""
    return max(a.samples[0].t, b.samples[0].t) - min(a.samples[-1].t, b.samples[-1].t)


def _same_car(a: Verdict, b: Verdict, H, settings) -> float | None:
    """The separation in metres if b's light rides on a's car, else None:
    same direction, within COMPANION_GAP_S in time, and no further apart
    than night_companion_max_m. Closer than that is still the same car (a
    light split into two tracks, or its reflection), just not a length."""
    if a.direction != b.direction:
        return None
    if _gap_s(a, b) > COMPANION_GAP_S:
        return None
    sep = _separation_m(a, b, H)
    return sep if sep <= settings.night_companion_max_m else None


def _reflects(car: Verdict, far: Verdict, H, settings) -> bool:
    """True when car is far's reflection on a wet road: far failed the lane
    rule, and it moves with car (same direction, overlapping in time, its
    path within night_reflection_m)."""
    if "lane" not in far.car_fails or len(far.samples) < 2 or far.direction != car.direction:
        return False
    if _gap_s(far, car) > 0:
        return False
    return _separation_m(far, car, H) <= settings.night_reflection_m


HANDOVER_FIT_S = 0.5  # the earlier piece's last seconds that predict where the light went


def _handover_m(early: Verdict, late: Verdict, H) -> float | None:
    """How far, in metres, late's first sample is from where early's moving
    run was heading: a line through early's last HANDOVER_FIT_S, at late's
    first time. None with fewer than two points to fit."""
    tail = [s for s in early.samples if s.t >= early.samples[-1].t - HANDOVER_FIT_S]
    if len(tail) < 2 or tail[-1].t <= tail[0].t:
        return None
    k, m = np.polyfit([s.t for s in tail], [s.x for s in tail], 1)
    first = late.samples[0]
    return float(abs(to_world(H, k * first.t + m, WHEEL_Y)[0] - to_world(H, first.x, WHEEL_Y)[0]))


def _stitch(verdicts: list[Verdict], lo: float, hi: float, H, settings, night_lane=None) -> tuple[list[Verdict], list[str]]:
    """A car that lacks only coverage for a speed, joined with an in-lane
    piece of the same light: the street tree's trunk splits a passing glow
    (plan 2026-11-08-night-split-stitch). The joined track is judged afresh
    and kept only if it shows a speed; otherwise nothing changes. A shown
    speed is never touched, and each piece is used once. A car that rides
    with one already showing a speed (its other light) is left alone: its
    pass has that speed, and a longer stitched track would take its place
    (18:12 11-08: 17.3 mph would have read 18.9). Returns the verdicts and
    each join, described for the log."""
    out = list(verdicts)
    joins: list[str] = []
    used: set[int] = set()
    speeded = [v for v in verdicts if v.is_car and v.show_speed]

    def rides(x: Verdict) -> bool:
        return any(_same_car(v, x, H, settings) is not None or _same_car(x, v, H, settings) is not None
                   for v in speeded)

    cands = sorted((i for i, v in enumerate(out) if v.is_car and v.speed_fails == ("speed_cover",)),
                   key=lambda i: -out[i].cover)
    for i in cands:
        c = out[i]
        if rides(c):
            continue
        partners = []
        for j, p in enumerate(out):
            if j == i or j in used or i in used or p.show_speed or not p.samples:
                continue
            if p.direction != c.direction or "lane" in p.car_fails:
                continue
            gap = _gap_s(c, p)
            if abs(gap) > settings.night_stitch_gap_s:
                continue
            early, late = (c, p) if c.samples[0].t <= p.samples[0].t else (p, c)
            d = _handover_m(early, late, H)
            if d is not None and d <= settings.night_stitch_m:
                partners.append((d, j, early, late, gap))
        for d, j, early, late, gap in sorted(partners, key=lambda x: (x[0], x[1])):
            joined = LightTrack(early.track.points + [pt for pt in late.track.points if pt[0] > early.track.t1])
            v = _judge(joined, lo, hi, H, settings, night_lane)
            if v.is_car and v.show_speed and not rides(v):
                joins.append(f"track {c.track.t0:.2f}-{c.track.t1:.2f} s with {out[j].track.t0:.2f}-"
                             f"{out[j].track.t1:.2f} s (first ends at x {early.samples[-1].x:.0f}, second "
                             f"starts at x {late.samples[0].x:.0f}), handover {d:.2f} m, "
                             f"gap {gap:.2f} s, {c.fit.mph:.1f} mph alone, {v.fit.mph:.1f} joined")
                out[i] = v
                out[j] = replace(out[j], is_car=False, show_speed=False, car_fails=out[j].car_fails + ("stitched",))
                used |= {i, j}
                break
    return out, joins


def _harmless(before: list[NightCar], after: list[NightCar]) -> bool:
    """A stitch may only give a speed to a car without one (plan): the same
    cars each way, and every shown speed still shown, bit for bit. Checked on
    the result, as the reflection and same-car steps after it can undo what
    each join was careful about (a joined car no longer a reflection; a light
    kept first that leaves another riding with nothing)."""
    if sorted(c.direction for c in before) != sorted(c.direction for c in after):
        return False
    left = [c.fit for c in after if c.fit]
    for fit in (c.fit for c in before if c.fit):
        if fit not in left:
            return False
        left.remove(fit)
    return True


def night_cars(frames: Frames, road, H, settings, night_lane=None) -> tuple[list[NightCar], list[Verdict]]:
    """The cars to store, and a verdict for every track (for the log). A car
    track a lane-failed light moves with is its reflection, not a car. Car
    tracks are then taken best-covering first; a later one that moves with a
    kept car (_same_car) joins it instead of being a second car. With
    NIGHT_STITCH on, the cars are found with and without the stitch, and the
    stitched ones are kept only if they are the same cars with more speeds."""
    H = np.asarray(H, dtype=float)
    xs = [p[0] for p in road]
    lo, hi = float(min(xs)), float(max(xs))
    verdicts = [_judge(tr, lo, hi, H, settings, night_lane) for tr in link_tracks(frames)]
    cars, out, merges = _cars(verdicts, H, settings)
    if settings.night_stitch == "on":
        stitched, joins = _stitch(verdicts, lo, hi, H, settings, night_lane)
        if joins:
            s_cars, s_out, s_merges = _cars(stitched, H, settings)
            if _harmless(cars, s_cars):
                cars, out, merges = s_cars, s_out, s_merges
                for join in joins:
                    log.info("night lights: stitched %s", join)
            else:
                log.info("night lights: stitch undone, it changed the cars: %s", "; ".join(joins))
    for a, b, sep in merges:
        # Each merge is logged for the night review: by position alone, two
        # cars following within night_companion_max_m read as one car.
        log.info("night lights: one car: track %.2f-%.2f s with %.2f-%.2f s, %.2f m apart, gap %.2f s",
                 a.track.t0, a.track.t1, b.track.t0, b.track.t1, sep, _gap_s(a, b))
    return cars, out


def _cars(verdicts: list[Verdict], H, settings) -> tuple[list[NightCar], list[Verdict], list[tuple]]:
    """night_cars' last steps: reflections out, then cars kept best cover
    first, a later one riding with a kept car joining it. Returns the cars,
    the verdicts and each merge (kept, joined, metres apart) for the log."""
    verdicts = list(verdicts)
    merges: list[tuple] = []
    lane_failed = [v for v in verdicts if "lane" in v.car_fails]
    for i, v in enumerate(verdicts):
        if v.is_car and any(_reflects(v, f, H, settings) for f in lane_failed):
            verdicts[i] = replace(v, is_car=False, car_fails=v.car_fails + ("reflection",), show_speed=False)
    kept: list[int] = []
    for i in sorted((i for i, v in enumerate(verdicts) if v.is_car), key=lambda i: -verdicts[i].cover):
        for k in kept:
            sep = _same_car(verdicts[k], verdicts[i], H, settings)
            if sep is not None:
                merges.append((verdicts[k], verdicts[i], sep))
                length = sep if sep >= settings.night_companion_min_m else None
                if verdicts[k].companion is None and length is not None:
                    verdicts[k] = replace(verdicts[k], companion=verdicts[i].track, length_m=length)
                verdicts[i] = replace(verdicts[i], companion=verdicts[k].track, length_m=length)
                break
        else:
            kept.append(i)
    cars = [NightCar(samples=list(verdicts[k].samples), direction=verdicts[k].direction,
                     fit=verdicts[k].fit if verdicts[k].show_speed else None,
                     length_m=verdicts[k].length_m) for k in sorted(kept)]
    return cars, verdicts, merges
