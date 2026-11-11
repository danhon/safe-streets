"""One window of road motion: export it at full resolution, find the vehicles,
and store a pass (with a speed when calibrated) for each one."""

import asyncio
import json
import logging
import resource
import shutil
import time
from datetime import datetime, timedelta
from functools import partial

import cv2

from . import gpu
from .geometry import to_world
from .lights import night_cars
from .logtime import local
from .speed import fit_speed, gate_flags
from .track import Track
from .kinds import TWO_WHEEL_CLS
from .vision import crop_for, episodes, frame_at_factory, wide_crop_for

log = logging.getLogger(__name__)

MIN_SAMPLES = 5
# A two-wheeler within RACK_S of a car in its event, the same way, its
# speed within RACK_SHARE of the car's, is a bike on the car's rack (plan
# 2026-11-09-two-wheelers, final review). A cyclist a car overtakes is far slower.
RACK_S, RACK_SHARE = 2.0, 0.15


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _screenshot(frame, box, dest) -> None:
    x0, y0, x1, y1 = box
    mx, my = 0.4 * (x1 - x0), 0.4 * (y1 - y0)
    h, w = frame.shape[:2]
    crop = frame[max(0, int(y0 - my)):min(h, int(y1 + my)), max(0, int(x0 - mx)):min(w, int(x1 + mx))]
    if crop.size == 0:
        crop = frame
    scale = 480 / crop.shape[1]
    crop = cv2.resize(crop, (480, max(1, int(crop.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])


def _candidates(result, cal, settings) -> list:
    """(tid, samples, fit, flags, mph, ±, direction) for each track worth
    storing: enough samples and not parked."""
    # A draft (the road drawn, no scale yet) keeps tracks for picking
    # drive-bys from, without speeds.
    usable = cal is not None and cal.status not in ("stale", "draft")
    out = []
    for tid, samples in result.tracks.items():
        if len(samples) < MIN_SAMPLES:
            continue
        fit = fit_speed(samples, cal.H) if cal is not None and cal.status != "draft" else None
        if usable:
            flags = gate_flags(fit, samples, cal, settings, result.fps)
            if "parked" in flags:
                continue
            mph, pm, direction = fit.mph, fit.plus_minus, fit.direction
        else:
            # Direction from image motion; no speed.
            dx = samples[-1].x - samples[0].x
            if abs(dx) < 60:
                continue  # stationary in the image: parked
            flags = ["uncalibrated" if cal is None or cal.status == "draft" else "camera_moved"]
            mph = pm = None
            direction = 1 if dx > 0 else -1
        out.append((tid, samples, fit, flags, mph, pm, direction))
    return out


def _night_clock(ev, frames) -> tuple[datetime, float]:
    """The clip's wall-clock start and its median frame step. The clip ends
    at the event's end; it starts up to 5 s early, at a keyframe."""
    steps = sorted(b[0] - a[0] for a, b in zip(frames, frames[1:]))
    step = steps[len(steps) // 2] if steps else 0.0
    if ev.get("clip_start"):  # the source said when the clip's first frame was
        return ev["clip_start"], step
    return _dt(ev["end_ts"]) - timedelta(seconds=frames[-1][0] + step), step


def _night_on(settings, cal) -> bool:
    return settings.night_lights != "off" and cal is not None and cal.status not in ("stale", "draft")


async def _night(clip, cal, settings, vision, ev, st, lights=None):
    """Look for headlight cars in an infrared clip YOLO found nothing in.
    Logs a line per light track; returns (cars, frames, low_res), cars only
    in "on" mode. Nothing without a usable calibration or when the mode is
    "off". When it looks, its wall and CPU time go in st. lights: the
    motion gate's blobs from its own decode, if it took them."""
    eid = ev["id"]
    if not _night_on(settings, cal):
        return [], None, False
    t0, c0, g0 = time.monotonic(), _cpu(), _gpu()
    # Best-effort: a tracker bug must never fail an event, in any mode.
    try:
        kw = {"frames": lights} if lights is not None else {}
        found = await asyncio.to_thread(partial(vision.night_lights, **kw), clip)
        if not found or not found[0]:
            return [], None, False
        frames, low_res = found
        cars, verdicts = await asyncio.to_thread(night_cars, frames, cal.road, cal.H, settings, cal.night_lane)
        clip_start, _ = _night_clock(ev, frames)
        for v in verdicts:
            xs = v.track.xs
            # ts: when its pass would be, to find it in the recording
            mid = v.samples[len(v.samples) // 2].t if v.samples else v.track.t0
            log.info("night %s t=%.2f-%.2f dir=%+d x=%d-%d y=%d mph=%s pm=%s companion=%s car=%s speed=%s fails=%s "
                     "mode=%s ts=%s",
                     eid, v.track.t0, v.track.t1, v.direction, min(xs), max(xs), v.track.median_y,
                     f"{v.fit.mph:.1f}" if v.fit and v.show_speed else "-",
                     f"{v.fit.plus_minus:.1f}" if v.fit and v.show_speed else "-",
                     v.companion is not None, v.is_car, v.show_speed,
                     ",".join(v.car_fails + v.speed_fails) or "-", settings.night_lights,
                     local(clip_start + timedelta(seconds=mid), settings.tz))
    except Exception:  # noqa: BLE001
        log.exception("night lights failed for %s", eid)
        return [], None, False
    finally:
        st["night_ms"], st["night_cpu_ms"] = _ms(t0), _cpu_ms(c0)
        _put_gpu(st, "night", g0)
    return (cars if settings.night_lights == "on" else []), frames, low_res


def _span_text(span) -> str:
    return f"{span[0]:.2f}-{span[1]:.2f}" if span else "-"


def _keep(clip, dest) -> bool:
    """A copy of the clip for later: not a hard link, because a reanalysis
    exports over the event's clip in place (uiprotect opens it "wb")."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(clip, dest)
        return True
    except OSError:
        log.exception("gate shadow: couldn't keep %s", clip)
        return False


async def _shadow_gate(clip, crop, road, g, ev, settings, vision) -> None:
    """MOTION_GATE=shadow: after today's gate (g), which still decides, run
    the motion pre-gate on the same clip and log both. When one saw a
    vehicle and the other didn't, keep the clip in data/shadow/<id>.mp4
    (pruned like clips). Best-effort: it must never fail the event."""
    eid = ev["id"]
    t0, c0, g0 = time.monotonic(), _cpu(), _gpu()
    try:
        m = await asyncio.to_thread(vision.motion_gate, clip, crop, road)
        ms, cpu_ms, gpu = _ms(t0), _cpu_ms(c0), {}
        _put_gpu(gpu, "motion", g0)
        agree = (g.span is None) == (m.span is None)
        kept = not agree and await asyncio.to_thread(_keep, clip, settings.data_dir / "shadow" / f"{eid}.mp4")
        log.info("gate shadow %s %s agree=%s span=%s motion_span=%s inferences=%d motion_inferences=%d "
                 "moving=%d motion_moving=%d motion_ms=%d motion_cpu_ms=%d motion_gpu_ms=%s kept=%s",
                 eid, local(ev["start_ts"], settings.tz), agree, _span_text(g.span), _span_text(m.span),
                 g.inferences, m.inferences, len(g.moving), len(m.moving), ms, cpu_ms,
                 gpu.get("motion_gpu_ms", "-"), kept)
    except Exception:  # noqa: BLE001
        log.exception("gate shadow failed for %s", eid)


def _stages(st: dict) -> str:
    """The stage timings for the event's log line: export_ms, gate_ms and
    gate_cpu_ms, track_ms and track_cpu_ms (when it tracked), night_ms and
    night_cpu_ms (when it looked for headlights), classify_ms and
    classify_cpu_ms (the vehicle sidecar's CPU), each stage's *_gpu_ms
    where the process has a GPU open, and last frames (decoded by the gate)."""
    return " ".join(f"{k}={v}" for k, v in sorted(st.items(), key=lambda kv: kv[0] == "frames"))


def _stages_json(st: dict, ms: int) -> str:
    """The event line's stages and its total, for the table's timing column."""
    return json.dumps({**st, "ms": ms})


def _ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


def _cpu() -> float:
    """The process's CPU seconds so far, every thread's (GPU-impact plan,
    step 2). Not the thread's own: OpenVINO infers on its own threads, so
    a stage's thread would miss most of its inference. Events run one at a
    time, so the web app and the poster are the only other CPU in it."""
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def _cpu_ms(c0: float) -> int:
    return int((_cpu() - c0) * 1000)


def _gpu() -> int | None:
    """The process's GPU engine time so far, in ns (gpu.busy_ns); None
    without a GPU open, e.g. with YOLO on the CPU before it first runs."""
    return gpu.busy_ns()


def _put_gpu(st: dict, stage: str, g0: int | None) -> None:
    """st[<stage>_gpu_ms]: the GPU time since g0, when both ends were read."""
    g1 = _gpu() if g0 is not None else None
    if g1 is not None:
        st[f"{stage}_gpu_ms"] = round((g1 - g0) / 1e6)


async def _store_night(ev, clip, cars, frames, low_res, cal, settings, db, started, st=None) -> int:
    """Store the night cars as headlights passes; returns how many were new.
    When any, the event is done and the clip kept."""
    eid = ev["id"]
    clip_start, step = _night_clock(ev, frames)
    flags = ["headlights"] + (["low_res"] if low_res else [])
    await db.delete_passes(eid)  # from an earlier analysis, if any
    frame_at = frame_at_factory(clip)
    stored = 0
    for car in cars:
        mid = car.samples[len(car.samples) // 2]
        ts = clip_start + timedelta(seconds=mid.t)
        if await db.pass_near(ts.isoformat(), car.direction, 2, exclude_event=eid):
            continue
        fit = car.fit
        pid = await db.add_pass(
            event_id=eid, ts=ts.isoformat(), cls="car", direction=car.direction,
            mph=fit.mph if fit else None, plus_minus=fit.plus_minus if fit else None,
            metres=fit.metres if fit else None, seconds=car.samples[-1].t - car.samples[0].t,
            n=len(car.samples), flags=flags, length_m=car.length_m, calibration_id=cal.id,
            clip_t=max(0.0, car.samples[0].t - 1.0),
        )
        overlay = Track(
            box=mid.box, trail=[(q.x, q.y, q.t) for q in car.samples], track_id=None, cls="headlights",
            conf=None, vote="headlights", mph=fit.mph if fit else None, n=len(car.samples),
            fps=1 / step if step else None, seconds=car.samples[-1].t - car.samples[0].t,
            metres=fit.metres if fit else None, length_m=car.length_m, direction=car.direction,
            road=list(cal.road), crop=None, limit=settings.speed_limit_mph)
        await db.set_pass(pid, track_json=json.dumps(overlay.to_json()))
        frame = await asyncio.to_thread(frame_at, mid.t)
        if frame is not None:
            shot = settings.data_dir / "shots" / f"{pid}.jpg"
            await asyncio.to_thread(_screenshot, frame, mid.box, shot)
            await db.set_pass(pid, shot_path=str(shot))
        stored += 1
        log.info("pass %s %s dir=%+d mph=%s n=%d flags=%s", pid, local(ts, settings.tz), car.direction,
                 f"{fit.mph:.1f}" if fit else "-", len(car.samples), ",".join(flags))
    if stored:
        ms = int((time.monotonic() - started) * 1000)
        await db.set_event(eid, status="done", fps=1 / step if step else None, error=None,
                           clip_path=str(clip), ms=ms, stages_json=_stages_json(st or {}, ms))
        log.info("event %s %s done passes=%d night ms=%d %s", eid, local(ev["start_ts"], settings.tz), stored, ms,
                 _stages(st or {}))
    return stored


async def process_event(ev, *, source, vision, cal, settings, db) -> dict:
    """ev is an `events` row: a window of the recording. source: where the
    footage comes from (sources.Source)."""
    started = time.monotonic()
    ev = dict(ev)
    eid = ev["id"]
    start, end = _dt(ev["start_ts"]), _dt(ev["end_ts"])
    clip = settings.data_dir / "clips" / f"{eid}.mp4"
    attempts = ev["attempts"] + 1
    try:
        await db.set_event(eid, status="exporting", attempts=attempts)
        t0 = time.monotonic()
        ev["clip_start"] = await source.export(start, end, clip)
        st = {"export_ms": _ms(t0)}
        await db.set_event(eid, status="processing", clip_path=str(clip))

        road = cal.lane  # the batcher never runs an event without a calibration
        crop, wide = crop_for(road), wide_crop_for(road)

        mode = settings.motion_gate.strip().lower()
        t0, c0, g0 = time.monotonic(), _cpu(), _gpu()
        if mode == "on":  # with the night path's blobs, from the same decode (plan step 5)
            gate = partial(vision.motion_gate, lights=_night_on(settings, cal))
        else:
            gate = vision.gate
        g = await asyncio.to_thread(gate, clip, wide, road)
        st["gate_ms"], st["gate_cpu_ms"], st["frames"] = _ms(t0), _cpu_ms(c0), len(g.times)
        _put_gpu(st, "gate", g0)
        st["gate_inferences"] = g.inferences
        if mode == "shadow":
            await _shadow_gate(clip, wide, road, g, ev, settings, vision)
        span = g.span
        if span is None:
            cars, frames, low_res = await _night(clip, cal, settings, vision, ev, st, g.lights)
            if cars:
                stored = await _store_night(ev, clip, cars, frames, low_res, cal, settings, db, started, st)
                if stored:
                    return {"status": "done", "passes": stored}
            clip.unlink(missing_ok=True)
            ms = int((time.monotonic() - started) * 1000)
            await db.delete_passes(eid)  # from an earlier analysis, if any
            await db.set_event(eid, status="no_vehicle", clip_path=None, error=None, ms=ms,
                               stages_json=_stages_json(st, ms))
            log.info("event %s %s no_vehicle ms=%d %s", eid, local(ev["start_ts"], settings.tz), ms, _stages(st))
            return {"status": "no_vehicle", "passes": 0}

        times = g.times or None  # every pts, so track needn't decode past the span
        pad = (settings.track_pad_before_s, settings.track_pad_after_s)
        intervals = None
        if settings.episodes.strip().lower() == "on" and g.moving:
            intervals = episodes(g.moving, pad)
            st["episodes"] = len(intervals)
        t0, c0, g0 = time.monotonic(), _cpu(), _gpu()
        tracker = partial(vision.track, pad=pad, intervals=intervals)
        result = await asyncio.to_thread(tracker, clip, crop, road, span, times)
        track_inferences = result.inferences
        cands = _candidates(result, cal, settings)
        extra = []
        used_crop = crop
        if not cands:
            # Near-curb and night cars are often only found with the whole car
            # in view; the wide crop's speeds read ~10% slow, so flag them.
            result = await asyncio.to_thread(tracker, clip, wide, road, span, times)
            track_inferences += result.inferences
            cands = _candidates(result, cal, settings)
            extra = ["coarse"]
            used_crop = wide
        st = {**st, "track_ms": _ms(t0), "track_cpu_ms": _cpu_ms(c0)}
        _put_gpu(st, "track", g0)
        st["track_inferences"] = track_inferences
        if not cands:
            cars, frames, low_res = await _night(clip, cal, settings, vision, ev, st, g.lights)
            if cars:
                stored = await _store_night(ev, clip, cars, frames, low_res, cal, settings, db, started, st)
                if stored:
                    return {"status": "done", "passes": stored}
        # The clip began at the keyframe before `start`, up to 5 s early; its
        # end is the requested end. Times from `start` read late by that much,
        # so two windows over one car disagreed and both stored it.
        clip_start = ev["clip_start"] or (end - timedelta(seconds=result.duration) if result.duration else start)
        # Replace any passes from an earlier analysis of this event, only now
        # that the new ones are ready, so a reanalyse doesn't empty the page.
        await db.delete_passes(eid)
        # split_track can cut one car into pieces (a car coming in at the
        # crop's edge grows a box width each frame); pass_near below keeps
        # the first of them, so offer the one covering the most ground first.
        # A 15 mph drive-by read 12.6 from a 0.2 s head, not 15.8 from its
        # 1 s body (2026-11-07). Passes are ordered by ts everywhere else.
        # Cars before two-wheelers, so a bike riding on one (below) meets it.
        def two_wheeled(c) -> bool:
            return result.classes.get(c[0], "car") in TWO_WHEEL_CLS
        cands.sort(key=lambda c: (two_wheeled(c), -(c[2].metres if c[2] else abs(c[1][-1].x - c[1][0].x))))
        stored = 0
        pids = []
        cars_here: list[tuple[datetime, int, float]] = []   # this event's stored cars: ts, direction, mph
        for tid, samples, fit, flags, mph, pm, direction in cands:
            flags = flags + (extra if mph is not None else [])
            mid = samples[len(samples) // 2]
            ts = clip_start + timedelta(seconds=mid.t)
            if result.low_res:
                flags.append("low_res")
            # The same vehicle, seen in an overlapping window or cut in pieces;
            # a car and a two-wheeler side by side are two (kinds.TWO_WHEEL_CLS).
            two = two_wheeled((tid,))
            if await db.pass_near(ts.isoformat(), direction, 2, two_wheeler=two):
                continue
            # A two-wheeler at a car's speed beside it is on its rack, not a
            # rider: a cyclist a car overtakes is far slower.
            if two and mph is not None and any(
                    d == direction and abs((ts - t).total_seconds()) <= RACK_S and abs(mph - m) <= RACK_SHARE * m
                    for t, d, m in cars_here):
                continue
            # Screenshot: the sample nearest the middle of the road outline.
            xs = [p[0] for p in road]
            centre = (min(xs) + max(xs)) / 2
            best = min(samples, key=lambda s: abs(s.x - centre))
            length = None
            if cal is not None and cal.status != "draft":
                bx0, _, bx1, by1 = best.box
                length = abs(to_world(cal.H, bx1, by1)[0] - to_world(cal.H, bx0, by1)[0])
            if not two and mph is not None:
                cars_here.append((ts, direction, mph))
            pid = await db.add_pass(
                event_id=eid, ts=ts.isoformat(), cls=result.classes.get(tid, "car"), direction=direction,
                mph=mph, plus_minus=pm, metres=fit.metres if fit else None, seconds=samples[-1].t - samples[0].t,
                n=len(samples), flags=flags, length_m=length, calibration_id=cal.id if cal else None,
                clip_t=max(0.0, samples[0].t - 1.0),
            )
            vote = result.classes.get(tid, "car")
            overlay = Track(
                box=tuple(best.box), trail=[(q.x, q.y, q.t) for q in samples], track_id=tid // 100,
                cls=best.cls or vote, conf=best.conf, vote=vote, mph=mph, n=len(samples), fps=result.fps,
                seconds=samples[-1].t - samples[0].t, metres=fit.metres if fit else None, length_m=length,
                direction=direction, road=list(road), crop=tuple(used_crop), limit=settings.speed_limit_mph)
            await db.set_pass(pid, track_json=json.dumps(overlay.to_json()))
            shot = settings.data_dir / "shots" / f"{pid}.jpg"
            frame = await asyncio.to_thread(result.frame_at, best.t)
            if frame is not None:
                await asyncio.to_thread(_screenshot, frame, best.box, shot)
                await db.set_pass(pid, shot_path=str(shot))
            stored += 1
            pids.append(pid)
            log.info("pass %s %s dir=%+d mph=%s n=%d flags=%s", pid, local(ts, settings.tz), direction,
                     f"{mph:.1f}" if mph is not None else "-", len(samples), ",".join(flags) or "-")

        if stored == 0:
            clip.unlink(missing_ok=True)
        ms = int((time.monotonic() - started) * 1000)
        await db.set_event(eid, status="done", fps=result.fps, error=None,
                           clip_path=str(clip) if stored else None, ms=ms, stages_json=_stages_json(st, ms))
        log.info("event %s %s done passes=%d tracks=%d wide=%s fps=%.1f ms=%d %s", eid,
                 local(ev["start_ts"], settings.tz), stored,
                 len(result.tracks), bool(extra), result.fps, ms, _stages(st))
        return {"status": "done", "passes": stored}
    except Exception as e:  # noqa: BLE001
        log.exception("event %s failed", eid)
        clip.unlink(missing_ok=True)
        await db.set_event(eid, status="failed", error=f"{type(e).__name__}: {e}"[:500], clip_path=None)
        return {"status": "failed", "passes": 0}
