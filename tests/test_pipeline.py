import json
import dataclasses
import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from night_helpers import CAL2_H, CAL2_ROAD, M_PER_PX, _path

from safe_streets.geometry import Calibration
from safe_streets import pipeline
from safe_streets.pipeline import process_event
from safe_streets.speed import Sample
from safe_streets.vision import Gate, TrackResult, crop_for, wide_crop_for
from listing import first_page

H = np.array([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 1.0]])  # 1 px = 5 cm
ROAD = [(0, 400), (1920, 400), (1920, 600), (0, 600)]
T0 = datetime(2026, 11, 2, 19, 21, 0, tzinfo=timezone.utc)


NARROW = [(500, 400), (1400, 400), (1400, 600), (500, 600)]  # room for the wide crop


def cal(status="active", road=ROAD):
    return Calibration(1, [], [], H, road, "", status)


def track(x0, px_per_s, n=40, fps=20, y=500):
    return [Sample(t=1 + i / fps, x=x0 + px_per_s * i / fps, y=y, box=(x0 - 100, y - 80, x0 + 100, y))
            for i in range(n)]


class FakeProtect:
    def __init__(self, fail=False):
        self.fail = fail

    async def export(self, start, end, dest, *, low_res=False):
        if self.fail:
            raise RuntimeError("export failed: boom")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"mp4")


class FakeVision:
    """motion: the motion pre-gate's verdict (default: the same as today's
    gate's). The two gates give different spans, so a test can tell which
    one tracking used; gates records which ran, in order."""

    def __init__(self, tracks=None, moving=True, lights=None, low_res=False, motion=None, classes=None):
        self.tracks = tracks or {}
        self.classes = classes or {}
        self.moving = moving
        self.motion = moving if motion is None else motion
        self.lights = lights
        self.low_res = low_res
        self.night_called = False
        self.gates = []

    def night_lights(self, clip, **kw):
        self.night_called = True
        self.night_kw = kw
        return None if self.lights is None else (self.lights, self.low_res)

    def gate(self, clip, crop, road):
        self.gates.append("gate")
        return Gate((3.0, 4.5) if self.moving else None, [i / 20 for i in range(200)],
                    [3.0, 4.5] if self.moving else [], 9)

    def motion_gate(self, clip, crop, road, **kw):
        """gate_lights: the night blobs it hands back when asked for them."""
        self.gates.append("motion")
        self.motion_kw = kw
        return Gate((3.5, 4.0) if self.motion else None, [i / 20 for i in range(200)],
                    [3.5, 4.0] if self.motion else [], 2,
                    getattr(self, "gate_lights", None) if kw.get("lights") else None)

    def track(self, clip, crop, road, span=None, times=None, **kw):
        self.span, self.times, self.kw = span, times, kw
        return TrackResult(self.tracks, 20.0, lambda t: np.zeros((1080, 1920, 3), np.uint8),
                           {k: self.classes.get(k, "car") for k in self.tracks}, False,
                           getattr(self, "duration", None), 7)


class CropVision(FakeVision):
    """Returns different tracks for the normal and the wide crop."""

    def __init__(self, normal, wide):
        super().__init__()
        self.by_crop = {crop_for(NARROW): normal, wide_crop_for(NARROW): wide}
        self.gate_crops, self.track_crops, self.spans, self.track_times, self.pads = [], [], [], [], []
        self.intervals = []

    def gate(self, clip, crop, road):
        self.gate_crops.append(crop)
        return Gate((3.0, 4.5), [i / 20 for i in range(200)], [3.0, 4.5], 9)

    def track(self, clip, crop, road, span=None, times=None, **kw):
        self.track_crops.append(crop)
        self.spans.append(span)
        self.track_times.append(times)
        self.pads.append(kw.get("pad"))
        self.intervals.append(kw.get("intervals"))
        self.tracks = self.by_crop[crop]
        return super().track(clip, crop, road, span)


async def event(db, id="w1", start=T0):
    await db.upsert_event(id, start.isoformat(), (start + timedelta(seconds=10)).isoformat())
    return await db.get_event(id)


async def run(db, settings, vision, c=None, protect=None, ev=None):
    ev = ev or await event(db)
    return await process_event(ev, source=protect or FakeProtect(), vision=vision, cal=c, settings=settings, db=db)


async def test_two_cars_two_passes(db, settings):
    v = FakeVision({1: track(10, 960), 2: track(1900, -960)})
    r = await run(db, settings, v, cal())
    rows, total = await first_page(db)
    assert r["passes"] == 2 and total == 2
    assert sorted(row["direction"] for row in rows) == [-1, 1]
    assert all(abs(row["mph"] - 48 * 2.23694) < 1 for row in rows)
    assert (await db.get_event("w1"))["status"] == "done"
    assert all(row["shot_path"] for row in rows)
    assert all(abs(row["length_m"] - 10.0) < 0.01 for row in rows)  # 200 px box at 5 cm/px










def _picture_logs(caplog):
    return [m for m in caplog.messages if " hero " in m or " card " in m or "overlay not drawn" in m]








async def test_pass_ts_is_clip_start_plus_mid_t(db, settings):
    await run(db, settings, FakeVision({1: track(10, 960)}), cal())
    rows, _ = await first_page(db)
    ts = datetime.fromisoformat(rows[0]["ts"])
    assert abs((ts - T0).total_seconds() - (1 + 39 / 40)) < 0.1


async def test_pass_ts_anchored_at_the_clips_end(db, settings):
    """Protect starts an export at the keyframe before the requested start
    (keyframes 5 s apart) and ends it at the requested end. The event
    window is 10 s; a 12.5 s clip started 2.5 s early."""
    v = FakeVision({1: track(10, 960)})
    v.duration = 12.5
    await run(db, settings, v, cal())
    rows, _ = await first_page(db)
    ts = datetime.fromisoformat(rows[0]["ts"])
    assert abs((ts - T0).total_seconds() - (1 + 39 / 40 - 2.5)) < 0.1


async def test_parked_track_dropped(db, settings):
    await run(db, settings, FakeVision({1: track(900, 0)}), cal())
    _, total = await first_page(db)
    assert total == 0


async def test_no_vehicle_deletes_clip(db, settings):
    r = await run(db, settings, FakeVision(moving=False), cal())
    ev = await db.get_event("w1")
    assert r["status"] == "no_vehicle" and ev["status"] == "no_vehicle" and ev["clip_path"] is None
    assert not list((settings.data_dir / "clips").glob("*.mp4"))




async def test_stale_calibration_flags_camera_moved(db, settings):
    await run(db, settings, FakeVision({1: track(10, 960)}), cal("stale"))
    rows, _ = await first_page(db)
    assert rows[0]["mph"] is None and "camera_moved" in rows[0]["flags"]


async def test_export_failure_marks_failed_and_counts_attempt(db, settings):
    r = await run(db, settings, FakeVision(), cal(), protect=FakeProtect(fail=True))
    ev = await db.get_event("w1")
    assert r["status"] == "failed" and ev["status"] == "failed" and ev["attempts"] == 1
    assert "boom" in ev["error"]


async def test_overlapping_events_one_pass(db, settings):
    v = FakeVision({1: track(10, 960)})
    await run(db, settings, v, cal(), ev=await event(db, "w1", T0))
    await run(db, settings, v, cal(), ev=await event(db, "w2", T0 + timedelta(seconds=0.5)))
    _, total = await first_page(db)
    assert total == 1


def piece(t0, x0, px_per_s, n, fps=20, y=500):
    return [Sample(t=t0 + i / fps, x=x0 + px_per_s * i / fps, y=y, box=(x0 - 100, y - 80, x0 + 100, y))
            for i in range(n)]


async def test_pieces_of_one_car_store_the_one_covering_most_ground(db, settings):
    """split_track can cut one car into a short head (as it comes in at the
    crop's edge) and the body; pass_near then kept whichever came first, the
    head. A 15 mph drive-by read 12.6 from a 0.2 s head; its
    1 s body read 15.8."""
    head = piece(1.0, 10, 480, n=5)    # 0.2 s, 96 px = 4.8 m, 24 m/s
    body = piece(1.25, 110, 960, n=20)  # 0.95 s, 912 px = 45.6 m, 48 m/s
    r = await run(db, settings, FakeVision({1: head, 2: body}), cal())
    rows, total = await first_page(db)
    assert r["passes"] == 1 and total == 1
    assert rows[0]["n"] == 20
    assert abs(rows[0]["mph"] - 48 * 2.23694) < 0.1


async def test_a_car_overtaking_a_cyclist_is_two_passes(db, settings):
    """The 2 s same-car check is per group (plan 2026-11-09-two-wheelers,
    final review): a car passing a cyclist mustn't hide either of them."""
    car = piece(1.0, 110, 960, n=20)     # 0.95 s across
    bike = piece(0.5, 110, 240, n=40)    # 1.95 s, slower
    r = await run(db, settings, FakeVision({1: car, 2: bike}, classes={1: "car", 2: "bicycle"}), cal())
    rows, total = await first_page(db)
    assert r["passes"] == 2 and total == 2
    assert sorted(row["cls"] for row in rows) == ["bicycle", "car"]


async def test_a_bike_moving_with_a_car_is_its_rack(db, settings):
    """A bike at the car's speed beside it is on its rack, whichever track
    covered more ground: one car pass, never a "Bike" at car speed."""
    car = piece(1.0, 110, 960, n=20)
    rack = piece(1.0, 60, 960, n=24)    # the same speed, more ground
    r = await run(db, settings, FakeVision({1: car, 2: rack}, classes={1: "car", 2: "bicycle"}), cal())
    rows, total = await first_page(db)
    assert r["passes"] == 1 and total == 1 and rows[0]["cls"] == "car"


async def test_pieces_of_one_bike_are_still_one_pass(db, settings):
    head, body = piece(1.0, 10, 480, n=5), piece(1.25, 110, 960, n=20)
    r = await run(db, settings, FakeVision({1: head, 2: body}, classes={1: "motorcycle", 2: "bicycle"}), cal())
    assert r["passes"] == 1


async def test_gate_uses_wide_crop(db, settings):
    v = CropVision({1: track(10, 960)}, {})
    await run(db, settings, v, cal(road=NARROW))
    assert v.gate_crops == [wide_crop_for(NARROW)]


async def test_wide_fallback_when_normal_crop_finds_nothing(db, settings):
    v = CropVision({}, {1: track(10, 960)})
    r = await run(db, settings, v, cal(road=NARROW))
    assert r["passes"] == 1
    assert v.track_crops == [crop_for(NARROW), wide_crop_for(NARROW)]
    p = (await db.recent_passes(10)) or (await first_page(db))[0]
    assert "coarse" in p[0]["flags"].split(",")


async def test_no_fallback_when_normal_crop_finds_a_car(db, settings):
    v = CropVision({1: track(10, 960)}, {2: track(10, 960)})
    r = await run(db, settings, v, cal(road=NARROW))
    assert r["passes"] == 1
    assert v.track_crops == [crop_for(NARROW)]
    rows, _ = await first_page(db)
    assert "coarse" not in rows[0]["flags"]


async def test_parked_only_on_normal_crop_still_falls_back(db, settings):
    v = CropVision({1: track(500, 0)}, {2: track(10, 960)})
    r = await run(db, settings, v, cal(road=NARROW))
    assert r["passes"] == 1 and len(v.track_crops) == 2


async def test_track_gets_the_gates_span(db, settings):
    v = CropVision({}, {1: track(10, 960)})
    await run(db, settings, v, cal(road=NARROW))
    assert v.spans == [(3.0, 4.5), (3.0, 4.5)]  # the normal track and the fallback


async def test_log_has_stage_timings(db, settings, caplog):
    caplog.set_level(logging.INFO)
    await run(db, settings, FakeVision({1: track(10, 960)}), cal())
    await run(db, settings, FakeVision(moving=False), cal(), ev=await event(db, "w2", T0 + timedelta(minutes=5)))
    [done] = [m for m in caplog.messages if m.startswith("event w1 ") and " done " in m]
    [none] = [m for m in caplog.messages if m.startswith("event w2 ") and " no_vehicle " in m]
    for key in ("export_ms=", "gate_ms=", "track_ms=", "frames="):
        assert key in done
    for key in ("export_ms=", "gate_ms=", "frames="):
        assert key in none
    assert "track_ms=" not in none


async def test_log_has_each_stages_cpu_beside_its_wall_time(db, settings, caplog, monkeypatch):
    # The process's CPU, not the thread's: OpenVINO infers on its own threads.
    caplog.set_level(logging.INFO)
    clock = iter(range(0, 100_000, 250))  # each reading 0.25 s of CPU after the last
    monkeypatch.setattr(pipeline, "_cpu", lambda: next(clock) / 1000)
    await run(db, settings, FakeVision({1: track(10, 960)}), cal())
    [done] = [m for m in caplog.messages if m.startswith("event w1 ") and " done " in m]
    kv = dict(f.split("=", 1) for f in done.split() if "=" in f)
    assert kv["gate_cpu_ms"] == "250" and kv["track_cpu_ms"] == "250"
    assert list(kv)[-1] == "frames"


async def test_each_events_stages_are_stored_for_the_table(db, settings):
    # The table's optional timing column: the event line's stages, in the DB.
    await run(db, settings, FakeVision({1: track(10, 960)}), cal())
    await run(db, settings, FakeVision(moving=False), cal(), ev=await event(db, "w2", T0 + timedelta(minutes=5)))
    done = json.loads((await db.get_event("w1"))["stages_json"])
    none = json.loads((await db.get_event("w2"))["stages_json"])
    assert done["ms"] == (await db.get_event("w1"))["ms"]
    assert {"export_ms", "gate_ms", "gate_cpu_ms", "gate_inferences", "track_ms", "track_inferences"} <= set(done)
    assert {"export_ms", "gate_ms", "gate_inferences"} <= set(none) and "track_ms" not in none


async def test_log_counts_yolo_calls_per_stage(db, settings, caplog):
    # The gate's and tracking's YOLO calls: what the GPU (or the CPU) does.
    caplog.set_level(logging.INFO)
    await run(db, settings, FakeVision({1: track(10, 960)}), cal())
    await run(db, settings, FakeVision(moving=False), cal(), ev=await event(db, "w2", T0 + timedelta(minutes=5)))
    [done] = [m for m in caplog.messages if m.startswith("event w1 ") and " done " in m]
    [none] = [m for m in caplog.messages if m.startswith("event w2 ") and " no_vehicle " in m]
    kv = dict(f.split("=", 1) for f in done.split() if "=" in f)
    assert kv["gate_inferences"] == "9" and kv["track_inferences"] == "7"
    assert list(kv)[-1] == "frames"
    kv = dict(f.split("=", 1) for f in none.split() if "=" in f)
    assert kv["gate_inferences"] == "9" and "track_inferences" not in kv


async def test_log_times_the_night_path(db, settings_on, caplog):
    caplog.set_level(logging.INFO)
    v = FakeVision(moving=False)  # no span and no lights: the night path runs, finds nothing
    await run(db, settings_on, v, night_cal())
    [none] = [m for m in caplog.messages if m.startswith("event w1 ") and " no_vehicle " in m]
    assert v.night_called
    kv = dict(f.split("=", 1) for f in none.split() if "=" in f)
    assert "night_ms" in kv and "night_cpu_ms" in kv
    assert list(kv)[-1] == "frames"


def _local(stamp: str, stored: str) -> bool:
    """A logged time is the stored instant (to the ms) in Pacific time,
    which is what logdy's Time column and Protect show."""
    shown, at = datetime.fromisoformat(stamp), datetime.fromisoformat(stored.replace("Z", "+00:00"))
    return abs((shown - at).total_seconds()) < 0.001 and shown.utcoffset() == timedelta(hours=-8)


async def test_log_gives_event_and_pass_times_in_local_time(db, settings, caplog):
    caplog.set_level(logging.INFO)
    s = dataclasses.replace(settings, tz="America/Los_Angeles")
    await run(db, s, FakeVision({1: track(10, 960)}), cal())
    await run(db, s, FakeVision(moving=False), cal(), ev=await event(db, "w2", T0 + timedelta(minutes=5)))
    rows, _ = await first_page(db)
    [done] = [m for m in caplog.messages if m.startswith("event w1 ") and " done " in m]
    [none] = [m for m in caplog.messages if m.startswith("event w2 ") and " no_vehicle " in m]
    [p] = [m for m in caplog.messages if m.startswith(f"pass {rows[0]['id']} ") and " dir=" in m]
    assert done.split()[2] == "2026-11-02T11:21:00.000-08:00"  # T0, 19:21 UTC
    assert none.split()[2] == "2026-11-02T11:26:00.000-08:00"
    assert _local(p.split()[2], rows[0]["ts"])


async def test_tracking_uses_the_pad_settings(db, settings):
    v = CropVision(normal={}, wide={1: track(10, 960)})  # both crops tracked
    s = dataclasses.replace(settings, track_pad_before_s=1.0, track_pad_after_s=0.5)
    await run(db, s, v, cal(road=NARROW))
    assert v.pads == [(1.0, 0.5), (1.0, 0.5)]


async def test_track_gets_the_gates_times(db, settings):
    v = CropVision(normal={}, wide={1: track(10, 960)})  # both crops tracked
    await run(db, settings, v, cal(road=NARROW))
    assert v.track_times == [[i / 20 for i in range(200)]] * 2


async def test_track_without_times_keeps_old_behaviour(db, settings):
    class NoTimes(FakeVision):
        def gate(self, clip, crop, road):
            return Gate((3.0, 4.5), [])

    v = NoTimes({1: track(10, 960)})
    await run(db, settings, v, cal())
    assert v.times is None
    [row], _ = await first_page(db)
    mid = track(10, 960)[20].t
    assert row["ts"] == (T0 + timedelta(seconds=mid)).isoformat()  # duration None: from start


async def test_reanalysed_event_replaces_its_old_passes(db, settings):
    ev = await event(db)
    await db.add_pass(event_id="w1", ts=(T0 + timedelta(seconds=5)).isoformat(), cls="car", direction=1, mph=99.0)
    await db.set_event("w1", status="queued")
    await run(db, settings, FakeVision({1: track(10, 960)}), cal(), ev=await db.get_event("w1"))
    rows, total = await first_page(db)
    assert total == 1 and rows[0]["mph"] != 99.0


async def test_reanalysed_event_with_no_vehicle_loses_its_old_passes(db, settings):
    await event(db)
    await db.add_pass(event_id="w1", ts=T0.isoformat(), cls="car", direction=1, mph=99.0)
    await run(db, settings, FakeVision(moving=False), cal(), ev=await db.get_event("w1"))
    assert (await first_page(db))[1] == 0


# --- night: headlights ----------------------------------------------------

def night_cal(status="active"):
    return Calibration(2, [], [], CAL2_H, CAL2_ROAD, "", status)


STEADY = _path(lambda t: 900 * t, 1920 / 900)  # a near light at ~22 mph, 20 fps


@pytest.fixture
def settings_on(settings):
    return dataclasses.replace(settings, night_lights="on")


@pytest.fixture
def settings_log(settings):
    return dataclasses.replace(settings, night_lights="log")


@pytest.fixture
def settings_off(settings):
    return dataclasses.replace(settings, night_lights="off")


async def test_night_clip_no_vehicle_stores_headlights_pass_when_on(db, settings_on):
    r = await run(db, settings_on, FakeVision(moving=False, lights=STEADY), night_cal())
    rows, total = await first_page(db)
    ev = await db.get_event("w1")
    assert r == {"status": "done", "passes": 1} and total == 1
    assert rows[0]["flags"] == "headlights" and rows[0]["direction"] == 1 and rows[0]["cls"] == "car"
    assert abs(rows[0]["mph"] - 900 * M_PER_PX * 2.23694) < 0.5
    assert ev["status"] == "done" and ev["clip_path"] and abs(ev["fps"] - 20) < 0.5
    assert (settings_on.data_dir / "clips" / "w1.mp4").exists()
    o = __import__("json").loads(rows[0]["track_json"])
    assert o["cls"] == "headlights" and o["conf"] is None and o["crop"] is None


async def test_night_pass_from_a_low_res_clip_is_flagged(db, settings_on):
    await run(db, settings_on, FakeVision(moving=False, lights=STEADY, low_res=True), night_cal())
    rows, _ = await first_page(db)
    assert rows[0]["flags"] == "headlights,low_res"


async def test_night_pass_time_anchored_at_the_clips_end(db, settings_on):
    await run(db, settings_on, FakeVision(moving=False, lights=STEADY), night_cal())
    rows, _ = await first_page(db)
    ts = datetime.fromisoformat(rows[0]["ts"])
    duration = STEADY[-1][0] + 0.05
    assert abs((ts - (T0 + timedelta(seconds=10 - duration))).total_seconds() - 1.07) < 0.15


async def test_night_lights_after_both_yolo_crops_empty(db, settings_on):
    class Empty(FakeVision):
        tracked = 0

        def gate(self, clip, crop, road):
            return Gate((3.0, 4.5), [i / 20 for i in range(200)])

        def track(self, clip, crop, road, span=None, times=None, **kw):
            self.tracked += 1
            return super().track(clip, crop, road, span, times)

    v = Empty(lights=STEADY)
    r = await run(db, settings_on, v, night_cal())
    rows, total = await first_page(db)
    assert r["passes"] == 1 and total == 1 and rows[0]["flags"] == "headlights"
    assert v.tracked == 2  # normal and wide crops, then the lights


@pytest.mark.parametrize("mode", ["on", "log"])
async def test_night_tracker_error_does_not_fail_event(db, settings, caplog, mode):
    class Boom(FakeVision):
        def night_lights(self, clip):
            raise RuntimeError("tracker bug")

    s = dataclasses.replace(settings, night_lights=mode)
    r = await run(db, s, Boom(moving=False), night_cal())
    ev = await db.get_event("w1")
    _, total = await first_page(db)
    assert r["status"] == "no_vehicle" and ev["status"] == "no_vehicle" and ev["clip_path"] is None
    assert total == 0 and not list((s.data_dir / "clips").glob("*.mp4"))
    assert any("night lights failed" in m and "w1" in m for m in caplog.messages)
    assert any(rec.exc_info for rec in caplog.records if "night lights failed" in rec.getMessage())


async def test_night_lights_not_run_when_yolo_found_a_car(db, settings_on):
    v = FakeVision({1: track(10, 960)}, lights=STEADY)
    await run(db, settings_on, v, cal())
    assert v.night_called is False


async def test_day_clip_never_stores_night_pass(db, settings_on):
    v = FakeVision(moving=False, lights=None)
    r = await run(db, settings_on, v, night_cal())
    ev = await db.get_event("w1")
    assert v.night_called and r["status"] == "no_vehicle" and ev["status"] == "no_vehicle"
    assert ev["clip_path"] is None and not list((settings_on.data_dir / "clips").glob("*.mp4"))


async def test_log_mode_stores_nothing(db, settings_log, caplog):
    caplog.set_level(logging.INFO)
    v = FakeVision(moving=False, lights=STEADY)
    r = await run(db, settings_log, v, night_cal())
    _, total = await first_page(db)
    assert settings_log.night_lights == "log" and total == 0 and r["status"] == "no_vehicle"
    assert (await db.get_event("w1"))["status"] == "no_vehicle"
    assert any(m.startswith("night w1 t=") and "mode=log" in m for m in caplog.messages)


async def test_night_log_line_gives_the_pass_time(db, settings_on, caplog):
    # the log-mode review finds each car in the recording by this ts
    caplog.set_level(logging.INFO)
    s = dataclasses.replace(settings_on, tz="America/Los_Angeles")
    await run(db, s, FakeVision(moving=False, lights=STEADY), night_cal())
    rows, _ = await first_page(db)
    [line] = [m for m in caplog.messages if m.startswith("night w1 t=") and "car=True" in m]
    [p] = [m for m in caplog.messages if m.startswith(f"pass {rows[0]['id']} ")]
    [done] = [m for m in caplog.messages if m.startswith("event w1 ") and " done " in m]
    assert _local(line.rsplit(" ts=", 1)[1], rows[0]["ts"])
    assert _local(p.split()[2], rows[0]["ts"])
    assert done.split()[2] == "2026-11-02T11:21:00.000-08:00"


async def test_off_mode_never_calls_night_lights(db, settings_off):
    v = FakeVision(moving=False, lights=STEADY)
    r = await run(db, settings_off, v, night_cal())
    assert v.night_called is False and r["status"] == "no_vehicle"


@pytest.mark.parametrize("c", [night_cal("stale")])
async def test_night_lights_skipped_without_usable_calibration(db, settings_on, c):
    v = FakeVision(moving=False, lights=STEADY)
    r = await run(db, settings_on, v, c)
    assert v.night_called is False and r["status"] == "no_vehicle"


async def test_two_night_cars_2s_apart_two_passes(db, settings_on):
    frames = _path(lambda t: [900 * t, 900 * (t - 2.4)], 1920 / 900 + 2.4)
    r = await run(db, settings_on, FakeVision(moving=False, lights=frames), night_cal())
    assert r["passes"] == 2


async def test_night_pass_dedups_against_another_events_pass(db, settings_on):
    await run(db, settings_on, FakeVision(moving=False, lights=STEADY), night_cal())
    rows, _ = await first_page(db)
    await db.delete_passes("w1")
    await db.add_pass(event_id="w0", ts=rows[0]["ts"], cls="car", direction=1, mph=20, plus_minus=1,
                      metres=10, seconds=1, n=10, flags="")
    ev = await event(db, "w2")
    r = await run(db, settings_on, FakeVision(moving=False, lights=STEADY), night_cal(), ev=ev)
    _, total = await first_page(db)
    assert r["status"] == "no_vehicle" and total == 1
    assert (await db.get_event("w2"))["clip_path"] is None


async def test_reanalysing_a_night_event_does_not_dedup_against_itself(db, settings_on):
    for _ in range(2):
        r = await run(db, settings_on, FakeVision(moving=False, lights=STEADY), night_cal())
        assert r["passes"] == 1
    _, total = await first_page(db)
    assert total == 1


async def test_night_speedless_pass_has_null_mph(db, settings_on):
    def x_at(t):
        return 400.0 if t < 9 else 400 + 60 * (t - 9) ** 3
    frames = _path(x_at, 9 + (1500 / 60) ** (1 / 3))
    r = await run(db, settings_on, FakeVision(moving=False, lights=frames), night_cal())
    rows, _ = await first_page(db)
    assert r["passes"] == 1 and rows[0]["mph"] is None and rows[0]["flags"] == "headlights"


NIGHT_CLIP = Path(__file__).parent / "fixtures" / "night_car.mp4"
MODELS = Path(__file__).parent.parent / "models"


class CopyProtect(FakeProtect):
    async def export(self, start, end, dest, *, low_res=False):
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(NIGHT_CLIP, dest)




OUTBACK_CLIP = Path(__file__).parent / "fixtures" / "outback_15mph.mp4"


class ClipProtect(FakeProtect):
    def __init__(self, clip):
        super().__init__()
        self.clip = clip

    async def export(self, start, end, dest, *, low_res=False):
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.clip, dest)




# The known-speed drive-bys, all
# rightward in the near lane. Each band is around the true speed; the
# comment gives what the test reads (CPU, 2026-11-08) and what live read.
# The fixtures not cut by hand were cut at a keyframe without re-encoding.
DRIVES = [
    # 11:21:14 2026-11-02, 15 mph cruise. Protect's motion events missed it;
    # exported from the continuous recording on 11-08 (19:21:00-19:21:30 UTC),
    # 10 s from 5.0 s. Reads 14.46 from a short 8-frame track (the car comes
    # out from behind a parked SUV).
    pytest.param("kia_15mph_1121.mp4", 14.0, 16.5, 8, id="15mph-1121"),
    # 11:22:28 2026-11-02, 15 mph cruise; live read 14.97 (pass 287). Reads 15.75.
    pytest.param("car.mp4", 14.0, 16.5, 12, id="15mph-1122"),
    # ~09:16:25 2026-11-04, 20 mph; missed live (event 9f96bb1e). Reads 21.88:
    # the open +1.9 (TODO.md), so the band is the design's ±2 mph.
    pytest.param("kia_20mph.mp4", 18.0, 22.0, 10, id="20mph"),
    # 16:29:56 2026-11-08, 19 mph on its digital speedometer, camera in
    # Protect's sport mode (25 fps); live read 19.3 (pass 2267, GPU). 10 s
    # from the event clip's 5.0 s. Reads 19.31 from all 26 frames.
    pytest.param("kia_19mph.mp4", 18.5, 20.0, 20, id="19mph"),
]




def _gated(settings, mode):
    return dataclasses.replace(settings, motion_gate=mode)


async def test_motion_gate_off_by_default(db, settings):
    v = FakeVision({1: track(10, 960)})
    await run(db, settings, v, cal())
    assert v.gates == ["gate"] and v.span == (3.0, 4.5)


async def test_motion_gate_unknown_mode_is_off(db, settings):
    v = FakeVision({1: track(10, 960)})
    await run(db, _gated(settings, "maybe"), v, cal())
    assert v.gates == ["gate"]


async def test_motion_gate_on_tracks_its_span(db, settings):
    v = FakeVision({1: track(10, 960)})
    r = await run(db, _gated(settings, "on"), v, cal())
    assert v.gates == ["motion"] and v.span == (3.5, 4.0)
    assert r["passes"] == 1


async def test_motion_gate_on_no_vehicle_when_it_sees_none(db, settings):
    v = FakeVision({1: track(10, 960)}, motion=False)
    r = await run(db, _gated(settings, "on"), v, cal())
    assert r["status"] == "no_vehicle" and v.gates == ["motion"]


async def test_motion_gate_shadow_tracks_todays_span_and_logs_both(db, settings, caplog):
    v = FakeVision({1: track(10, 960)})
    with caplog.at_level(logging.INFO, logger="safe_streets.pipeline"):
        r = await run(db, _gated(settings, "shadow"), v, cal())
    assert r["passes"] == 1
    assert v.gates == ["gate", "motion"] and v.span == (3.0, 4.5)
    line = next(m for m in (rec.getMessage() for rec in caplog.records) if m.startswith("gate shadow"))
    assert line.startswith("gate shadow w1 2026-11-02T11:21:00.000-08:00 agree=True ")
    for part in ("span=3.00-4.50", "motion_span=3.50-4.00", "inferences=9", "motion_inferences=2",
                 "moving=2", "motion_moving=2", "motion_ms=", "motion_cpu_ms=", "kept=False"):
        assert f" {part}" in line, (part, line)
    assert not (settings.data_dir / "shadow").exists()  # they agree: no clip kept


@pytest.mark.parametrize("moving,motion,status", [(False, True, "no_vehicle"), (True, False, "done")])
async def test_motion_gate_shadow_keeps_the_clip_when_the_gates_disagree(db, settings, caplog, moving,
                                                                         motion, status):
    v = FakeVision({1: track(10, 960)}, moving=moving, motion=motion)
    with caplog.at_level(logging.INFO, logger="safe_streets.pipeline"):
        r = await run(db, _gated(settings, "shadow"), v, cal())
    assert r["status"] == status  # today's gate decides
    assert (settings.data_dir / "shadow" / "w1.mp4").read_bytes() == b"mp4"
    assert "agree=False" in caplog.text and "kept=True" in caplog.text
    if status == "no_vehicle":
        assert not (settings.data_dir / "clips" / "w1.mp4").exists()  # the event's own copy still goes


async def test_motion_gate_shadow_failure_never_fails_the_event(db, settings, caplog):
    v = FakeVision({1: track(10, 960)})

    def boom(*a):
        raise RuntimeError("boom")

    v.motion_gate = boom
    r = await run(db, _gated(settings, "shadow"), v, cal())
    assert r["status"] == "done" and r["passes"] == 1
    assert "gate shadow failed for w1" in caplog.text


def _episodes(settings, pads, **kw):
    return dataclasses.replace(settings, episodes="on", track_pad_before_s=pads[0], track_pad_after_s=pads[1], **kw)


async def test_episodes_off_by_default_track_the_span(db, settings):
    v = FakeVision({1: track(10, 960)})
    await run(db, settings, v, cal())
    assert v.kw.get("intervals") is None and v.span == (3.0, 4.5)


async def test_episodes_on_track_around_the_gates_moving_samples(db, settings, caplog):
    v = FakeVision({1: track(10, 960)})
    with caplog.at_level(logging.INFO, logger="safe_streets.pipeline"):
        r = await run(db, _episodes(settings, (0.5, 0.5)), v, cal())
    assert r["passes"] == 1
    assert v.kw["intervals"] == [(2.5, 3.5), (4.0, 5.0)]  # today's gate: moving at 3.0 and 4.5
    line = next(m for m in (rec.getMessage() for rec in caplog.records) if m.startswith("event w1"))
    assert " episodes=2 " in line


async def test_episodes_on_use_the_motion_gates_samples(db, settings):
    v = FakeVision({1: track(10, 960)})
    await run(db, _episodes(settings, (1.0, 0.5), motion_gate="on"), v, cal())
    assert v.kw["intervals"] == [(2.5, 4.5)]  # moving at 3.5 and 4.0: one episode


async def test_episodes_on_the_wide_crop_too(db, settings):
    v = CropVision({}, {1: track(10, 960)})
    await run(db, _episodes(settings, (1.0, 0.5)), v, cal(road=NARROW))
    assert v.intervals == [[(2.0, 5.0)]] * 2  # moving at 3.0 and 4.5; touching, so one


async def test_episodes_on_without_moving_samples_track_the_span(db, settings):
    """A gate that reports a span but no moving samples (none do) is
    tracked as before, not skipped."""
    v = FakeVision({1: track(10, 960)})
    v.gate = lambda clip, crop, road: Gate((3.0, 4.5), [i / 20 for i in range(200)])
    r = await run(db, _episodes(settings, (1.0, 0.5)), v, cal())
    assert r["passes"] == 1 and v.kw.get("intervals") is None and v.span == (3.0, 4.5)


async def test_motion_gate_on_hands_its_night_blobs_to_the_night_path(db, settings):
    """Step 5: one decode per event. The motion gate takes the headlight
    blobs from its own decode, and the night path uses them."""
    v = FakeVision({}, motion=False)
    v.gate_lights = [(0.0, [])]
    await run(db, _gated(settings, "on"), v, cal())
    assert v.motion_kw == {"lights": True}
    assert v.night_called and v.night_kw["frames"] is v.gate_lights


async def test_motion_gate_on_asks_for_no_blobs_without_a_night_path(db, settings):
    v = FakeVision({}, motion=False)
    await run(db, dataclasses.replace(_gated(settings, "on"), night_lights="off"), v, cal())
    assert v.motion_kw == {"lights": False}
    v = FakeVision({}, motion=False)
    await run(db, _gated(settings, "on"), v, cal(status="stale"), ev=await event(db, id="w2"))
    assert v.motion_kw == {"lights": False}


async def test_night_path_decodes_as_before_without_the_gates_blobs(db, settings):
    v = FakeVision({}, motion=False)  # e.g. a 640x360 export: the gate collects none
    await run(db, _gated(settings, "on"), v, cal())
    assert v.night_called and v.night_kw == {}




async def test_each_stage_logs_and_stores_its_gpu_time(db, settings, caplog, monkeypatch):
    # gpu.busy_ns: the process's GPU engine time; each reading 2 ms after the last.
    caplog.set_level(logging.INFO)
    clock = iter(range(0, 10**12, 2_000_000))
    monkeypatch.setattr(pipeline.gpu, "busy_ns", lambda: next(clock))
    await run(db, settings, FakeVision({1: track(10, 960)}), cal())
    [done] = [m for m in caplog.messages if m.startswith("event w1 ") and " done " in m]
    kv = dict(f.split("=", 1) for f in done.split() if "=" in f)
    assert kv["gate_gpu_ms"] == "2" and kv["track_gpu_ms"] == "2"
    assert list(kv)[-1] == "frames"
    stored = json.loads((await db.get_event("w1"))["stages_json"])
    assert stored["gate_gpu_ms"] == 2 and stored["track_gpu_ms"] == 2


async def test_no_gpu_time_without_a_gpu(db, settings, caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(pipeline.gpu, "busy_ns", lambda: None)
    await run(db, settings, FakeVision({1: track(10, 960)}), cal())
    [done] = [m for m in caplog.messages if m.startswith("event w1 ") and " done " in m]
    assert "_gpu_ms" not in done


async def test_the_night_path_and_the_shadow_gate_log_gpu_time(db, settings, settings_on, caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    clock = iter(range(0, 10**12, 3_000_000))
    monkeypatch.setattr(pipeline.gpu, "busy_ns", lambda: next(clock))
    await run(db, settings_on, FakeVision(moving=False), night_cal())
    [none] = [m for m in caplog.messages if m.startswith("event w1 ") and " no_vehicle " in m]
    kv = dict(f.split("=", 1) for f in none.split() if "=" in f)
    assert kv["night_gpu_ms"] == "3"
    await run(db, _gated(settings, "shadow"), FakeVision({1: track(10, 960)}), cal(),
              ev=await event(db, "w2", T0 + timedelta(minutes=5)))
    line = next(m for m in caplog.messages if m.startswith("gate shadow w2 "))
    assert " motion_gpu_ms=3 " in line


async def test_the_shadow_gate_without_a_gpu(db, settings, caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(pipeline.gpu, "busy_ns", lambda: None)
    await run(db, _gated(settings, "shadow"), FakeVision({1: track(10, 960)}), cal())
    line = next(m for m in caplog.messages if m.startswith("gate shadow w1 "))
    assert " motion_gpu_ms=- " in line


class FakeVehicles:
    def __init__(self, fail=False):
        self.ids, self.fail = [], fail

    async def classify_new(self, ids):
        self.ids.append(list(ids))
        if self.fail:
            raise RuntimeError("classifier bug")
        return {"classify_ms": 650, "classify_cpu_ms": 630}




