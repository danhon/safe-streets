"""The scene lives in the calibration (lane, trusted region, night lane line,
masks, occluders, the snapshot's size), any clip size works, and a
calibration can be made from drive-bys at known speeds."""

import dataclasses
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from safe_streets.calibrate import DriveFit, scale_from_drives
from safe_streets.geometry import Calibration, to_working
from safe_streets.lights import night_cars
from safe_streets.scan import find_windows
from safe_streets.speed import Sample, confidence, fit_speed
from safe_streets.synth import BASE, LANE_Y, read_pts, render_clip, scene_for, uneven_times
from safe_streets.vision import Vision
from listing import first_page
from test_synth_pipeline import SynthProtect, TruthVision

MPH = 0.44704  # m/s


def drive(tmp_path, name, mph, seed):
    """A drive-by: its cleaned track (true boxes at the clip's real pts) and true speed in m/s."""
    times = uneven_times(80, 12, 0.3, seed=seed, start=2.5)
    clip = tmp_path / f"{name}.mp4"
    sc = render_clip(clip, speed_mps=mph * MPH, frame_times=times, bframes=True)
    t0 = times[0]
    samples = []
    for t in read_pts(clip):
        st = t + t0
        if sc.in_view(st):
            x, y = sc.wheel_point(st)
            samples.append(Sample(t=t, x=x, y=y, box=sc.box(st)))
    return sc, samples


def true_px_per_m(sc):
    a, b = sc._img(10.0, LANE_Y), sc._img(11.0, LANE_Y)
    return abs(b[0] - a[0])


def test_drive_by_scale_recovers_the_truth(tmp_path):
    sc, d1 = drive(tmp_path, "a", 15, 1)
    _, d2 = drive(tmp_path, "b", 15, 2)
    fit = scale_from_drives([(d1, 15 * MPH), (d2, 15 * MPH)])
    assert isinstance(fit, DriveFit)
    assert fit.px_per_m == pytest.approx(true_px_per_m(sc), rel=0.01)
    assert all(r == pytest.approx(15.0, abs=0.3) for r in fit.readings_mph)
    assert fit.spread_mph < 0.5
    _, held_out = drive(tmp_path, "c", 25, 3)
    assert fit_speed(held_out, fit.H).mph == pytest.approx(25.0, abs=0.5)


def test_one_drive_alone_cant_be_saved(tmp_path):
    _, d1 = drive(tmp_path, "a", 15, 1)
    with pytest.raises(ValueError, match="two"):
        scale_from_drives([(d1, 15 * MPH)])


def test_a_calibration_made_at_another_size_is_scaled_to_the_clip():
    sc = scene_for(BASE, 9.0, 1, 0.0)
    s = np.diag([1280 / 1920, 720 / 1080, 1.0])  # 1080p px -> 720p px
    H720 = sc.H @ np.linalg.inv(s)
    road720 = [(x * 1280 / 1920, y * 720 / 1080) for x, y in sc.road]
    cal = Calibration(1, [], [], H720, road720, "", "active", size=(1280, 720),
                      trusted=[(0, 0), (1280, 0), (1280, 720), (0, 720)])
    w = to_working(cal)
    assert np.allclose(w.road, sc.road)
    assert np.allclose(w.lane, sc.road)  # the lane defaults to the road
    assert np.allclose(w.trusted[2], (1920, 1080))
    p = sc.wheel_point(1.0)
    assert np.allclose(w.H @ [*p, 1] / (w.H @ [*p, 1])[2], sc.H @ [*p, 1] / (sc.H @ [*p, 1])[2])


@pytest.mark.parametrize("size,low_res", [((1280, 720), True), ((3840, 2160), False)])
async def test_pipeline_at_720p_and_4k(db, settings, tmp_path, size, low_res):
    """Review Focus 2: the night path decodes the clip itself, at any size,
    with a calibration made at 1920x1080."""
    times = uneven_times(90, 15, 0.3, seed=4)
    clip = tmp_path / "src.mp4"
    sc = render_clip(clip, speed_mps=20 * MPH, frame_times=times, night=True, size=size)
    ref = scene_for(BASE, 20 * MPH, 1, sc.t0)
    cal = Calibration(1, [], [], ref.H, ref.road, "", "active", size=BASE)
    t0 = datetime(2026, 11, 2, 19, 21, tzinfo=timezone.utc)
    await db.upsert_event("w1", t0.isoformat(), (t0 + timedelta(seconds=10)).isoformat())
    await __import__("safe_streets.pipeline", fromlist=["x"]).process_event(
        await db.get_event("w1"), source=SynthProtect(clip), vision=TruthVision(sc, clip, day=False),
        cal=cal, settings=settings, db=db)
    rows, n = await first_page(db)
    assert n == 1
    assert rows[0]["mph"] == pytest.approx(20.0, abs=0.5)
    assert ("low_res" in rows[0]["flags"]) is low_res


async def test_a_pass_outside_the_trusted_region_is_flagged_rough(db, settings, tmp_path):
    times = uneven_times(72, 12, 0.3, seed=2)
    clip = tmp_path / "src.mp4"
    sc = render_clip(clip, speed_mps=20 * MPH, frame_times=times)
    far_above = [(0, 0), (1920, 0), (1920, 100), (0, 100)]
    for trusted, flagged in ((far_above, True), (None, False)):
        await db.conn.execute("DELETE FROM passes")
        cal = Calibration(1, [], [], sc.H, sc.road, "", "active", trusted=trusted)
        await db.upsert_event("w1", "2026-11-02T19:21:00+00:00", "2026-11-02T19:21:10+00:00")
        await __import__("safe_streets.pipeline", fromlist=["x"]).process_event(
            await db.get_event("w1"), source=SynthProtect(clip), vision=TruthVision(sc, clip), cal=cal,
            settings=settings, db=db)
        rows, _ = await first_page(db)
        flags = (rows[0]["flags"] or "").split(",")
        assert ("outside_trusted" in flags) is flagged
        if flagged:
            assert confidence(flags, rows[0]["plus_minus"])[0] == "rough"


def test_motion_inside_a_mask_queues_nothing(tmp_path):
    times = uneven_times(36 * 10, 10, 0.2, seed=5)
    clip = tmp_path / "c.mp4"
    sc = render_clip(clip, speed_mps=20 * MPH, frame_times=times, size=(640, 360))
    road = scene_for(BASE, 1, 1, 0).road
    assert find_windows(clip, road, warmup_s=2.0)
    assert find_windows(clip, road, warmup_s=2.0, mask=[road]) == []


@pytest.mark.parametrize("night_lane,kept", [((425.0, 440.0), True), ((700.0, 700.0), False)])
def test_the_night_lane_line_comes_from_the_calibration(tmp_path, settings, night_lane, kept):
    times = uneven_times(90, 15, 0.3, seed=6)
    clip = tmp_path / "n.mp4"
    sc = render_clip(clip, speed_mps=20 * MPH, frame_times=times, night=True)
    frames, _ = Vision(tmp_path, "cpu", 0.35).night_lights(clip)
    cal = Calibration(1, [], [], sc.H, sc.road, "", "active", night_lane=night_lane)
    cars, _ = night_cars(frames, cal.road, cal.H, settings, night_lane=cal.night_lane)
    assert (len(cars) == 1) is kept


async def test_without_a_calibration_events_wait(db, settings):
    from safe_streets.batcher import Batcher

    class Src:
        connected, last_error = True, None

        async def ensure_connected(self):
            return True

        async def list_events(self, since, until):
            return []

        async def export(self, *a, **kw):
            raise AssertionError("nothing is exported without a calibration")

    await db.upsert_event("w1", "2026-11-02T19:21:00+00:00", "2026-11-02T19:21:10+00:00")
    b = Batcher(settings, db, Src(), None, load_fn=lambda: (0.1, 0, 0))
    summary = await b.run_once(manual=True)
    assert summary["processed"] == 0 and summary.get("uncalibrated") is True
    assert (await db.get_event("w1"))["status"] == "queued"


async def test_a_saved_calibration_keeps_its_scene(db):
    from safe_streets.geometry import calibration_from_row

    road = [(0, 400), (1920, 400), (1920, 600), (0, 600)]
    await db.save_calibration(image_pts=[], world_pts=[], H=np.eye(3).tolist(), road=road, reference_path="",
                              status="active", lane=[(0, 450), (1920, 450), (1920, 600), (0, 600)],
                              trusted=[(0, 500), (1920, 500), (1920, 560), (0, 560)], night_lane=(425, 440),
                              occluders=[1235.0], mask=[[(0, 0), (1920, 0), (1920, 90), (0, 90)]], size=(1920, 1080))
    cal = calibration_from_row(await db.active_calibration())
    assert cal.lane[0] == (0, 450) and cal.trusted[0] == (0, 500)
    assert cal.night_lane == (425, 440) and cal.occluders == [1235.0] and cal.size == (1920, 1080)
    assert cal.mask[0][2] == (1920, 90)
    await db.save_calibration(image_pts=[], world_pts=[], H=np.eye(3).tolist(), road=road, reference_path="",
                              status="active")
    bare = calibration_from_row(await db.active_calibration())
    assert bare.lane == road and bare.trusted is None and bare.mask == [] and bare.size == (1920, 1080)
