"""The pipeline on synthetic clips with known speeds: real exported files,
real frame times (OpenCV's, as production reads them), and at night the
real headlight tracker. By day the detector is the scene's true boxes: the
default detector comes in with the Detector interface (Task 8)."""

from datetime import datetime, timedelta, timezone

import pytest

from safe_streets.geometry import Calibration
from safe_streets.pipeline import process_event
from safe_streets.speed import Sample, confidence
from safe_streets.synth import read_pts, render_clip, uneven_times
from safe_streets.vision import Gate, TrackResult, Vision, fps_from
from listing import first_page

MPH20 = 8.9408
T0 = datetime(2026, 11, 2, 19, 21, 0, tzinfo=timezone.utc)


class SynthProtect:
    """Exports a synthetic clip rendered once per test."""

    def __init__(self, clip):
        self.clip = clip

    async def export(self, start, end, dest, *, low_res=False):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.clip.read_bytes())


class TruthVision:
    """Gate and tracker that see the scene's true boxes at the clip's own
    frame times; night_lights is the real one."""

    def __init__(self, scene, clip, day=True):
        self.sc, self.times, self.day = scene, read_pts(clip), day
        self.real = Vision(clip.parent / "models", "cpu", 0.35)

    def _seen(self):
        return [t for t in self.times if self.sc.in_view(t)] if self.day else []

    def gate(self, clip, crop, road):
        seen = self._seen()
        return Gate((seen[0], seen[-1]) if seen else None, self.times, seen, 0)

    motion_gate = None

    def track(self, clip, crop, road, span=None, times=None, **kw):
        samples = [Sample(t=t, x=self.sc.wheel_point(t)[0], y=self.sc.wheel_point(t)[1], box=self.sc.box(t))
                   for t in self._seen()]
        return TrackResult({1: samples}, fps_from(self.times), lambda t: None, {1: "car"}, False, None, len(samples))

    def night_lights(self, clip, **kw):
        return self.real.night_lights(clip, **kw)


def cal_for(sc):
    return Calibration(1, [], [], sc.H, sc.road, "", "active")


async def analyse(db, settings, tmp_path, *, fps=12.0, night=False, n=None, seed=2):
    times = uneven_times(n or int(fps * 6), fps, 0.3, seed=seed)
    clip = tmp_path / "src.mp4"
    sc = render_clip(clip, speed_mps=MPH20, frame_times=times, night=night)
    await db.upsert_event("w1", T0.isoformat(), (T0 + timedelta(seconds=10)).isoformat())
    r = await process_event(await db.get_event("w1"), source=SynthProtect(clip),
                            vision=TruthVision(sc, clip, day=not night), cal=cal_for(sc),
                            settings=settings, db=db)
    rows, _ = await first_page(db)
    return r, rows


async def test_a_20mph_day_car_reads_20_and_good(db, settings, tmp_path):
    r, rows = await analyse(db, settings, tmp_path)
    assert r["status"] == "done" and len(rows) == 1
    p = rows[0]
    assert p["mph"] == pytest.approx(20.0, abs=0.5)
    flags = [f for f in (p["flags"] or "").split(",") if f]
    assert confidence(flags, p["plus_minus"])[0] == "good"


@pytest.mark.parametrize("fps,low", [(9.0, False), (4.0, True)])
async def test_low_fps_is_flagged_only_below_min_fps(db, settings, tmp_path, fps, low):
    r, rows = await analyse(db, settings, tmp_path, fps=fps)
    assert len(rows) == 1
    assert ("low_fps" in (rows[0]["flags"] or "")) is low
    assert settings.min_fps == 5.0


async def test_a_night_car_is_found_by_its_lights_and_rough(db, settings, tmp_path):
    r, rows = await analyse(db, settings, tmp_path, night=True, fps=15.0)
    assert r["status"] == "done" and len(rows) == 1
    p = rows[0]
    assert "headlights" in p["flags"]
    flags = p["flags"].split(",")
    assert confidence(flags, p["plus_minus"])[0] == "rough"
    assert p["mph"] == pytest.approx(20.0, abs=0.5)

