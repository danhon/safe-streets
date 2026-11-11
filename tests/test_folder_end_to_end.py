"""Three synthetic segments in a folder, one batch: the batcher scans the
recording for road motion, exports the window and finds one pass at the
true speed."""

import dataclasses
import os
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from safe_streets.batcher import Batcher
from safe_streets.sources.folder import FolderSource
from safe_streets.speed import Sample
from safe_streets.synth import read_pts, render_clip, uneven_times
from safe_streets.vision import Gate, TrackResult, fps_from
from listing import first_page

MPH20 = 8.9408
TZ = ZoneInfo("America/Los_Angeles")
T = datetime(2026, 11, 2, 11, 21, 0, tzinfo=TZ)


class TruthVision:
    """Sees the scene's true boxes. The clip's frame times are mapped to the
    scene's clock through the clip's wall start, which the test learns from
    the source (the batcher passes nothing else)."""

    def __init__(self, scene, rec_start_pts, rec_start_wall, source):
        self.sc, self.p0, self.w0, self.source = scene, rec_start_pts, rec_start_wall, source
        self.exports = []

    def _scene_t(self, clip_start, t):
        return self.p0 + (clip_start - self.w0).total_seconds() + t

    def gate(self, clip, crop, road):
        times = read_pts(clip)
        cs = self.source.last_start
        seen = [t for t in times if self.sc.in_view(self._scene_t(cs, t))]
        return Gate((seen[0], seen[-1]) if seen else None, times, seen, 0)

    motion_gate = None

    def track(self, clip, crop, road, span=None, times=None, **kw):
        times = read_pts(clip)
        cs = self.source.last_start
        samples = []
        for t in times:
            st = self._scene_t(cs, t)
            if self.sc.in_view(st):
                x, y = self.sc.wheel_point(st)
                samples.append(Sample(t=t, x=x, y=y, box=self.sc.box(st)))
        return TrackResult({1: samples}, fps_from(times), lambda t: None, {1: "car"}, False, None, len(samples))

    def night_lights(self, clip, **kw):
        return None


class RecordingFolder(FolderSource):
    """A FolderSource that remembers the wall start of its last export."""

    async def export(self, start, end, dest, *, low_res=False):
        got = await super().export(start, end, dest, low_res=low_res)
        self.last_start = got or start
        return got


async def test_three_segments_one_batch_one_pass(tmp_path, settings, db):
    d = tmp_path / "segments"
    d.mkdir()
    s = dataclasses.replace(settings, source="folder", source_dir=d, source_settle_s=5.0, lookback_hours=1.0,
                            night_lights="off")
    # One recording, 36 s long, the car crossing mid-way, cut into three 12 s
    # segments by ffmpeg's segment muxer keeping the running pts, as the
    # bundled recorder does (-reset_timestamps 0), and named by start time.
    times = uneven_times(36 * 12, 12, 0.25, seed=7, start=1000.0)
    whole = tmp_path / "whole.mp4"
    scene = render_clip(whole, speed_mps=MPH20, frame_times=times, gop=1)
    for path, first in _segments(whole, tmp_path / "cut", [1012, 1024]):
        dest = d / (T + timedelta(seconds=first - times[0])).strftime("%Y%m%d-%H%M%S.mp4")
        path.rename(dest)
        old = time.time() - 600
        os.utime(dest, (old, old))

    await db.save_calibration(image_pts=[], world_pts=[], H=scene.H.tolist(), road=scene.road,
                              reference_path="", status="active")
    src = RecordingFolder(s)
    vision = TruthVision(scene, times[0], T.astimezone(timezone.utc), src)
    rec_end = T + timedelta(seconds=times[-1] - times[0])
    b = Batcher(s, db, src, vision, load_fn=lambda: (0.1, 0, 0), clock=lambda: rec_end.timestamp() + 60)
    summary = await b.run_once(manual=True)
    assert summary.get("error") is None, summary
    rows, n = await first_page(db)
    assert n == 1, summary
    assert rows[0]["mph"] == pytest.approx(20.0, abs=0.5)


def _segments(whole, out, at):
    """Cut `whole` at these pts (seconds) with ffmpeg's segment muxer, keeping
    its pts. Returns (path, first pts) for each segment, in order."""
    import subprocess
    out.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-copyts", "-i", str(whole), "-c", "copy", "-f", "segment",
                    "-segment_times", ",".join(map(str, at)), "-reset_timestamps", "0", str(out / "s%d.mp4")],
                   check=True)
    res = []
    for p in sorted(out.iterdir()):
        first = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "stream=start_time", "-of", "csv=p=0", str(p)], capture_output=True, text=True,
                               check=True).stdout
        res.append((p, float(first)))
    return res


async def test_a_car_still_crossing_at_the_recordings_end_waits(tmp_path, settings, db):
    d = tmp_path / "segments"
    d.mkdir()
    s = dataclasses.replace(settings, source="folder", source_dir=d, source_settle_s=5.0, lookback_hours=1.0,
                            night_lights="off")
    # 36 s rendered, but only the first 18 s recorded so far: the car is mid-crossing.
    times = uneven_times(36 * 12, 12, 0.25, seed=7, start=1000.0)
    whole = tmp_path / "whole.mp4"
    scene = render_clip(whole, speed_mps=MPH20, frame_times=times, gop=1)
    for path, first in _segments(whole, tmp_path / "cut", [1009, 1018])[:2]:
        dest = d / (T + timedelta(seconds=first - times[0])).strftime("%Y%m%d-%H%M%S.mp4")
        path.rename(dest)
        old = time.time() - 600
        os.utime(dest, (old, old))
    await db.save_calibration(image_pts=[], world_pts=[], H=scene.H.tolist(), road=scene.road,
                              reference_path="", status="active")
    src = RecordingFolder(s)
    b = Batcher(s, db, src, TruthVision(scene, times[0], T.astimezone(timezone.utc), src),
                load_fn=lambda: (0.1, 0, 0), clock=lambda: T.timestamp() + 600)
    assert await b.queue_events() == 0
    cursor = datetime.fromisoformat(await db.get_state("scan_cursor"))
    assert cursor < T + timedelta(seconds=16)  # before the car's window: the next batch sees it whole
