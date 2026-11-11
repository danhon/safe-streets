"""The folder source: a directory of recorded segments (any NVR or the
bundled ffmpeg recorder), named by their start time."""

import dataclasses
import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import pytest

from safe_streets.config import DEFAULT_SOURCE_PATTERN
from safe_streets.sources import NoFootage, make_source
from safe_streets.sources.folder import FolderSource
from safe_streets.synth import read_pts, render_clip, uneven_times

TZ = ZoneInfo("America/Los_Angeles")
T = datetime(2026, 11, 2, 11, 21, 0, tzinfo=TZ)  # the first segment's start, local


@pytest.fixture
def folder(tmp_path, settings):
    d = tmp_path / "segments"
    d.mkdir()
    s = dataclasses.replace(settings, source="folder", source_dir=d, source_settle_s=10.0)
    return d, s


def segment(d, start: datetime, times, *, continuing=True, age_s=60.0, **kw):
    """A segment named by its local start time and offset (the default). continuing: its pts carry on
    from the recording's clock (ffmpeg's -reset_timestamps 0), else from 0."""
    path = d / start.strftime(DEFAULT_SOURCE_PATTERN)
    render_clip(path, speed_mps=8.94, frame_times=times, **kw)
    old = time.time() - age_s
    os.utime(path, (old, old))
    return path


def two_segments(d, *, continuing=True):
    ta = uneven_times(60, 10, 0.3, seed=1, start=500.0 if continuing else 0.0)
    gap = ta[-1] - ta[0] + 0.1
    tb = uneven_times(60, 10, 0.3, seed=2, start=(ta[-1] + 0.1) if continuing else 0.0)
    segment(d, T, ta, bframes=True)
    segment(d, T + timedelta(seconds=gap), tb, bframes=True)
    # Each frame's time from the recording's start. Segments whose pts start
    # again from 0 are placed by their file names (to the second).
    wall_b0 = (tb[0] - ta[0]) if continuing else float(int(gap))
    return [t - ta[0] for t in ta] + [wall_b0 + t - tb[0] for t in tb]


@pytest.mark.parametrize("continuing", [True, False])
async def test_a_window_across_two_segments_is_one_clip_with_their_times(folder, tmp_path, continuing):
    d, s = folder
    wall = two_segments(d, continuing=continuing)
    src = FolderSource(s)
    start, end = T + timedelta(seconds=2), T + timedelta(seconds=9)
    out = tmp_path / "clip.mp4"
    clip_start = await src.export(start, end, out)
    got = read_pts(out)
    # The clip starts at a keyframe at or before `start`: here the first segment's first frame.
    assert clip_start == T
    want = [w for w in wall if w <= 9.0 + 1.0]
    assert got[:len(want)] == pytest.approx(want, abs=1e-6)
    assert got[-1] >= 9.0 - 0.2  # covers the window's end


async def test_a_window_in_a_gap_raises_no_footage(folder, tmp_path):
    d, s = folder
    segment(d, T, uneven_times(30, 10, 0.2, seed=1))
    with pytest.raises(NoFootage):
        await FolderSource(s).export(T + timedelta(minutes=5), T + timedelta(minutes=6), tmp_path / "x.mp4")


async def test_skips_a_growing_file(folder, tmp_path):
    """Review Focus 3: a segment still being written is invisible."""
    d, s = folder
    done = segment(d, T, uneven_times(30, 10, 0.2, seed=1))
    segment(d, T + timedelta(seconds=3), uneven_times(30, 10, 0.2, seed=2), age_s=1.0)
    src = FolderSource(s)
    until = await src.recorded_until()
    assert until is not None and until < T + timedelta(seconds=3.5)
    with pytest.raises(NoFootage):
        await src.export(T + timedelta(seconds=4), T + timedelta(seconds=5), tmp_path / "x.mp4")
    assert done.exists()


async def test_recorded_until_is_the_end_of_the_newest_complete_segment(folder):
    d, s = folder
    times = uneven_times(30, 10, 0.2, seed=1)
    segment(d, T, times)
    until = await FolderSource(s).recorded_until()
    assert until == pytest.approx(T + timedelta(seconds=times[-1] - times[0]), abs=timedelta(seconds=0.2))


async def test_an_empty_folder_has_nothing_recorded(folder):
    assert await FolderSource(folder[1]).recorded_until() is None


async def test_low_res_is_the_same_clip(folder, tmp_path):
    d, s = folder
    segment(d, T, uneven_times(30, 10, 0.2, seed=1))
    src = FolderSource(s)
    await src.export(T, T + timedelta(seconds=2), tmp_path / "a.mp4")
    await src.export(T, T + timedelta(seconds=2), tmp_path / "b.mp4", low_res=True)
    assert read_pts(tmp_path / "a.mp4") == read_pts(tmp_path / "b.mp4")


async def test_snapshot_is_the_newest_segments_last_frame(folder):
    d, s = folder
    segment(d, T, uneven_times(30, 10, 0.2, seed=1))
    jpg = await FolderSource(s).snapshot()
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (1080, 1920, 3)


async def test_connected_only_when_the_folder_exists(folder, tmp_path, settings):
    d, s = folder
    assert await FolderSource(s).ensure_connected()
    missing = FolderSource(dataclasses.replace(s, source_dir=tmp_path / "nope"))
    assert not await missing.ensure_connected()
    assert "nope" in missing.last_error


def test_make_source_picks_by_setting(folder):
    assert isinstance(make_source(folder[1]), FolderSource)


async def test_a_clip_starts_at_the_keyframe_at_or_before_the_window(folder, tmp_path):
    d, s = folder
    times = uneven_times(60, 10, 0.2, seed=3, start=200.0)
    segment(d, T, times, gop=10, bframes=True)
    src = FolderSource(s)
    want_start = T + timedelta(seconds=times[20] - times[0])  # frames 20, 30, ... are keyframes
    clip_start = await src.export(T + timedelta(seconds=times[25] - times[0]), T + timedelta(seconds=5),
                                  tmp_path / "c.mp4")
    assert clip_start == want_start
    got = read_pts(tmp_path / "c.mp4")
    assert got[:30] == pytest.approx([t - times[20] for t in times[20:50]], abs=1e-6)


async def test_a_hung_ffmpeg_times_out(folder, tmp_path, monkeypatch):
    """A stale network mount can hang ffprobe/ffmpeg; the export fails after
    EXPORT_TIMEOUT_SECONDS instead of stalling the batcher for good."""
    d, s = folder
    two_segments(d)
    src = FolderSource(dataclasses.replace(s, export_timeout_seconds=1.0))
    assert src.segments()  # probed for real first
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for name in ("ffmpeg", "ffprobe"):
        (bin_ / name).write_text("#!/bin/sh\nexec sleep 30\n")
        (bin_ / name).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_}:{os.environ['PATH']}")
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out"):
        await src.export(T + timedelta(seconds=2), T + timedelta(seconds=9), tmp_path / "clip.mp4")
    assert time.monotonic() - t0 < 10


def test_the_hour_the_clocks_go_back_names_two_segments_apart(folder):
    """At the autumn change 01:30 happens twice. The default names carry the
    offset, so the second hour neither overwrites the first nor reads as it."""
    d, s = folder
    first = datetime(2026, 11, 1, 1, 30, tzinfo=TZ)            # PDT, -0700
    second = datetime(2026, 11, 1, 1, 30, fold=1, tzinfo=TZ)   # PST, -0800, an hour later
    paths = []
    for start in (first, second):
        path = d / start.strftime(s.source_pattern)
        assert not path.exists(), "the second hour's name is the first's"
        render_clip(path, speed_mps=8.94, frame_times=uneven_times(20, 10, 0.3, seed=1))
        os.utime(path, (time.time() - 60,) * 2)
        paths.append(path)
    a, b = FolderSource(s).segments()
    assert b.start - a.start == timedelta(hours=1)
    assert a.start == first.astimezone(ZoneInfo("UTC"))
