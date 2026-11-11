import sys
from pathlib import Path

import cv2
import numpy as np
import pytest



from safe_streets.speed import Sample, fit_speed  # noqa: E402
from safe_streets.config import Settings  # noqa: E402
from safe_streets.synth import read_pts, render_clip, uneven_times  # noqa: E402

MPH20 = 8.9408  # m/s


def samples(sc, times):
    out = []
    for t in times:
        if sc.in_view(t):
            x, y = sc.wheel_point(t)
            out.append(Sample(t=t, x=x, y=y, box=sc.box(t)))
    return out


def test_pts_are_exactly_the_given_times(tmp_path):
    times = uneven_times(40, 15, 0.3, seed=1)
    render_clip(tmp_path / "c.mp4", speed_mps=MPH20, frame_times=times)
    assert read_pts(tmp_path / "c.mp4") == pytest.approx(times, abs=1 / 90000)


def test_fit_recovers_the_true_speed(tmp_path):
    times = uneven_times(60, 12, 0.3, seed=2)
    sc = render_clip(tmp_path / "c.mp4", speed_mps=MPH20, frame_times=times)
    s = samples(sc, read_pts(tmp_path / "c.mp4"))
    assert len(s) >= 10
    fit = fit_speed(s, sc.H)
    assert fit.mph == pytest.approx(20.0, abs=0.05)
    assert fit.direction == 1


def test_uneven_pts_with_bframes(tmp_path):
    """Review Focus 1: B-frames (pts out of decode order), a clip starting
    at 3.7 s, a car going left."""
    times = uneven_times(60, 9, 0.4, seed=3, start=3.7)
    sc = render_clip(tmp_path / "c.mp4", speed_mps=MPH20, frame_times=times, bframes=True, direction=-1)
    got = read_pts(tmp_path / "c.mp4")
    # OpenCV counts from the clip's first frame, as the pipeline's sample times do.
    assert got == pytest.approx([t - times[0] for t in times], abs=1 / 90000)
    times_from_start = got
    fit = fit_speed(samples(sc, [t + times[0] for t in times_from_start]), sc.H)
    assert fit.mph == pytest.approx(20.0, abs=0.05)
    assert fit.direction == -1


@pytest.mark.parametrize("size", [(1280, 720), (640, 360), (3840, 2160)])
def test_other_sizes_keep_the_truth(tmp_path, size):
    times = uneven_times(30, 15, 0.2, seed=4)
    sc = render_clip(tmp_path / "c.mp4", speed_mps=MPH20, frame_times=times, size=size)
    cap = cv2.VideoCapture(str(tmp_path / "c.mp4"))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[1::-1] == size
    assert fit_speed(samples(sc, times), sc.H).mph == pytest.approx(20.0, abs=0.05)


def test_night_frames_are_grey_with_saturated_lights(tmp_path):
    times = uneven_times(30, 15, 0.2, seed=5)
    sc = render_clip(tmp_path / "c.mp4", speed_mps=MPH20, frame_times=times, night=True)
    cap = cv2.VideoCapture(str(tmp_path / "c.mp4"))
    mid = len(times) // 2
    for _ in range(mid + 1):
        ok, frame = cap.read()
    cap.release()
    assert ok
    b, g, r = (frame[..., i].astype(int) for i in range(3))
    assert np.abs(b - g).max() <= 2 and np.abs(g - r).max() <= 2
    x0, y0, x1, y1 = (int(v) for v in sc.box(times[mid]))
    assert frame[y0:y1 + 1, x0:x1 + 1].max() >= 245  # the lights
    assert frame[:y0 - 20].max() < 100  # nothing else is bright


def test_demo_writes_a_recent_segment_the_folder_source_reads(tmp_path, monkeypatch):
    import json
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace

    from safe_streets.sources.folder import FolderSource
    from safe_streets.synth import demo

    monkeypatch.setenv("TZ", "Europe/London")
    now = datetime(2026, 11, 10, 18, 0, tzinfo=timezone.utc)
    demo(tmp_path, now=now)
    segs = list(tmp_path.glob("*.mp4"))
    assert [p.name for p in segs] == ["20261110-175500+0000.mp4"]  # 5 min ago, London time (UTC in November)
    scene = json.loads((tmp_path / "scene.json").read_text())
    assert len(scene["H"]) == 3 and len(scene["road"]) == 4 and scene["truth_mph"] == pytest.approx(20.0)
    s = SimpleNamespace(source_dir=tmp_path, source_pattern=Settings().source_pattern, tz="Europe/London", source_settle_s=0,
                        export_timeout_seconds=120.0)
    seg, = FolderSource(s, clock=lambda: now.timestamp() + 3600).segments()
    assert seg.start == now - timedelta(minutes=5) and seg.end - seg.start > timedelta(seconds=15)
