"""What an install plugs in: hooks called with each batch's new passes, and
the detector behind the Detector protocol (RTMDet-tiny by default, through
the vendored MIT ByteTrack; ultralytics is never imported)."""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from safe_streets import batcher as batcher_mod
from safe_streets.batcher import Batcher
from safe_streets.detector import Detections
from safe_streets.geometry import Calibration
from safe_streets.hooks import PassRecord, load_hooks
from safe_streets.pipeline import process_event
from safe_streets.synth import render_clip, uneven_times
from safe_streets.vision import Vision
from listing import first_page
from test_synth_pipeline import MPH20, T0, SynthProtect

MODELS = Path(__file__).parent.parent / "models"
CALLS: list = []


class Recording:
    async def batch_done(self, passes):
        CALLS.append(("recording", [p.ts for p in passes], passes))


class Raising:
    async def batch_done(self, passes):
        raise RuntimeError("a broken hook")


def make_recording():
    return Recording()


@pytest.fixture(autouse=True)
def clear():
    CALLS.clear()


async def calibrate(db):
    await db.save_calibration(image_pts=[], world_pts=[], H=np.eye(3).tolist(),
                              road=[[0, 0], [1920, 0], [1920, 1080], [0, 1080]], reference_path="", status="active")


class Src:
    connected, last_error = True, None

    async def ensure_connected(self):
        return True

    async def list_events(self, since, until):
        return []


def fake_process(times_by_event):
    """process_event that stores one pass per given time, out of order."""
    async def run(ev, *, db, **kw):
        for ts in times_by_event[ev["id"]]:
            await db.add_pass(event_id=ev["id"], ts=ts.isoformat(), cls="car", direction=1, mph=22.0, plus_minus=1.0)
        await db.set_event(ev["id"], status="done")
        return {"status": "done", "passes": len(times_by_event[ev["id"]])}
    return run


async def run_batch(db, settings, hooks, monkeypatch):
    a = datetime(2026, 11, 10, 18, 0, tzinfo=timezone.utc)
    times = {"w2": [a + timedelta(minutes=5)], "w1": [a + timedelta(seconds=40), a + timedelta(seconds=10)]}
    for eid in ("w2", "w1"):
        t = times[eid][0]
        await db.upsert_event(eid, (t - timedelta(seconds=5)).isoformat(), (t + timedelta(seconds=5)).isoformat())
    monkeypatch.setattr(batcher_mod, "process_event", fake_process(times))
    b = Batcher(settings, db, Src(), None, load_fn=lambda: (0.1, 0, 0), clock=lambda: a.timestamp() + 600)
    b.hooks = hooks
    return await b.run_once(manual=True), sorted(t for ts in times.values() for t in ts)


async def test_a_hook_gets_the_batchs_passes_in_time_order(db, settings, monkeypatch):
    await calibrate(db)
    hooks = load_hooks("test_hooks:Recording, test_hooks:make_recording")
    assert len(hooks) == 2
    summary, want = await run_batch(db, settings, hooks, monkeypatch)
    assert summary["cars"] == 3
    assert len(CALLS) == 2
    _, ts, passes = CALLS[0]
    assert ts == want
    p = passes[0]
    assert isinstance(p, PassRecord) and p.mps == pytest.approx(22 * 0.44704) and p.event_id == "w1"


async def test_a_raising_hook_is_logged_and_the_next_still_runs(db, settings, monkeypatch, caplog):
    await calibrate(db)
    caplog.set_level(logging.ERROR)
    summary, _ = await run_batch(db, settings, [Raising(), Recording()], monkeypatch)
    assert summary.get("error") is None and summary["cars"] == 3
    assert len(CALLS) == 1
    assert "hook" in caplog.text and "RuntimeError" in caplog.text


async def test_no_hook_runs_for_a_batch_without_new_passes(db, settings, monkeypatch):
    await calibrate(db)
    b = Batcher(settings, db, Src(), None, load_fn=lambda: (0.1, 0, 0))
    b.hooks = [Recording()]
    await b.run_once(manual=True)
    assert CALLS == []


def test_a_bad_hook_spec_refuses_to_start():
    with pytest.raises(SystemExit, match="HOOKS"):
        load_hooks("no_such_module_here:thing")
    with pytest.raises(SystemExit, match="HOOKS"):
        load_hooks("test_hooks")


class ColourDetector:
    """Finds the synthetic car by its body colour, as a detector would by its
    shape: the box is the body's, its bottom where the wheels touch."""
    names = {2: "car"}

    def detect(self, image):
        mask = cv2.inRange(image, (40, 20, 120), (85, 65, 185))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        boxes = [(x, y, x + w, y + h * 8 / 7) for x, y, w, h, area in stats[1:] if area > 200]
        if not boxes:
            return Detections()
        return Detections(np.array(boxes, np.float32), np.full(len(boxes), 0.9, np.float32), np.full(len(boxes), 2))


async def test_a_detector_giving_the_true_box_gives_the_true_speed(db, settings, tmp_path):
    clip = tmp_path / "src.mp4"
    sc = render_clip(clip, speed_mps=MPH20, frame_times=uneven_times(84, 12, 0.3, seed=2))
    vision = Vision(tmp_path / "models", "cpu", 0.35, cache_dir=None, detector=ColourDetector())
    await db.upsert_event("w1", T0.isoformat(), (T0 + timedelta(seconds=10)).isoformat())
    r = await process_event(await db.get_event("w1"), source=SynthProtect(clip), vision=vision,
                            cal=Calibration(1, [], [], sc.H, sc.road, "", "active"), settings=settings, db=db)
    rows, n = await first_page(db)
    assert r["status"] == "done" and n == 1
    assert rows[0]["mph"] == pytest.approx(20.0, abs=0.5)


def test_importing_the_package_never_imports_ultralytics():
    code = ("import importlib, pkgutil, sys, safe_streets\n"
            "for m in pkgutil.walk_packages(safe_streets.__path__, 'safe_streets.'):\n"
            "    importlib.import_module(m.name)\n"
            "assert 'ultralytics' not in sys.modules, 'ultralytics was imported'\n"
            "print('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "SPEED_LIMIT": "20"})
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr[-2000:]
    src = Path(__file__).parent.parent / "src" / "safe_streets"
    import re
    assert not [p for p in src.rglob("*.py") if re.search(r"^\s*(import|from)\s+ultralytics", p.read_text(), re.M)]


@pytest.mark.skipif(os.environ.get("SAFE_STREETS_REAL_MODEL") != "1" or not (MODELS / "rtmdet-tiny").exists(),
                    reason="the real model: SAFE_STREETS_REAL_MODEL=1 and models/rtmdet-tiny")
async def test_the_default_detector_finds_the_synthetic_car(db, settings, tmp_path):
    clip = tmp_path / "src.mp4"
    sc = render_clip(clip, speed_mps=MPH20, frame_times=uneven_times(84, 12, 0.3, seed=2), photo=True)
    vision = Vision(MODELS, "cpu", 0.35, cache_dir=None)
    await db.upsert_event("w1", T0.isoformat(), (T0 + timedelta(seconds=10)).isoformat())
    await process_event(await db.get_event("w1"), source=SynthProtect(clip), vision=vision,
                        cal=Calibration(1, [], [], sc.H, sc.road, "", "active"), settings=settings, db=db)
    rows, n = await first_page(db)
    assert n == 1 and rows[0]["mph"] == pytest.approx(20.0, abs=1.0)
