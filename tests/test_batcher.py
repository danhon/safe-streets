import asyncio
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest

from safe_streets.batcher import Batcher, cpu_pressure
from safe_streets.vision import Gate
from listing import first_page


@pytest.fixture
def settings(settings):
    """These tests pace by CPU pressure as the reference install does
    (MAX_CPU_PRESSURE=60); the default is no pacing (test_config)."""
    import dataclasses
    return dataclasses.replace(settings, max_cpu_pressure=60.0)

NOW = datetime(2026, 11, 2, 20, 0, tzinfo=timezone.utc).timestamp()


class FakeProtect:
    def __init__(self, ok=True, gate=None, events=None):
        self.ok = ok
        self.gate = gate
        self.events = events if events is not None else []
        self.asked = None
        self.logins = 0
        self.exports = []
        self.connected = False
        self.last_error = None
        self.snap = None

    async def ensure_connected(self):
        self.logins += 1
        self.connected = self.ok
        self.last_error = None if self.ok else "401"
        return self.ok

    async def export(self, start, end, dest, *, low_res=False):
        self.exports.append((start, end, low_res))
        if self.gate:
            await self.gate.wait()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")

    async def snapshot(self):
        return self.snap

    async def list_events(self, since, until):
        self.asked = (since, until)
        return self.events


class FakeVision:
    def gate(self, *a):
        return Gate(None, [])


@pytest.fixture(autouse=True)
async def calibrated(db):
    """Events wait without a calibration: these tests have one."""
    road = [(0, 400), (1920, 400), (1920, 600), (0, 600)]
    await db.save_calibration(image_pts=[], world_pts=[], H=np.diag([0.05, 0.05, 1.0]).tolist(), road=road,
                              reference_path="", status="active")


def make(settings, db, protect=None, load=0.5, now=None):
    clock = now or [NOW]
    b = Batcher(settings, db, protect or FakeProtect(), FakeVision(), load_fn=lambda: (load, 0, 0),
                clock=lambda: clock[0])
    return b, clock


async def test_trigger_while_running_is_ignored(settings, db):
    gate = asyncio.Event()
    start = datetime.fromtimestamp(NOW - 600, timezone.utc)
    p = FakeProtect(gate=gate, events=[("e1", start, start + timedelta(seconds=5))])
    b, _ = make(settings, db, p)
    task = asyncio.create_task(b.run_once(manual=True))
    for _ in range(50):
        await asyncio.sleep(0.01)
        if p.exports:
            break
    assert b.running
    assert b.trigger() is False
    gate.set()
    await task
    assert not b.running


async def test_load_skip_only_when_not_manual(settings, db):
    b, _ = make(settings, db, load=9.0)
    assert (await b.run_once(manual=False))["skipped"] == "load"
    assert "skipped" not in await b.run_once(manual=True)


async def test_runs_anyway_after_30_busy_minutes(settings, db):
    """Load holds work back for MAX_SKIPS intervals (~30 min), then a batch
    runs anyway, so a busy box can't hold the queue back for ever."""
    b, clock = make(settings, db, load=9.0)
    patience = settings.max_skips * settings.batch_interval_seconds
    results = []
    for _ in range(int(patience / settings.load_retry_seconds)):
        results.append(await b.run_once(manual=False))
        clock[0] += settings.load_retry_seconds
    assert all(r.get("skipped") == "load" for r in results)
    assert "skipped" not in await b.run_once(manual=False)
    # ...and the patience starts again after that batch:
    assert (await b.run_once(manual=False)).get("skipped") == "load"


async def test_login_failure_recorded(settings, db):
    p = FakeProtect(ok=False)
    b, _ = make(settings, db, p)
    r = await b.run_once(manual=False)
    assert r["error"] == "401"
    assert (await db.get_state("last_run"))["error"] == "401"


async def test_events_queued_with_widened_window(settings, db):
    ev_start = datetime.fromtimestamp(NOW - 600, timezone.utc)
    p = FakeProtect(events=[("e1", ev_start, ev_start + timedelta(seconds=8))])
    b, _ = make(settings, db, p)
    await b.run_once(manual=False)
    row = await db.get_event("e1")
    assert datetime.fromisoformat(row["start_ts"]) == ev_start - timedelta(seconds=5)
    assert datetime.fromisoformat(row["end_ts"]) == ev_start + timedelta(seconds=13)


async def test_window_capped_at_60s(settings, db):
    ev_start = datetime.fromtimestamp(NOW - 600, timezone.utc)
    p = FakeProtect(events=[("long", ev_start, ev_start + timedelta(seconds=300))])
    b, _ = make(settings, db, p)
    await b.run_once(manual=False)
    row = await db.get_event("long")
    span = datetime.fromisoformat(row["end_ts"]) - datetime.fromisoformat(row["start_ts"])
    assert span.total_seconds() == 60


async def test_event_cursor_from_lookback_then_overlaps(settings, db):
    p = FakeProtect(events=[])
    b, clock = make(settings, db, p)
    await b.run_once(manual=False)
    assert p.asked[0] == datetime.fromtimestamp(NOW - settings.lookback_hours * 3600, timezone.utc)
    clock[0] += 300
    await b.run_once(manual=False)
    assert p.asked[0] == datetime.fromtimestamp(NOW - 300, timezone.utc)  # last "now" minus 5 min overlap


async def test_retention_deletes_old_clips_keeps_rows(settings, db):
    clip = settings.data_dir / "clips" / "old.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"x")
    old = datetime.now(timezone.utc) - timedelta(days=31)
    await db.upsert_event("old", old.isoformat(), old.isoformat())
    await db.set_event("old", status="done", clip_path=str(clip))
    old_hero = settings.data_dir / "heroes" / "old.jpg"
    new_hero = settings.data_dir / "heroes" / "new.jpg"
    old_hero.parent.mkdir(parents=True)
    old_hero.write_bytes(b"x")
    new_hero.write_bytes(b"x")
    old_card = settings.data_dir / "cards" / "old.jpg"
    old_card.parent.mkdir(parents=True)
    old_card.write_bytes(b"x")
    old_id = await db.add_pass(event_id="old", ts=old.isoformat(), cls="car", direction=1, mph=20,
                               plus_minus=1, hero_path=str(old_hero), card_path=str(old_card))
    new_id = await db.add_pass(event_id="new", ts=datetime.now(timezone.utc).isoformat(), cls="car",
                               direction=1, mph=20, plus_minus=1, hero_path=str(new_hero))
    b, _ = make(settings, db)
    await b.prune_clips()
    assert not clip.exists()
    assert (await db.get_event("old"))["clip_path"] is None
    assert (await first_page(db))[1] == 2
    assert not old_hero.exists() and (await db.get_pass(old_id))["hero_path"] is None
    assert not old_card.exists() and (await db.get_pass(old_id))["card_path"] is None
    assert new_hero.exists() and (await db.get_pass(new_id))["hero_path"] == str(new_hero)


async def test_camera_moved_marks_stale_and_recovers(settings, db, tmp_path):
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur((rng.random((1080, 1920)) * 255).astype(np.uint8), (3, 3), 0)
    ref = tmp_path / "ref.jpg"
    cv2.imwrite(str(ref), base)
    cid = await db.save_calibration(image_pts=[], world_pts=[], H=np.eye(3).tolist(),
                                    road=[[870, 457], [1920, 465], [1920, 566], [870, 503]],
                                    reference_path=str(ref), status="provisional")
    p = FakeProtect()
    p.snap = cv2.imencode(".jpg", np.roll(base, 20, axis=1))[1].tobytes()
    b, clock = make(settings, db, p)
    await b.check_camera()
    assert (await db.active_calibration())["status"] == "stale"
    p.snap = cv2.imencode(".jpg", base)[1].tobytes()
    clock[0] += settings.camera_check_seconds + 1
    await b.check_camera()
    assert (await db.active_calibration())["status"] == "provisional"


def test_log_handlers_write_a_file(tmp_path):
    import logging

    from safe_streets.main import log_handlers

    handlers, problem = log_handlers(tmp_path / "logs", 30)
    assert problem == ""
    rec = logging.LogRecord("x", logging.INFO, __file__, 1, "hello", None, None)
    for h in handlers:
        h.setFormatter(logging.Formatter("%(message)s"))
        h.emit(rec)
        h.flush()
    assert "hello" in (tmp_path / "logs" / "app.log").read_text()
    assert log_handlers(None, 30)[0] and log_handlers(None, 30)[1] == ""


async def test_long_batch_polls_for_new_events_and_does_them_first(settings, db):
    """A batch that runs past the interval (a backlog) looks for new motion
    events between events, and the newest go first."""
    clock = [NOW]
    old = datetime.fromtimestamp(NOW - 7200, timezone.utc)
    for i in range(3):
        await db.upsert_event(f"old{i}", (old + timedelta(minutes=i)).isoformat(),
                              (old + timedelta(minutes=i, seconds=10)).isoformat())
    await db.set_state("event_cursor", datetime.fromtimestamp(NOW, timezone.utc).isoformat())

    class SlowProtect(FakeProtect):
        async def export(self, start, end, dest, *, low_res=False):
            clock[0] += 200  # each event takes 200 s
            await super().export(start, end, dest, low_res=low_res)

        async def list_events(self, since, until):
            self.polls = getattr(self, "polls", 0) + 1
            if self.polls == 2:  # a car passes while the backlog is being worked through
                s = datetime.fromtimestamp(clock[0] - 30, timezone.utc)
                return [("live", s, s + timedelta(seconds=10))]
            return []

    p = SlowProtect()
    b, _ = make(settings, db, p, now=clock)
    await b.run_once(manual=True)
    assert p.polls >= 2
    done = await db._all("SELECT id, status FROM events")
    assert all(r["status"] == "no_vehicle" for r in done)
    # "live" was queued mid-batch and done before the remaining backlog:
    assert p.exports.index(next(e for e in p.exports if e[0] > datetime.fromtimestamp(NOW, timezone.utc))) < 3


def test_camera_mask_is_scaled_to_the_snapshot():
    """Protect returns 640x360 snapshots; the road outline is in 1080p px."""
    from safe_streets.batcher import camera_mask

    road = [[870, 457], [1920, 465], [1920, 615], [870, 600]]
    m = camera_mask((360, 640), road)
    assert m[180, 500] == 0  # the road, at 1/3 scale (x 290-640, y ~73-215)
    assert m[300, 100] == 255  # the lawn in front of the porch
    assert m[5, 100] == 0  # the timestamp overlay
    assert camera_mask((1080, 1920), road)[540, 1500] == 0


async def test_covered_event_is_not_queued_partial_overlap_is(settings, db):
    """Triage step 4: an event whose window an existing event already
    contains isn't queued; one that only partly overlaps still is."""
    t = datetime.fromtimestamp(NOW - 600, timezone.utc)
    await db.upsert_event("wide", (t - timedelta(seconds=20)).isoformat(), (t + timedelta(seconds=40)).isoformat())
    p = FakeProtect(events=[
        ("inside", t, t + timedelta(seconds=10)),                               # window t-5 .. t+15: covered
        ("partly", t + timedelta(seconds=30), t + timedelta(seconds=40)),       # window t+25 .. t+45: sticks out
    ])
    b, _ = make(settings, db, p)
    assert await b.queue_events() == 1
    assert await db.get_event("inside") is None
    assert await db.get_event("partly") is not None


class Load:
    """A load average the test can change between events."""
    def __init__(self, v):
        self.v = v

    def __call__(self):
        return (self.v, 0, 0)


async def _backlog(db, n):
    old = datetime.fromtimestamp(NOW - 7200, timezone.utc)
    for i in range(n):
        await db.upsert_event(f"old{i}", (old + timedelta(minutes=i)).isoformat(),
                              (old + timedelta(minutes=i, seconds=10)).isoformat())
    await db.set_state("event_cursor", datetime.fromtimestamp(NOW, timezone.utc).isoformat())


async def test_batch_pauses_between_events_when_load_rises(settings, db):
    """The load is checked before every event, not once per batch: when the
    box gets busy mid-batch, the batch stops after the event in hand."""
    await _backlog(db, 5)
    load = Load(1.0)

    class BusyAfterOne(FakeProtect):
        async def export(self, start, end, dest, *, low_res=False):
            load.v = 9.0  # something else starts using the box
            await super().export(start, end, dest, low_res=low_res)

    p = BusyAfterOne()
    b = Batcher(settings, db, p, FakeVision(), load_fn=load, clock=lambda: NOW)
    r = await b.run_once(manual=False)
    assert r["processed"] == 1 and r["paused"] == "load" and r["queued"] == 4


async def test_manual_batch_ignores_load_between_events(settings, db):
    await _backlog(db, 3)
    b, _ = make(settings, db, load=9.0)
    r = await b.run_once(manual=True)
    assert r["processed"] == 3 and "paused" not in r


async def test_next_run_follows_the_backlog(settings, db):
    """With a backlog and a quiet box the next batch starts at once; a busy
    box retries in LOAD_RETRY_SECONDS; with nothing queued, the interval."""
    b, _ = make(settings, db)
    assert b._delay({"processed": 30, "queued": 200}) == 0
    assert b._delay({"processed": 3, "queued": 200, "paused": "load"}) == settings.load_retry_seconds
    assert b._delay({"skipped": "load", "queued": 200}) == settings.load_retry_seconds
    assert b._delay({"skipped": "load", "queued": 0}) == settings.batch_interval_seconds
    assert b._delay({"processed": 2, "queued": 0}) == settings.batch_interval_seconds
    assert b._delay({"error": "401"}) == settings.batch_interval_seconds
    assert b._delay({}) == settings.batch_interval_seconds


def hooked(monkeypatch, db, settings, eid, status="done"):
    """A Batcher with a fake process_event that marks the event `status`."""
    calls = []

    async def fake(ev, **kw):
        await db.set_event(ev["id"], status=status)
        return {"status": status, "passes": 1}

    async def hook(event_id):
        calls.append(event_id)

    monkeypatch.setattr("safe_streets.batcher.process_event", fake)
    b, _ = make(settings, db)
    b.on_first_analysis = hook
    return b, calls


async def queue(db, eid):
    start = datetime.fromtimestamp(NOW - 600, timezone.utc)
    await db.upsert_event(eid, start.isoformat(), (start + timedelta(seconds=5)).isoformat())


async def test_hook_runs_on_first_analysis_only(settings, db, monkeypatch):
    b, calls = hooked(monkeypatch, db, settings, "e1")
    await queue(db, "e1")
    assert (await b.run_once(manual=True))["processed"] == 1
    assert calls == ["e1"]
    assert (await db.get_event("e1"))["first_done_at"] is not None
    await db.requeue_since("2000-01-01T00:00:00Z")
    assert (await b.run_once(manual=True))["processed"] == 1
    assert calls == ["e1"]


async def test_hook_skips_scan_events(settings, db, monkeypatch):
    b, calls = hooked(monkeypatch, db, settings, "scan-1")
    await queue(db, "scan-1")
    await b.run_once(manual=True)
    assert calls == []
    assert (await db.get_event("scan-1"))["first_done_at"] is not None


async def test_hook_skips_no_vehicle_but_marks_it(settings, db, monkeypatch):
    b, calls = hooked(monkeypatch, db, settings, "e1", status="no_vehicle")
    await queue(db, "e1")
    await b.run_once(manual=True)
    assert calls == []
    assert (await db.get_event("e1"))["first_done_at"] is not None


async def test_failed_event_is_not_marked(settings, db, monkeypatch):
    b, calls = hooked(monkeypatch, db, settings, "e1", status="failed")
    await queue(db, "e1")
    await b.run_once(manual=True)
    assert calls == []
    assert (await db.get_event("e1"))["first_done_at"] is None


async def test_hook_errors_dont_stop_the_batch(settings, db, monkeypatch):
    b, _ = hooked(monkeypatch, db, settings, "e1")

    async def boom(event_id):
        raise RuntimeError("bsky down")

    b.on_first_analysis = boom
    await queue(db, "e1")
    assert (await b.run_once(manual=True))["processed"] == 1


async def test_batch_end_flushes_held_posts_and_reanalyses_flush_first(settings, db, monkeypatch):
    """Live cars are held by the poster until the batch ends, or until the
    batch reaches a reanalysis, so a long backlog doesn't hold them."""
    b, _ = hooked(monkeypatch, db, settings, "new")
    seen = []

    async def hook(event_id):
        seen.append(event_id)

    b.on_first_analysis = hook
    b.on_batch_end = lambda: seen.append("flush")
    start = datetime.fromtimestamp(NOW - 300, timezone.utc)
    await db.upsert_event("new", start.isoformat(), (start + timedelta(seconds=5)).isoformat())
    await queue(db, "re")
    await db.set_event("re", first_done_at="2026-11-04T00:00:00+00:00")
    assert (await b.run_once(manual=True))["processed"] == 2
    assert seen == ["new", "flush", "flush"]


async def test_batch_end_flush_runs_when_the_batch_fails(settings, db, monkeypatch):
    b, _ = hooked(monkeypatch, db, settings, "e1")
    seen = []
    b.on_batch_end = lambda: seen.append("flush")

    async def boom(*a, **kw):
        raise RuntimeError("protect down")

    monkeypatch.setattr(b, "queue_events", boom)
    with pytest.raises(RuntimeError):
        await b.run_once(manual=True)
    assert seen == ["flush"]


async def test_recent_live_events_are_analysed_oldest_first(settings, db, monkeypatch):
    order = []

    async def fake(ev, **kw):
        order.append(ev["id"])
        await db.set_event(ev["id"], status="no_vehicle")
        return {"status": "no_vehicle", "passes": 0}

    monkeypatch.setattr("safe_streets.batcher.process_event", fake)
    b, _ = make(settings, db)
    for i, eid in enumerate(["a", "b", "c"]):
        s = datetime.fromtimestamp(NOW - 600 + 60 * i, timezone.utc)
        await db.upsert_event(eid, s.isoformat(), (s + timedelta(seconds=5)).isoformat())
    await b.run_once(manual=True)
    assert order == ["a", "b", "c"]


PSI = """some avg10=21.99 avg60=23.15 avg300=22.72 total=25753386319
full avg10=0.00 avg60=0.00 avg300=0.00 total=0
"""


def test_cpu_pressure_reads_some_avg60(tmp_path):
    f = tmp_path / "cpu"
    f.write_text(PSI)
    assert cpu_pressure(f) == 23.15


@pytest.mark.parametrize("text", [None, "", "garbage\n", "some avg10=1.0\n", "some avg60=x\n"])
def test_cpu_pressure_is_none_when_unreadable(tmp_path, text):
    f = tmp_path / "cpu"
    if text is not None:
        f.write_text(text)
    assert cpu_pressure(f) is None


async def test_cpu_pressure_overrides_a_high_load(settings, db):
    """A load of 9 with low CPU pressure is I/O and GPU waits, not a busy CPU
    (2026-11-06): work goes ahead."""
    await _backlog(db, 2)
    b = Batcher(settings, db, FakeProtect(), FakeVision(), load_fn=lambda: (9.0, 0, 0),
                clock=lambda: NOW, pressure_fn=lambda: 25.0)
    r = await b.run_once(manual=False)
    assert r["processed"] == 2 and "skipped" not in r


async def test_high_cpu_pressure_skips_and_reports_it(settings, db):
    await _backlog(db, 2)
    b = Batcher(settings, db, FakeProtect(), FakeVision(), load_fn=lambda: (0.5, 0, 0),
                clock=lambda: NOW, pressure_fn=lambda: settings.max_cpu_pressure + 1)
    r = await b.run_once(manual=False)
    assert r["skipped"] == "load" and r["cpu"] == settings.max_cpu_pressure + 1 and "load" not in r


async def test_pressure_rising_mid_batch_pauses_it(settings, db):
    await _backlog(db, 5)
    cpu = [10.0]

    class BusyAfterOne(FakeProtect):
        async def export(self, start, end, dest, *, low_res=False):
            cpu[0] = 95.0
            await super().export(start, end, dest, low_res=low_res)

    b = Batcher(settings, db, BusyAfterOne(), FakeVision(), load_fn=lambda: (0.5, 0, 0),
                clock=lambda: NOW, pressure_fn=lambda: cpu[0])
    r = await b.run_once(manual=False)
    assert r["processed"] == 1 and r["paused"] == "load" and r["cpu"] == 95.0


async def test_retention_prunes_old_shadow_clips(settings, db):
    """MOTION_GATE=shadow keeps a clip where the gates disagree
    (data/shadow); those go after CLIP_RETENTION_DAYS, by their mtime."""
    import os
    import time

    shadow = settings.data_dir / "shadow"
    shadow.mkdir()
    old, new = shadow / "old.mp4", shadow / "new.mp4"
    old.write_bytes(b"x")
    new.write_bytes(b"x")
    t = time.time() - (settings.clip_retention_days + 1) * 86400
    os.utime(old, (t, t))
    b, _ = make(settings, db, now=[time.time()])
    await b.prune_clips()
    assert not old.exists() and new.exists()


class FakeVehicles:
    def __init__(self, log):
        self.log, self.calls = log, 0

    async def catch_up(self, busy):
        self.calls += 1
        self.log.append("catch_up")
        return {"vehicle_classified": 2, "vehicle_skipped": 0, "classify_ms": 1300, "classify_cpu_ms": 1260}






async def test_no_catch_up_when_paused_by_load(settings, db):
    await _backlog(db, 3)
    load = Load(1.0)

    class BusyAfterOne(FakeProtect):
        async def export(self, start, end, dest, *, low_res=False):
            load.v = 9.0
            await super().export(start, end, dest, low_res=low_res)

    b = Batcher(settings, db, BusyAfterOne(), FakeVision(), load_fn=load, clock=lambda: NOW)
    b.vehicle = FakeVehicles([])
    r = await b.run_once(manual=False)
    assert r["paused"] == "load" and b.vehicle.calls == 0


async def test_a_catch_up_failure_never_fails_the_batch(settings, db):
    class Broken:
        async def catch_up(self, busy):
            raise RuntimeError("classifier bug")

    b, _ = make(settings, db)
    b.vehicle = Broken()
    r = await b.run_once(manual=False)
    assert "error" not in r and "processed" in r




class RecordingVehicles:
    def __init__(self):
        self.busy = None

    async def catch_up(self, busy):
        self.busy = busy()
        return {}




# Staleness (2026-11-08): after the 11-04 reboot no batch finished for ~36 h
# and nothing said so. age() feeds /healthz's batch_stale.


async def test_age_counts_from_the_last_finished_batch(settings, db):
    b, clock = make(settings, db)
    await b.run_once(manual=True)
    clock[0] += 90
    assert b.age() == pytest.approx(90)


async def test_age_after_a_restart_counts_from_the_start(settings, db):
    b, clock = make(settings, db)
    # The loop loads the last batch from the DB: a day old after a long outage.
    b.last_run = {"at": datetime.fromtimestamp(NOW - 86400, timezone.utc).isoformat()}
    clock[0] += 30
    assert b.age() == pytest.approx(30)


async def test_a_batch_skipped_as_busy_counts_as_finished(settings, db):
    b, clock = make(settings, db, load=9.0)
    clock[0] += 600
    assert (await b.run_once(manual=False))["skipped"] == "load"
    assert b.age() == pytest.approx(0)
