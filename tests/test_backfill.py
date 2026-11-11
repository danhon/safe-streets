import asyncio
import dataclasses
import logging
from datetime import datetime, timedelta, timezone

import pytest

from safe_streets.backfill import chunks, resume_backfill, run_backfill

T0 = datetime(2026, 11, 2, 20, 0, tzinfo=timezone.utc)
ROAD = [(870, 457), (1920, 465), (1920, 615), (870, 600)]


def test_chunks_overlap_by_the_warmup():
    cs = list(chunks(T0, T0 + timedelta(minutes=12), chunk_s=300, warmup_s=10))
    # Each chunk starts warmup_s early, so motion during its warmup was in the previous one.
    assert cs[0] == (T0 - timedelta(seconds=10), T0 + timedelta(minutes=5))
    assert cs[1] == (T0 + timedelta(minutes=5) - timedelta(seconds=10), T0 + timedelta(minutes=10))
    assert cs[-1][1] == T0 + timedelta(minutes=12)
    assert len(cs) == 3


class FakeProtect:
    def __init__(self, connect_after=0):
        self.exports = []
        self.connect_after = connect_after  # failed ensure_connected calls before it connects

    async def ensure_connected(self):
        self.connect_after -= 1
        return self.connect_after < 0

    async def export(self, start, end, dest, *, low_res=False):
        self.exports.append((start, end, low_res))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"mp4")


@pytest.fixture
def windows_at(monkeypatch):
    """find_windows returns (seconds into the chunk) windows from this dict,
    keyed by chunk start."""
    table = {}

    def fake(clip, road, **kw):
        return table.get(clip.name, [])

    monkeypatch.setattr("safe_streets.backfill.find_windows", fake)
    return table


async def test_backfill_queues_windows_and_skips_covered_ones(db, settings, windows_at):
    p = FakeProtect()
    # An existing motion event already covers 20:01:00–20:01:30.
    await db.upsert_event("motion1", (T0 + timedelta(seconds=55)).isoformat(),
                          (T0 + timedelta(seconds=95)).isoformat())
    first = T0 - timedelta(seconds=10)
    name = f"scan-{first:%Y%m%dT%H%M%S}.mp4"
    windows_at[name] = [(70.0, 80.0), (200.0, 215.0)]  # 20:01:00–10 (covered), 20:03:10–25
    n = await run_backfill(settings, db, p, ROAD, T0, T0 + timedelta(minutes=5))
    assert n == 1
    ids = [r["id"] for r in await db.next_events(10)]
    assert "motion1" in ids and len(ids) == 2
    ev = await db.get_event(next(i for i in ids if i.startswith("scan-")))
    assert ev["start_ts"] == (first + timedelta(seconds=200)).isoformat()
    assert ev["end_ts"] == (first + timedelta(seconds=215)).isoformat()
    assert all(low for _, _, low in p.exports)  # the low-res stream
    assert not list((settings.data_dir / "scan").glob("*.mp4"))  # chunks deleted


async def test_backfill_is_idempotent(db, settings, windows_at):
    p = FakeProtect()
    first = T0 - timedelta(seconds=10)
    windows_at[f"scan-{first:%Y%m%dT%H%M%S}.mp4"] = [(100.0, 110.0)]
    assert await run_backfill(settings, db, p, ROAD, T0, T0 + timedelta(minutes=5)) == 1
    assert await run_backfill(settings, db, p, ROAD, T0, T0 + timedelta(minutes=5)) == 0


class DyingProtect(FakeProtect):
    """Exports die (as a deploy would kill the task) from the nth export on."""

    def __init__(self, die_at):
        super().__init__()
        self.die_at = die_at

    async def export(self, start, end, dest, *, low_res=False):
        if len(self.exports) == self.die_at:
            raise asyncio.CancelledError
        await super().export(start, end, dest, low_res=low_res)


async def test_backfill_records_how_far_it_got(db, settings, windows_at):
    # Killed during the third chunk: the first two are done.
    with pytest.raises(asyncio.CancelledError):
        await run_backfill(settings, db, DyingProtect(die_at=2), ROAD, T0, T0 + timedelta(minutes=15))
    assert await db.get_state("backfill") == {
        "start": T0.isoformat(), "end": (T0 + timedelta(minutes=15)).isoformat(),
        "done_to": (T0 + timedelta(minutes=10)).isoformat()}


async def test_finished_backfill_leaves_nothing_to_resume(db, settings, windows_at):
    await run_backfill(settings, db, FakeProtect(), ROAD, T0, T0 + timedelta(minutes=10))
    assert await db.get_state("backfill") is None
    assert await resume_backfill(settings, db, FakeProtect(), ROAD) is None


async def test_resume_scans_only_what_is_left(db, settings, windows_at):
    with pytest.raises(asyncio.CancelledError):
        await run_backfill(settings, db, DyingProtect(die_at=2), ROAD, T0, T0 + timedelta(minutes=15))
    last = T0 + timedelta(minutes=10) - timedelta(seconds=10)
    windows_at[f"scan-{last:%Y%m%dT%H%M%S}.mp4"] = [(100.0, 110.0)]
    p = FakeProtect()
    assert await resume_backfill(settings, db, p, ROAD) == 1
    assert [s for s, _, _ in p.exports] == [last]
    assert await db.get_state("backfill") is None


async def test_resume_waits_for_protect(db, settings, windows_at, monkeypatch):
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr("safe_streets.backfill.asyncio.sleep", fake_sleep)
    await db.set_state("backfill", {"start": T0.isoformat(), "end": (T0 + timedelta(minutes=5)).isoformat(),
                                    "done_to": T0.isoformat()})
    p = FakeProtect(connect_after=2)
    assert await resume_backfill(settings, db, p, ROAD) == 0
    assert len(sleeps) == 2 and len(p.exports) == 1


LA = "America/Los_Angeles"


class BrokenProtect(FakeProtect):
    async def export(self, start, end, dest, *, low_res=False):
        raise RuntimeError("no export")


async def test_backfill_logs_its_times_in_local_time(db, settings, windows_at, caplog):
    # T0 is 20:00 UTC, 12:00 Pacific; each chunk starts 10 s early.
    caplog.set_level(logging.INFO, "safe_streets.backfill")
    s = dataclasses.replace(settings, tz=LA)
    await run_backfill(s, db, FakeProtect(), ROAD, T0, T0 + timedelta(minutes=5))
    await run_backfill(s, db, BrokenProtect(), ROAD, T0, T0 + timedelta(minutes=5))
    assert "backfill chunk 2026-11-02T11:59:50.000-08:00 windows=0" in caplog.messages
    assert "backfill chunk 2026-11-02T11:59:50.000-08:00 failed" in caplog.messages
    assert ("backfill 2026-11-02T12:00:00.000-08:00 to 2026-11-02T12:05:00.000-08:00: "
            "queued=0 skipped_covered=0") in caplog.messages


async def test_backfill_resume_is_logged_in_local_time(db, settings, windows_at, caplog):
    caplog.set_level(logging.INFO, "safe_streets.backfill")
    await db.set_state("backfill", {"start": T0.isoformat(), "end": (T0 + timedelta(minutes=10)).isoformat(),
                                    "done_to": (T0 + timedelta(minutes=5)).isoformat()})
    await resume_backfill(dataclasses.replace(settings, tz=LA), db, FakeProtect(), ROAD)
    assert ("backfill resuming at 2026-11-02T12:05:00.000-08:00 (started from 2026-11-02T12:00:00.000-08:00) "
            "to 2026-11-02T12:10:00.000-08:00") in caplog.messages
