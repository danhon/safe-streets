from safe_streets.sources.protect import Protect


class FailingClient:
    def __init__(self):
        self.calls = 0

    async def update(self):
        self.calls += 1
        raise RuntimeError("401 bad password")


async def test_login_failure_not_retried_within_300s(settings):
    client = FailingClient()
    now = [1000.0]
    p = Protect(settings, client=client, clock=lambda: now[0])
    assert await p.ensure_connected() is False
    now[0] += 10
    assert await p.ensure_connected() is False
    assert client.calls == 1
    assert "401" in p.last_error
    now[0] += 300
    await p.ensure_connected()
    assert client.calls == 2


class EventsClient:
    def __init__(self, raw):
        self.raw = raw
        self.asked = None

    async def get_events_raw(self, start, end, limit, types):
        self.asked = (start, end)
        return self.raw


async def test_list_events_keeps_ended_motion_on_our_camera(settings):
    from datetime import datetime, timezone

    raw = [
        {"id": "a", "type": "motion", "camera": "cam", "start": 1793649600000, "end": 1793649607000},
        {"id": "b", "type": "motion", "camera": "cam", "start": 1793649700000, "end": None},  # still going
        {"id": "c", "type": "motion", "camera": "other", "start": 1793649600000, "end": 1793649605000},
    ]
    p = Protect(settings, client=EventsClient(raw))
    p.camera_id = "cam"
    p.connected = True
    since = datetime(2026, 11, 2, 8, 0, tzinfo=timezone.utc)
    evs = await p.list_events(since, datetime(2026, 11, 2, 20, 0, tzinfo=timezone.utc))
    assert [e[0] for e in evs] == ["a"]
    assert evs[0][1] == datetime.fromtimestamp(1793649600, timezone.utc)
    assert (evs[0][2] - evs[0][1]).total_seconds() == 7


# Timeouts (2026-11-08). After the 11-04 reboot the batcher went silent for
# ~36 h with /healthz ok: a Protect call that never returned. uiprotect
# downloads exports with no timeout at all (timeout=0), so each call has one.

import asyncio  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402

import pytest  # noqa: E402

GUARD_S = 2  # the test's own limit: a call that hangs fails here, not forever


async def within(coro):
    """Awaits coro, failing the test if it runs past GUARD_S. A TimeoutError
    from the code under test passes through: the guard's own fails."""
    try:
        async with asyncio.timeout(GUARD_S) as guard:
            return await coro
    except TimeoutError:
        if guard.expired():
            pytest.fail(f"still waiting after {GUARD_S} s: no timeout")
        raise


async def _hang(*a, **k):
    await asyncio.Event().wait()


class HangingClient:
    """Each call hangs while `hangs` > 0 (counting down), then succeeds."""

    def __init__(self, hangs=10**6):
        self.hangs = hangs
        self.calls = 0

    async def _maybe_hang(self):
        self.calls += 1
        if self.hangs > 0:
            self.hangs -= 1
            await _hang()

    async def update(self):
        await self._maybe_hang()

    async def get_events_raw(self, **k):
        await self._maybe_hang()
        return []

    async def get_camera_snapshot(self, *a):
        await self._maybe_hang()
        return b"jpg"

    async def get_camera_video(self, camera_id, start, end, channel_index, output_file):
        await self._maybe_hang()
        output_file.write_bytes(b"mp4")


def fast(settings):
    from dataclasses import replace

    return replace(settings, protect_timeout_seconds=0.05, export_timeout_seconds=0.05)


async def no_sleep(_s):
    pass


async def test_a_login_that_hangs_times_out(settings):
    p = Protect(fast(settings), client=HangingClient())
    assert await within(p.ensure_connected()) is False
    assert "Timeout" in p.last_error


async def test_an_event_list_that_hangs_times_out(settings):
    p = Protect(fast(settings), client=HangingClient())
    p.camera_id = "cam"
    now = datetime(2026, 11, 8, tzinfo=timezone.utc)
    with pytest.raises(TimeoutError):
        await within(p.list_events(now - timedelta(minutes=5), now))


async def test_a_snapshot_that_hangs_times_out(settings):
    p = Protect(fast(settings), client=HangingClient())
    p.camera_id = "cam"
    with pytest.raises(TimeoutError):
        await within(p.snapshot())


async def test_an_export_that_hangs_is_tried_again(settings, tmp_path):
    client = HangingClient(hangs=1)
    p = Protect(fast(settings), client=client, sleep=no_sleep)
    p.camera_id = "cam"
    now = datetime(2026, 11, 8, tzinfo=timezone.utc)
    dest = tmp_path / "clips" / "e.mp4"
    await within(p.export(now - timedelta(seconds=20), now, dest))
    assert client.calls == 2 and dest.read_bytes() == b"mp4"


async def test_an_export_that_always_hangs_fails(settings, tmp_path):
    p = Protect(fast(settings), client=HangingClient(), sleep=no_sleep)
    p.camera_id = "cam"
    now = datetime(2026, 11, 8, tzinfo=timezone.utc)
    with pytest.raises(RuntimeError, match="export failed: TimeoutError"):
        await within(p.export(now - timedelta(seconds=20), now, tmp_path / "e.mp4"))
