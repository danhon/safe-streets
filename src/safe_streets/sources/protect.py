"""UniFi Protect through the unofficial `uiprotect` library.

The official Integration API has no clip export, so this uses the private API
with a local console user. It is undocumented and can break on a Protect
update. PROTECT_CAMERA names the camera.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

HIGH, LOW = 0, 2  # channel indexes: full resolution and the low-res stream (640x360 on a G3 Flex)


class Protect:
    def __init__(self, settings, client=None, clock=time.time, sleep=asyncio.sleep):
        self.s = settings
        self._client = client
        self.clock = clock
        self._sleep = sleep  # between export tries
        self.camera_id: str | None = None
        self.connected = False
        self.last_error: str | None = None
        self._last_attempt = float("-inf")

    @property
    def client(self):
        if self._client is None:
            from uiprotect import ProtectApiClient

            d = Path(self.s.data_dir) / "uiprotect"
            d.mkdir(parents=True, exist_ok=True)
            self._client = ProtectApiClient(
                self.s.protect_host, self.s.protect_port, self.s.protect_username, self.s.protect_password,
                verify_ssl=False,  # the console's self-signed cert; LAN hop only
                config_dir=d, cache_dir=d,
            )
        return self._client

    async def ensure_connected(self) -> bool:
        """Logs in if needed. A failed login isn't retried for
        login_retry_seconds, so a bad password can't lock out the console."""
        if self.connected:
            return True
        now = self.clock()
        if now - self._last_attempt < self.s.login_retry_seconds:
            return False
        self._last_attempt = now
        try:
            await asyncio.wait_for(self.client.update(), self.s.protect_timeout_seconds)
            cams = self.client.bootstrap.cameras.values()
            cam = next((c for c in cams if c.name.lower() == self.s.protect_camera.lower()), None)
            if cam is None:
                raise RuntimeError(f"no camera named {self.s.protect_camera!r}; have {sorted(c.name for c in cams)}")
            self.camera_id = cam.id
            self.connected = True
            self.last_error = None
            log.info("connected to Protect; camera %s is %s", self.s.protect_camera, cam.id)
        except Exception as e:  # noqa: BLE001  (uiprotect raises many unrelated types)
            self.connected = False
            self.last_error = f"{type(e).__name__}: {e}"[:300]
            log.warning("Protect login failed (next try in %ss): %s", self.s.login_retry_seconds, self.last_error)
        return self.connected

    async def list_events(self, since: datetime, until: datetime) -> list[tuple[str, datetime, datetime]]:
        """Ended motion events on our camera that started in [since, until]."""
        from uiprotect.data import EventType

        raw = await asyncio.wait_for(
            self.client.get_events_raw(start=since, end=until, limit=2000, types=[EventType.MOTION]),
            self.s.protect_timeout_seconds)
        out = []
        for r in raw:
            if r.get("camera") != self.camera_id or r.get("type") != "motion" or not r.get("end"):
                continue
            out.append((r["id"], datetime.fromtimestamp(r["start"] / 1000, timezone.utc),
                        datetime.fromtimestamp(r["end"] / 1000, timezone.utc)))
        return sorted(out, key=lambda e: e[1])

    async def recorded_until(self) -> datetime | None:
        """Protect records continuously; footage is exportable a little behind now."""
        return datetime.fromtimestamp(self.clock() - self.s.protect_lag_s, timezone.utc)

    async def export(self, start: datetime, end: datetime, dest: Path, *, low_res: bool = False) -> None:
        """Protect's clip starts at a keyframe up to a few seconds early, so
        the pipeline places it by its end (returns None)."""
        channel = LOW if low_res else HIGH
        dest.parent.mkdir(parents=True, exist_ok=True)
        last_err = None
        for attempt in range(3):
            try:
                await asyncio.wait_for(
                    self.client.get_camera_video(self.camera_id, start, end, channel_index=channel, output_file=dest),
                    self.s.export_timeout_seconds)
                if dest.exists() and dest.stat().st_size > 0:
                    return
                last_err = RuntimeError("empty export")
            except Exception as e:  # noqa: BLE001
                last_err = e
                if "401" in str(e) or "403" in str(e):
                    self.connected = False
            await self._sleep(3 * (attempt + 1))
        raise RuntimeError(f"export failed: {type(last_err).__name__}: {last_err}")

    async def snapshot(self) -> bytes:
        return await asyncio.wait_for(self.client.get_camera_snapshot(self.camera_id, 1920, 1080),
                                      self.s.protect_timeout_seconds)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close_session()
