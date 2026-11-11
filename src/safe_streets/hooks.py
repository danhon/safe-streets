"""Hooks: what an install adds without changing the code (a notifier, a
poster, an export). HOOKS is a comma-separated list of `module:attr` entry
points, loaded at start; each is an object with `async batch_done(passes)`,
or a callable returning one. After each batch, each hook gets that batch's
new passes, oldest first: passes from an event's first live analysis, not
reanalyses or backfills of old footage (scan-). A hook that raises, or
runs past HOOK_TIMEOUT_SECONDS (60) and is cancelled, is logged; the others
and the batch carry on. Do blocking work in asyncio.to_thread: a hook that
blocks the event loop can't be cancelled.

Hooks run in the app's process with the operator's data: makes and models
aside (there are none), a pass here has its exact time and its picture.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PassRecord:
    id: int
    event_id: str
    ts: datetime  # UTC, the car at mid-road
    mps: float | None  # the speed, m/s; None when there's none
    plus_minus_mps: float | None
    direction: int  # +1 or -1 along the road
    cls: str  # the detector's class (car, truck, bus, motorcycle, bicycle)
    flags: tuple[str, ...]
    rough: bool  # a caveat about the track or clip (speed.ROUGH)
    shot_path: str | None

    @classmethod
    def from_row(cls, r) -> "PassRecord":
        from .speed import ROUGH
        flags = tuple(f for f in (r["flags"] or "").split(",") if f)
        pm = r["plus_minus"]
        return cls(id=r["id"], event_id=r["event_id"], ts=datetime.fromisoformat(r["ts"]),
                   mps=r["mps"], plus_minus_mps=None if pm is None else pm * 0.44704, direction=r["direction"],
                   cls=r["cls"], flags=flags, rough=any(f in ROUGH for f in flags), shot_path=r["shot_path"])


class PassHook(Protocol):
    async def batch_done(self, passes: list[PassRecord]) -> None: ...


def load_hooks(spec: str) -> list[PassHook]:
    """HOOKS: "pkg.module:attr, other:attr". A bad entry stops the start."""
    hooks = []
    for entry in (e.strip() for e in (spec or "").split(",")):
        if not entry:
            continue
        mod, _, attr = entry.partition(":")
        try:
            if not attr:
                raise ValueError("needs module:attr")
            obj = getattr(importlib.import_module(mod), attr)
            hook = obj() if isinstance(obj, type) or (callable(obj) and not hasattr(obj, "batch_done")) else obj
            if not hasattr(hook, "batch_done"):
                raise TypeError("has no batch_done(passes)")
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"HOOKS entry {entry!r} can't be loaded: {e}")
        hooks.append(hook)
    return hooks


async def run_hooks(hooks: list[PassHook], passes: list[PassRecord], timeout: float = 60.0) -> None:
    """Each hook in turn, each given `timeout` seconds (HOOK_TIMEOUT_SECONDS)
    before it's cancelled, so a hook that hangs can't hold up the batches."""
    if not hooks or not passes:
        return
    passes = sorted(passes, key=lambda p: (p.ts, p.id))
    for hook in hooks:
        try:
            await asyncio.wait_for(hook.batch_done(passes), timeout)
        except TimeoutError:
            log.error("hook %s took over %g s and was cancelled", type(hook).__name__, timeout)
        except Exception:  # noqa: BLE001  (one broken hook mustn't stop the rest)
            log.exception("hook %s failed", type(hook).__name__)
