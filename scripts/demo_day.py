"""Serve the public pages on a made-up day of passes (no footage, no real
data), for docs/img/public-day.png:

  uv run python scripts/demo_day.py /tmp/demo-data      # then open
  http://127.0.0.1:18090/?day=2026-11-03
"""
import asyncio, math, os, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import uvicorn

data = Path(sys.argv[1]); data.mkdir(parents=True, exist_ok=True)
os.environ.update(SPEED_LIMIT="20", TZ="Europe/London", UNITS="mph", DATA_DIR=str(data),
                  PUBLIC_DELAY_HOURS="0", AUTH_MODE="password")
from safe_streets.config import Settings
from safe_streets.db import Database
from safe_streets.web import create_app

DAY = datetime(2026, 11, 3, tzinfo=timezone.utc)  # a Tuesday, London = UTC in November

async def fill(db):
    rng = np.random.default_rng(3)
    # A weekday's shape: a morning and an afternoon peak, quiet at night.
    w = np.array([1, .5, .3, .3, .5, 1.5, 4, 9, 12, 8, 6, 6, 7, 7, 8, 11, 13, 12, 9, 6, 4, 3, 2, 1.5])
    for h, n in enumerate(rng.poisson(w * 2.2)):
        for _ in range(n):
            ts = DAY + timedelta(hours=h, seconds=float(rng.uniform(0, 3600)))
            mph = float(np.clip(rng.normal(19.5 if 6 <= h <= 21 else 23, 4.5), 6, 42))
            cls = "car" if rng.random() > .04 else rng.choice(["bicycle", "motorcycle"])
            flags = "short" if rng.random() < .12 else ""
            await db.add_pass(event_id=f"demo-{h}-{_}", ts=ts.isoformat(), cls=str(cls),
                              direction=int(rng.choice([1, -1], p=[.55, .45])), mph=mph,
                              plus_minus=float(rng.uniform(.3, 2.5)), flags=flags)

async def main():
    s = Settings()
    db = Database(s.db_path); await db.init()
    await fill(db)
    app = create_app(s, db, None, None, None, now=lambda: DAY + timedelta(days=2))
    await uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=18090, log_level="warning")).serve()

asyncio.run(main())
