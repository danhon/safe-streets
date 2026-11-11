"""The only source of public data (Tier A of the privacy spike).

Public templates and JSON get nothing but what this module returns:

- times floored to PUBLIC_TIME_BUCKET_MIN (15) in local time, and a
  bucket appears only once all of it is PUBLIC_DELAY_HOURS (24) old, so
  polling can't time a pass more finely than its bucket;
- a time bucket of fewer than PUBLIC_MIN_GROUP (5) passes counts in the
  day's totals and its hour's count only: its passes aren't listed and no
  speed figure includes them, so none can be worked out by subtraction;
- an hour's speed figures (over, fastest) appear only once the whole hour
  is published and it has PUBLIC_MIN_GROUP listed passes;
- every stored pass is counted, night and rough ones included (marked
  rough); a pass without a speed is counted and not listed;
- speeds rounded, in UNITS; the kind only as car or two-wheeler;
- no pass ids, event ids, exact times, pictures, clips, makes or models.

It never reads who's signed in: it takes no request.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from .kinds import TWO_WHEEL_CLS
from .speed import ROUGH, whole

HOURS = 24


@dataclass(frozen=True)
class PublicPass:
    bucket: datetime  # local, floored to PUBLIC_TIME_BUCKET_MIN
    speed: int  # rounded, in the configured unit
    rough: bool
    direction: int
    kind: str  # "car" | "two-wheeler"


@dataclass(frozen=True)
class PublicHour:
    hour: int
    n: int  # every pass, listed or not
    over: int | None  # of the listed passes; None until the hour is whole
    fastest: int | None  # and has PUBLIC_MIN_GROUP listed passes


@dataclass(frozen=True)
class PublicDay:
    day: date
    unit: str  # "mph" | "km/h"
    limit: float
    bucket_min: int
    published_until: datetime | None  # local; None when the whole day is published
    total: int  # every pass up to published_until
    over: int  # of the listed passes, over the limit
    unlisted: int  # passes in buckets too small to list
    passes: tuple[PublicPass, ...]  # the listed ones, oldest first
    hours: tuple[PublicHour, ...]  # 24, midnight first

    def as_json(self) -> dict:
        """What the zoo and day dial read: each pass as its quarter of the
        day (bucket index), speed, rough, direction and kind."""
        return {"date": self.day.isoformat(), "unit": self.unit, "limit": self.limit,
                "bucket_min": self.bucket_min, "total": self.total, "over": self.over,
                "unlisted": self.unlisted,
                "passes": [{"q": (p.bucket.hour * 60 + p.bucket.minute) // self.bucket_min, "speed": p.speed,
                            "rough": p.rough, "dir": p.direction, "kind": p.kind} for p in self.passes],
                "hours": [{"h": h.hour, "n": h.n, "over": h.over, "fastest": h.fastest} for h in self.hours]}


def cutoff(settings, now: datetime) -> datetime:
    """Nothing at or after this is public: now less the delay, floored to the
    bucket (in local time), so a bucket is published whole or not at all."""
    tz = ZoneInfo(settings.tz)
    return _floor((now - timedelta(hours=settings.public_delay_hours)).astimezone(tz),
                  max(1, int(settings.public_time_bucket_min)))


def latest_day(settings, now: datetime) -> date:
    """The newest day with anything published: the one the cutoff falls in."""
    return cutoff(settings, now).astimezone(ZoneInfo(settings.tz)).date()


def _floor(d: datetime, minutes: int) -> datetime:
    m = (d.hour * 60 + d.minute) // minutes * minutes
    return d.replace(hour=m // 60, minute=m % 60, second=0, microsecond=0)


async def public_day(db, day: date, settings, now: datetime) -> PublicDay:
    tz = ZoneInfo(settings.tz)
    start = datetime.combine(day, time(0), tz)
    end = min(datetime.combine(day + timedelta(days=1), time(0), tz), cutoff(settings, now))
    rows = []
    if end > start:
        rows = await db.public_rows(start.astimezone(timezone.utc).isoformat(),
                                    end.astimezone(timezone.utc).isoformat())
    bucket_min = max(1, int(settings.public_time_bucket_min))
    k = max(1, int(settings.public_min_group))
    limit = float(settings.speed_limit or 0)

    every: list[PublicPass | None] = []
    speeds: list[int | None] = []
    for r in rows:
        local = datetime.fromisoformat(r["ts"].replace("Z", "+00:00")).astimezone(tz)
        flags = [f for f in (r["flags"] or "").split(",") if f]
        two = r["cls"] in TWO_WHEEL_CLS and "headlights" not in flags
        v = settings.speed(r["mps"])
        speeds.append(None if v is None else whole(round(v, 6)))  # m/s and back: 22.5 stays 22.5
        every.append(PublicPass(bucket=_floor(local, bucket_min), speed=speeds[-1] if v is not None else -1,
                                rough=any(f in ROUGH for f in flags), direction=1 if r["direction"] >= 0 else -1,
                                kind="two-wheeler" if two else "car"))
    per_bucket: dict[datetime, int] = {}
    for p in every:
        per_bucket[p.bucket] = per_bucket.get(p.bucket, 0) + 1
    listed = tuple(p for p in every if per_bucket[p.bucket] >= k and p.speed >= 0)

    hours = []
    for h in range(HOURS):
        n = sum(1 for p in every if p.bucket.hour == h)
        shown = [p.speed for p in listed if p.bucket.hour == h]
        whole_hour = datetime.combine(day, time(h), tz) + timedelta(hours=1) <= end
        if whole_hour and len(shown) >= k:
            hours.append(PublicHour(h, n, sum(1 for s in shown if s > limit), max(shown)))
        else:
            hours.append(PublicHour(h, n, None, None))
    whole_day = end >= datetime.combine(day + timedelta(days=1), time(0), tz)
    return PublicDay(
        day=day, unit=settings.unit_label, limit=limit, bucket_min=bucket_min,
        published_until=None if whole_day else end.astimezone(tz),
        total=len(every), over=sum(1 for p in listed if p.speed > limit),
        unlisted=sum(1 for p in every if per_bucket[p.bucket] < k),
        passes=listed, hours=tuple(hours))
