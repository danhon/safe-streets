from datetime import datetime, timezone

from safe_streets.sun import elevation

PORTLAND = (45.52, -122.68)


def at(*t):
    return datetime(*t, tzinfo=timezone.utc)


def test_noon_at_the_june_solstice():
    # 90 - 45.52 + 23.44
    assert abs(elevation(at(2026, 6, 21, 20, 12), *PORTLAND) - 67.9) < 0.3


def test_midnight_is_far_below():
    assert elevation(at(2026, 11, 7, 8, 0), *PORTLAND) < -50


def test_sunrise_crosses_zero():
    # Sunrise on 2026-11-07 is about 06:59 PST (14:59 UTC).
    assert elevation(at(2026, 11, 7, 14, 50), *PORTLAND) < 0 < elevation(at(2026, 11, 7, 15, 15), *PORTLAND)
