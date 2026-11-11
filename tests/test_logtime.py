from datetime import datetime, timezone

from safe_streets.logtime import local, one_line

LA = "America/Los_Angeles"


def test_utc_instant_is_logged_in_local_time_with_its_offset():
    t = datetime(2026, 11, 7, 5, 30, 40, 892000, tzinfo=timezone.utc)
    assert local(t, LA) == "2026-11-06T21:30:40.892-08:00"


def test_db_strings_are_converted_too():
    assert local("2026-11-07T05:30:40.892000+00:00", LA) == "2026-11-06T21:30:40.892-08:00"
    assert local("2026-11-07T05:30:40Z", LA) == "2026-11-06T21:30:40.000-08:00"


def test_summer_time_gets_its_own_offset():
    assert local("2026-07-01T19:00:00+00:00", LA) == "2026-07-01T12:00:00.000-07:00"


def test_a_value_that_isnt_a_time_is_logged_as_it_is():
    # a log line must never raise
    assert local("soon", LA) == "soon"
    assert local(None, LA) == "None"


def test_text_is_put_on_one_line_with_its_breaks_shown():
    assert one_line("HEY.\n\nAt roughly 27 mph.\nSee more.\n") == "HEY. ⏎ At roughly 27 mph. ⏎ See more."
