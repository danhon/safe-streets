import pytest

from safe_streets.histogram import BIN_MPH, histogram


def counts(h):
    return [(b["lo"], b["hi"], b["solid"], b["rough"]) for b in h["bins"] if b["solid"] or b["rough"]]


def test_the_limit_is_a_bin_edge_and_counts_as_under():
    h = histogram([(20.0, False), (20.1, False)], limit=20.0)
    assert counts(h) == [(17.5, 20.0, 1, 0), (20.0, 22.5, 1, 0)]
    assert (h["under"], h["over"]) == (1, 1)


def test_bins_run_from_zero_to_the_fastest_with_empty_bins_kept():
    h = histogram([(5.0, False), (31.0, False)], limit=20.0)
    assert h["bins"][0]["lo"] == 0.0 and h["bins"][-1]["hi"] == 32.5
    assert len(h["bins"]) == round(32.5 / BIN_MPH)
    assert sum(b["solid"] for b in h["bins"]) == 2


def test_bins_reach_past_the_limit_when_every_car_is_slow():
    h = histogram([(10.0, False)], limit=20.0)
    assert h["bins"][-1]["hi"] == 22.5
    assert (h["under"], h["over"]) == (1, 0)


def test_over_bins_take_the_row_colour_bands():
    h = histogram([(21.0, False), (26.0, False), (31.0, False), (36.0, False)], limit=20.0)
    band = {b["lo"]: b["band"] for b in h["bins"]}
    assert band[17.5] == "" and band[20.0] == "over-0" and band[22.5] == "over-0"
    assert band[25.0] == "over-5" and band[30.0] == "over-10" and band[35.0] == "over-15"


def test_rough_estimates_are_counted_apart_and_in_the_totals():
    h = histogram([(18.0, True), (19.0, False), (24.0, True)], limit=20.0)
    assert counts(h) == [(17.5, 20.0, 1, 1), (22.5, 25.0, 0, 1)]
    assert (h["under"], h["over"]) == (2, 1)
    assert h["tallest"] == 2


def test_passes_without_a_speed_are_counted_but_not_binned():
    h = histogram([(None, False), (12.0, False)], limit=20.0)
    assert h["no_speed"] == 1 and h["under"] == 1


def test_no_speeds_gives_no_bins():
    h = histogram([(None, False)], limit=20.0)
    assert h["bins"] == [] and h["no_speed"] == 1 and (h["under"], h["over"]) == (0, 0)


def test_the_limit_need_not_be_a_multiple_of_the_bin_width():
    h = histogram([(1.0, False), (21.0, False), (21.1, False)], limit=21.0)
    assert any(b["hi"] == pytest.approx(21.0) for b in h["bins"])
    assert h["bins"][0]["lo"] <= 0.0
    assert (h["under"], h["over"]) == (2, 1)


def test_chart_geometry_puts_the_limit_line_on_its_edge():
    h = histogram([(5.0, False), (31.0, False)], limit=20.0)
    c = h["chart"]
    edge = next(b for b in c["bars"] if b["lo"] == 20.0)
    assert c["limit_x"] == pytest.approx(edge["x"])
    assert all(b["x"] >= 0 and b["x"] + b["w"] <= c["width"] + 1e-9 for b in c["bars"])
