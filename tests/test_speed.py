import numpy as np

from safe_streets.geometry import Calibration
from safe_streets.speed import Sample, band, fit_speed, gate_flags

# 1 px = 0.05 m, X along the image x axis.
H = np.array([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 1.0]])
ROAD = [(0, 400), (1920, 400), (1920, 600), (0, 600)]
CAL = Calibration(1, [], [], H, ROAD, "", "active")


def track(ts, x0=100.0, px_per_s=160.0, y=500.0, noise=None):
    out = []
    for i, t in enumerate(ts):
        x = x0 + px_per_s * t
        if noise is not None:
            x += noise[i]
        out.append(Sample(t=t, x=x, y=y, box=(x - 100, y - 80, x + 100, y)))
    return out


def test_constant_speed_exact():
    # 8 px a frame at 20 fps = 160 px/s = 8 m/s = 17.9 mph.
    ts = [i / 20 for i in range(40)]
    fit = fit_speed(track(ts, px_per_s=160), H)
    assert abs(fit.mph - 8 * 2.23694) < 0.2
    assert fit.plus_minus < 0.2
    assert fit.direction == 1


def test_uses_pts_not_index():
    ts = [i / 8.7 for i in range(10)] + [10 / 8.7 + i / 20 for i in range(1, 20)]
    fit = fit_speed(track(ts, px_per_s=160), H)
    assert abs(fit.mph - 17.9) < 0.2


def test_jitter_outliers_ignored():
    ts = [i / 20 for i in range(60)]
    noise = np.zeros(60)
    noise[::10] = 30
    noise[5::10] = -30
    fit = fit_speed(track(ts, px_per_s=160, noise=noise), H)
    assert abs(fit.mph - 17.9) < 0.5


def test_direction_sign():
    ts = [i / 20 for i in range(40)]
    fit = fit_speed(track(ts, x0=1800, px_per_s=-160), H)
    assert fit.direction == -1 and fit.mph > 0


def test_stationary_is_parked(settings):
    rng = np.random.default_rng(1)
    ts = [i / 20 for i in range(60)]
    s = track(ts, x0=900, px_per_s=0, noise=rng.normal(0, 3, 60))
    fit = fit_speed(s, H)
    assert "parked" in gate_flags(fit, s, CAL, settings, fps=20)


def test_good_track_has_no_flags(settings):
    ts = [i / 20 for i in range(40)]
    s = track(ts, x0=10, px_per_s=960)  # crosses the whole road outline
    fit = fit_speed(s, H)
    assert gate_flags(fit, s, CAL, settings, fps=20) == []


def test_reaching_the_end_allows_for_half_a_car(settings):
    # Leading-edge positions sit half a car (100 px here) inside the outline's ends.
    ts = [i / 20 for i in range(40)]
    s = track(ts, x0=330, px_per_s=(1590 - 330) / 1.95)  # 330 > the 288 px margin alone
    fit = fit_speed(s, H)
    assert "partial" not in gate_flags(fit, s, CAL, settings, fps=20)


def test_partial_flag(settings):
    ts = [i / 20 for i in range(20)]
    s = track(ts, x0=900, px_per_s=480)  # starts mid-road
    fit = fit_speed(s, H)
    assert "partial" in gate_flags(fit, s, CAL, settings, fps=20)


def test_short_and_low_fps_flags(settings):
    ts = [i / 4 for i in range(3)]
    s = track(ts, x0=10, px_per_s=100)  # 2.5 m, under the 4 m minimum
    fit = fit_speed(s, H)
    flags = gate_flags(fit, s, CAL, settings, fps=4)
    assert "short" in flags and "low_fps" in flags


def test_band_edges():
    assert band(20, 20) == ""
    assert band(20.1, 20) == "over-0"
    assert band(25, 20) == "over-0"
    assert band(25.1, 20) == "over-5"
    assert band(30.1, 20) == "over-10"
    assert band(35.1, 20) == "over-15"
    assert band(None, 20) == ""


