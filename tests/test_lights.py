import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from night_helpers import CAL2_H, CAL2_ROAD, M_PER_PX, _path

from safe_streets.lights import (WHEEL_Y, Blob, Frames, find_blobs, is_infrared, light_frames, link_tracks,
                             night_cars)

FIX = Path(__file__).parent / "fixtures"
NIGHT = FIX / "night_car.mp4"
DAY = FIX / "car.mp4"


def test_find_blobs_centroid_and_extent():
    g = np.zeros((340, 1920), np.uint8); g[80:100, 600:640] = 255   # 20x40 = 800 px
    [b] = find_blobs(g, y_offset=380)
    assert (round(b.x), round(b.y), b.area, b.x0, b.x1) == (620, 470, 800, 600, 640)


def test_find_blobs_drops_small_and_dim():
    g = np.zeros((340, 1920), np.uint8); g[10:15, 10:15] = 255; g[50:90, 50:90] = 200
    assert find_blobs(g, 380) == []


def test_find_blobs_ignores_a_saturated_road():
    g = np.full((340, 1920), 255, np.uint8)
    assert find_blobs(g, 380) == []












CASES = json.loads((FIX / "night_blobs.json").read_text())


def case(name: str) -> Frames:
    return [(t, [Blob(*b) for b in bl]) for t, bl in CASES[name]["frames"]]


def _steady(xs_at, t0=16.0, n=32, skip=lambda t: False, y=470.0):
    """20 fps frames from t0; xs_at(t) gives the blob xs in that frame."""
    out = []
    for i in range(n):
        t = round(t0 + i * 0.05, 3)
        out.append((t, [] if skip(t) else [Blob(x, y, 200, int(x) - 10, int(x) + 10) for x in xs_at(t)]))
    return out


def test_link_one_steady_light():
    [tr] = link_tracks(_steady(lambda t: [400 + 900 * (t - 16.0)]))
    assert len(tr.points) == 32 and tr.t0 == 16.0 and tr.t1 == 17.55
    assert tr.xs[0] == 400 and tr.median_y == 470


def test_link_bridges_an_exposure_gap():
    fr = _steady(lambda t: [400 + 900 * (t - 16.0)], skip=lambda t: 16.74 <= t <= 17.09)
    [tr] = link_tracks(fr)
    assert len(tr.points) == 32 - 7


def test_link_keeps_front_and_rear_apart():
    tracks = link_tracks(_steady(lambda t: [400 + 900 * (t - 16.0), 820 + 900 * (t - 16.0)]))
    assert len(tracks) == 2
    for tr in tracks:
        assert len(tr.points) == 32 and (np.diff(tr.xs) > 0).all()
    assert sorted(tr.xs[0] for tr in tracks) == [400, 820]


def test_link_does_not_swap_to_the_rear_light_after_a_gap():
    # n0809 from the fixture: after the gap, the front track's x keeps rising
    # at about its pre-gap rate (within 25%), never stepping back ~400 px
    [front] = [tr for tr in link_tracks(case("n0809")) if tr.xs[-1] > 1800]
    v = np.diff(front.xs) / np.diff([t for t, _ in front.points])
    assert (v > 0).all()
    assert (abs(v - np.median(v)) < 0.25 * np.median(v)).all()


# --- judging tracks -------------------------------------------------------

def test_steady_near_light_is_a_car_with_speed(settings):
    cars, [v] = night_cars(_path(lambda t: 900 * t, 1920 / 900), CAL2_ROAD, CAL2_H, settings)
    assert v.is_car and v.show_speed and v.direction == 1
    [car] = cars
    assert car.direction == 1 and car.fit is not None
    assert abs(car.fit.mph - 900 * M_PER_PX * 2.23694) < 0.3
    assert all(s.y == WHEEL_Y for s in car.samples)


def test_far_side_light_not_a_car(settings):
    cars, [v] = night_cars(_path(lambda t: 900 * t, 1920 / 900, y=418), CAL2_ROAD, CAL2_H, settings)
    assert cars == [] and not v.is_car and v.car_fails == ("lane",)


def test_short_light_not_a_car(settings):
    # 900 → 1320 px: 40% of the outline's 870..1920
    cars, [v] = night_cars(_path(lambda t: 900 + 900 * t, 420 / 900), CAL2_ROAD, CAL2_H, settings)
    assert cars == [] and "cover" in v.car_fails


def test_stationary_lights_not_a_car(settings):
    rng = np.random.default_rng(1)
    cars, verdicts = night_cars(_path(lambda t: 400 + rng.uniform(-3, 3), 20.0), CAL2_ROAD, CAL2_H, settings)
    assert cars == [] and verdicts and not any(v.is_car for v in verdicts)


def test_accelerating_car_counted_without_speed(settings):
    # 08:22: waits 9 s at x≈400, then pulls away, speeding up all the way to 1900
    def x_at(t):
        return 400.0 if t < 9 else 400 + 60 * (t - 9) ** 3
    cars, [v] = night_cars(_path(x_at, 9 + (1500 / 60) ** (1 / 3)), CAL2_ROAD, CAL2_H, settings)
    assert v.is_car and not v.show_speed and "steady" in v.speed_fails
    [car] = cars
    assert car.fit is None


def test_leftward_near_light_is_a_car(settings):
    cars, [v] = night_cars(_path(lambda t: 1900 - 900 * t, 1900 / 900, y=495), CAL2_ROAD, CAL2_H, settings)
    assert v.is_car and v.show_speed and v.direction == -1
    [car] = cars
    assert car.direction == -1


def test_companion_light_is_one_car(settings):
    cars, verdicts = night_cars(_path(lambda t: [900 * t, 430 + 900 * t], 1920 / 900),
                                CAL2_ROAD, CAL2_H, settings)
    assert len(verdicts) == 2 and all(v.is_car for v in verdicts)
    [car] = cars
    assert abs(car.length_m - 430 * M_PER_PX) < 0.2


def test_two_cars_a_second_apart_are_two_cars(settings):
    # the second car's light trails by 1 s at 900 px/s: ~10 m, not a companion
    cars, _ = night_cars(_path(lambda t: [900 * t, 900 * (t - 1)], 1 + 1920 / 900), CAL2_ROAD, CAL2_H, settings)
    assert len(cars) == 2 and all(c.length_m is None for c in cars)


def _two(x_a, x_b, t1, ya=470.0, yb=470.0):
    """Two lights' frames merged: x_a at ya, x_b at yb."""
    return [(t, a + b) for (t, a), (_, b) in zip(_path(x_a, t1, y=ya), _path(x_b, t1, y=yb))]


def test_slow_close_following_cars_merge_documented_limit(settings, caplog):
    # A known limit (labels doc): two cars at 5.5 m/s, 1.0 s apart, are 5.5 m
    # light to light, which by position alone is one car's front and rear.
    v = 5.5 / M_PER_PX
    caplog.set_level("INFO", logger="safe_streets.lights")
    cars, _ = night_cars(_two(lambda t: v * t, lambda t: v * (t - 1), 1 + 1920 / v), CAL2_ROAD, CAL2_H, settings)
    assert len(cars) == 1 and abs(cars[0].length_m - 5.5) < 0.2
    assert "one car" in caplog.text


def test_following_cars_beyond_companion_range_are_two(settings):
    # 8 m/s, 1.0 s apart: 8 m, beyond night_companion_max_m (7.0)
    v = 8.0 / M_PER_PX
    cars, _ = night_cars(_two(lambda t: v * t, lambda t: v * (t - 1), 1 + 1920 / v), CAL2_ROAD, CAL2_H, settings)
    assert len(cars) == 2 and all(c.length_m is None for c in cars)


def test_split_light_is_one_car(settings):
    # one light and its reflection, 0.3 m (27 px) apart: one car, no length
    d = 0.3 / M_PER_PX
    cars, verdicts = night_cars(_two(lambda t: 900 * t, lambda t: 900 * t + d, 1920 / 900, yb=515.0),
                                CAL2_ROAD, CAL2_H, settings)
    assert len(verdicts) == 2 and all(v.is_car for v in verdicts)
    [car] = cars
    assert car.length_m is None


def test_walker_with_a_torch_not_a_car(settings):
    # 1.5 m/s (135 px/s) along the near footpath: it crosses the outline and
    # keeps going, but is far too slow for a car
    cars, [v] = night_cars(_path(lambda t: 700 + 135 * t, 1300 / 135, y=520), CAL2_ROAD, CAL2_H, settings)
    assert cars == [] and not v.is_car and v.car_fails == ("min_speed",)


def test_runner_not_a_car(settings):
    # 3 m/s, about 6.7 mph: under night_car_min_mph (7.0)
    v = 3.0 / M_PER_PX
    cars, _ = night_cars(_path(lambda t: 700 + v * t, 1300 / v, y=520), CAL2_ROAD, CAL2_H, settings)
    assert cars == []


def test_cyclist_light_is_kept(settings):
    # a single front light at 4.9 m/s (~11 mph), no companion: stored
    v = 4.9 / M_PER_PX
    cars, [vd] = night_cars(_path(lambda t: 700 + v * t, 1300 / v, y=500), CAL2_ROAD, CAL2_H, settings)
    assert vd.is_car and vd.companion is None
    [car] = cars
    assert car.length_m is None


def test_pull_away_passes_the_speed_floor(settings):
    # 08:22: the run's average, not its slow start, is judged against the floor
    def x_at(t):
        return 400.0 if t < 9 else 400 + 60 * (t - 9) ** 3
    cars, [v] = night_cars(_path(x_at, 9 + (1500 / 60) ** (1 / 3)), CAL2_ROAD, CAL2_H, settings)
    assert v.is_car and "min_speed" not in v.car_fails and len(cars) == 1


def test_far_car_reflection_not_a_car(settings):
    # a far-lane car's light (y 418) and its wet-road reflection (y 468), leftward
    cars, verdicts = night_cars(_two(lambda t: 1920 - 700 * t, lambda t: 1920 - 700 * t, 1920 / 700,
                                     ya=418.0, yb=468.0), CAL2_ROAD, CAL2_H, settings)
    assert cars == [] and len(verdicts) == 2
    fails = sorted(v.car_fails for v in verdicts)
    assert fails == [("lane",), ("reflection",)]


def test_near_car_beside_an_unrelated_far_light_is_a_car(settings):
    # a far-lane light going the other way, and one going the same way 3 m ahead:
    # neither moves with the near car, so it's still a car
    for far_x in (lambda t: 1920 - 700 * t, lambda t: 900 * t + 3.0 / M_PER_PX):
        cars, verdicts = night_cars(_two(lambda t: 900 * t, far_x, 1920 / 900, ya=470.0, yb=418.0),
                                    CAL2_ROAD, CAL2_H, settings)
        assert len(cars) == 1 and not any("reflection" in v.car_fails for v in verdicts)


def test_link_fast_car_at_low_fps(settings):
    # clips run as slow as 8.7 fps: ~100 px a frame (870 px/s) is one track, and a car
    fr = _path(lambda t: 870 * t, 1920 / 870, fps=8.7)
    [tr] = link_tracks(fr)
    assert len(tr.points) == sum(1 for _, bl in fr if bl)
    cars, [v] = night_cars(fr, CAL2_ROAD, CAL2_H, settings)
    assert v.is_car and len(cars) == 1


FOUND = {"n0809", "n0908", "n0802", "n1259"}   # must give a car; see the labels doc


@pytest.mark.parametrize("name", sorted(CASES))
def test_labelled_case(name, settings):
    c = CASES[name]
    cars, _ = night_cars(case(name), CAL2_ROAD, CAL2_H, settings)
    if c["label"] in ("far", "not_car"):
        assert cars == []                         # no false cars, ever
    else:
        assert len(cars) <= c["cars"]             # never more cars than there were
        if name in FOUND:
            assert cars


@pytest.mark.parametrize("name, lo, hi", [("n0908", 19, 25), ("n1259", 21, 26), ("n0802", 13, 17)])
def test_full_res_speeds(name, lo, hi, settings):
    [car], _ = night_cars(case(name), CAL2_ROAD, CAL2_H, settings)
    assert car.fit is not None and lo <= car.fit.mph <= hi
    if name == "n0908":
        assert car.direction == 1


def test_n0809_is_one_car(settings):
    cars, _ = night_cars(case("n0809"), CAL2_ROAD, CAL2_H, settings)
    assert len(cars) == 1


def _bands():
    """Grey bands: two street lights, then a headlight crossing, then the
    street lights alone again."""
    out = []
    for i in range(12):
        g = np.zeros((340, 1920), np.uint8)
        g[20:40, 100:140] = 255
        g[20:40, 1700:1740] = 255
        if 3 <= i < 8:
            x = 400 + 150 * i
            g[200:230, x:x + 40] = 255
        out.append(g)
    return out


def test_blob_finder_reuses_unchanged_frames_with_the_same_blobs():
    """GPU-impact spike, finding 7: most frames' lights don't change, and a
    frame's blobs depend only on its opened mask, so reusing the last
    frame's blobs when the mask is identical gives the same output."""
    from safe_streets.lights import BlobFinder

    finder = BlobFinder()
    bands = _bands()
    assert [finder(g, 380) for g in bands] == [find_blobs(g, 380) for g in bands]
    assert finder.reused == 5  # frames 1-2 repeat frame 0; 9-11 repeat frame 8




# The stitch (NIGHT_STITCH; plan 2026-11-08-night-split-stitch): the street
# tree's trunk at x≈1235 cuts a passing car's glow in two for a frame or two,
# and the linker starts a second track. Each piece alone covers too little of
# the road for a speed.

V = 900.0  # px/s: 9.96 m/s, 22.3 mph with calibration 2


@pytest.fixture
def settings_stitch(settings):
    import dataclasses
    return dataclasses.replace(settings, night_stitch="on")


def _pieces(*pieces, fps=20.0):
    """Frames with each piece's light at x0 + V·t (+ shift) while t0 <= t <= t1, at its y.
    A piece is (t0, t1, y) or (t0, t1, y, shift_px, v)."""
    end = max(p[1] for p in pieces)
    out = []
    for i in range(int(end * fps) + 1):
        t = round(i / fps, 3)
        blobs = []
        for p in pieces:
            t0, t1, y = p[:3]
            shift, v = (p[3], p[4]) if len(p) > 3 else (0.0, V)
            x = 500.0 + v * t + shift
            if t0 - 1e-9 <= t <= t1 + 1e-9 and 0 <= x <= 1920:
                blobs.append(Blob(x, y, 200, int(x) - 10, int(x) + 10))
        out.append((t, blobs))
    return out


def test_tree_split_has_no_speed_without_the_stitch(settings):
    # 20:57:08 11-08, pass 2341: 39% and 66% of the road, a car without a speed.
    [car], _ = night_cars(case("s2057"), CAL2_ROAD, CAL2_H, settings)
    assert car.fit is None


def test_tree_split_gets_a_speed(settings_stitch):
    [car], _ = night_cars(case("s2057"), CAL2_ROAD, CAL2_H, settings_stitch)
    assert car.fit is not None and 18 <= car.fit.mph <= 23 and car.direction == 1


def test_stitch_is_off_by_default(settings):
    assert settings.night_stitch == "off"


@pytest.mark.parametrize("name", sorted(CASES))
def test_stitch_never_changes_a_shown_speed(name, settings, settings_stitch):
    off, _ = night_cars(case(name), CAL2_ROAD, CAL2_H, settings)
    on, _ = night_cars(case(name), CAL2_ROAD, CAL2_H, settings_stitch)
    assert len(on) == len(off)
    shown = [c.fit for c in on if c.fit]
    assert all(c.fit in shown for c in off if c.fit)  # every shown speed survives, bit for bit


def test_stitch_joins_pieces_on_one_path(settings_stitch, caplog):
    # 500->1250 px, then 1220->1920 px 50 px lower: 36% and 67% of the road.
    caplog.set_level("INFO", logger="safe_streets.lights")
    frames = _pieces((0.0, 0.83, 445.0), (0.8, 1.58, 495.0))
    [car], verdicts = night_cars(frames, CAL2_ROAD, CAL2_H, settings_stitch)
    assert car.fit is not None and abs(car.fit.mph - 22.3) < 0.5
    assert "stitched" in caplog.text
    assert sum("stitched" in v.car_fails for v in verdicts) == 1


def test_stitch_needs_one_path(settings_stitch):
    # A short piece 2 m behind the car's path. At 10 fps (clips run as slow as
    # 8.7) the jump stays under the moving run's 3000 px/s, and Theil-Sen
    # outvotes three points: only the handover limit keeps it out.
    frames = _pieces((0.5, 0.7, 445.0, -181.0, V), (0.8, 1.58, 495.0), fps=10.0)
    [car], _ = night_cars(frames, CAL2_ROAD, CAL2_H, settings_stitch)
    assert car.fit is None


def test_stitch_skips_a_lane_failed_partner(settings_stitch):
    # A short piece above the lane line, ending just before the car starts (so
    # not its reflection): the joined track would be in the lane and steady.
    frames = _pieces((0.5, 0.8, 400.0), (0.85, 1.58, 495.0))
    [car], _ = night_cars(frames, CAL2_ROAD, CAL2_H, settings_stitch)
    assert car.fit is None


def test_stitch_skips_a_late_partner(settings_stitch):
    frames = _pieces((0.2, 0.45, 445.0), (0.8, 1.58, 495.0))  # 0.35 s between them
    [car], _ = night_cars(frames, CAL2_ROAD, CAL2_H, settings_stitch)
    assert car.fit is None


def test_stitch_uses_a_partner_once(settings_stitch, caplog):
    caplog.set_level("INFO", logger="safe_streets.lights")
    frames = _pieces((0.0, 0.83, 445.0), (0.8, 1.58, 495.0), (0.8, 1.58, 545.0))
    night_cars(frames, CAL2_ROAD, CAL2_H, settings_stitch)
    assert caplog.text.count("stitched track") == 1


def _lights(*lights, fps=20.0, end=3.5):
    """Frames with each light at x = 500 + V·t + offset while xmin <= x <= xmax.
    A light is (y, offset_px, xmin, xmax)."""
    out = []
    for i in range(int(end * fps) + 1):
        t = round(i / fps, 3)
        blobs = []
        for y, offset, xmin, xmax in lights:
            x = 500.0 + V * t + offset
            if xmin <= x <= xmax:
                blobs.append(Blob(x, y, 200, int(x) - 10, int(x) + 10))
        out.append((t, blobs))
    return out


PX_PER_M = 1 / M_PER_PX


def test_stitch_leaves_a_reflection_uncounted(settings, settings_stitch):
    # Review focus 5: the car's later piece moves with a light above the lane
    # line, so it is that light's reflection and no car. Joined with the earlier
    # piece it no longer moves with it: the stitch must not make it a car.
    k, tm = 0.75, 1.19
    frames = _pieces((0.0, 0.83, 495.0), (0.8, 1.58, 445.0), (0.8, 1.58, 400.0, (1 - k) * V * tm, k * V))
    assert night_cars(frames, CAL2_ROAD, CAL2_H, settings)[0] == []
    assert night_cars(frames, CAL2_ROAD, CAL2_H, settings_stitch)[0] == []


def test_stitch_never_adds_a_car(settings, settings_stitch):
    # A light split at the trunk, with lights 5 and 10 m behind it: one car, as
    # each rides with the next. Stitched, the split light would be kept first
    # and the light 10 m back would ride with nothing kept: a second car.
    frames = _lights((445.0, 0.0, 0, 1283), (495.0, 0.0, 1213, 1920),
                     (545.0, -5 * PX_PER_M, 870, 1700), (595.0, -10 * PX_PER_M, 870, 1605))
    assert len(night_cars(frames, CAL2_ROAD, CAL2_H, settings)[0]) == 1
    assert len(night_cars(frames, CAL2_ROAD, CAL2_H, settings_stitch)[0]) == 1


def test_stitch_keeps_only_a_join_that_shows_a_speed(settings_stitch, caplog):
    # Review focus 1: a car split in three. With the short middle piece it still
    # covers too little of the road; with the first piece it covers it all.
    caplog.set_level("INFO", logger="safe_streets.lights")
    frames = _lights((445.0, 0.0, 0, 1100), (495.0, 0.0, 1130, 1283), (545.0, 0.0, 1213, 1920))
    [car], verdicts = night_cars(frames, CAL2_ROAD, CAL2_H, settings_stitch)
    assert car.fit is not None and abs(car.fit.mph - 22.3) < 0.5
    assert caplog.text.count("stitched track") == 1
    [middle] = [v for v in verdicts if abs(v.track.median_y - 495.0) < 1]
    assert "stitched" not in middle.car_fails


def test_stitch_joins_front_and_rear_lights_each_split(settings, settings_stitch, caplog):
    # Review focus 3: front and rear lights 4 m apart, each split at the trunk.
    # Two stitches, still one car, now with a speed.
    frames = _lights((445.0, 0.0, 0, 1283), (495.0, 0.0, 1213, 1920),
                     (545.0, -4 * PX_PER_M, 0, 1283), (595.0, -4 * PX_PER_M, 1213, 1920), end=3.0)
    [car] = night_cars(frames, CAL2_ROAD, CAL2_H, settings)[0]
    assert car.fit is None
    caplog.set_level("INFO", logger="safe_streets.lights")
    [car], _ = night_cars(frames, CAL2_ROAD, CAL2_H, settings_stitch)
    assert car.fit is not None and abs(car.fit.mph - 22.3) < 0.5
    assert caplog.text.count("stitched track") == 2


def test_stitch_keeps_two_cars_a_second_apart(settings_stitch):
    # Review focus 2: two cars 1 s (10 m) apart, each split at the trunk: each
    # car's pieces join each other, never the other car's.
    frames = _lights((445.0, 0.0, 0, 1283), (495.0, 0.0, 1213, 1920),
                     (545.0, -V, 0, 1283), (595.0, -V, 1213, 1920))
    cars, _ = night_cars(frames, CAL2_ROAD, CAL2_H, settings_stitch)
    assert len(cars) == 2 and all(c.fit is not None and abs(c.fit.mph - 22.3) < 0.5 for c in cars)


def test_stitch_at_the_slowest_frame_rate(settings_stitch):
    # Review focus 4: 8.7 fps, the slowest clips: four points in the last 0.5 s.
    # (The later piece starts a frame earlier than elsewhere: at 8.7 fps one
    # from 1213 px samples only 59% of the road, under a car's 60%.)
    frames = _lights((445.0, 0.0, 0, 1283), (495.0, 0.0, 1100, 1920), fps=8.7)
    [car], _ = night_cars(frames, CAL2_ROAD, CAL2_H, settings_stitch)
    assert car.fit is not None and abs(car.fit.mph - 22.3) < 0.5


def test_handover_needs_two_times():
    from types import SimpleNamespace as NS

    from safe_streets.lights import _handover_m
    late = NS(samples=[NS(t=1.0, x=1300.0)])
    H = np.asarray(CAL2_H, dtype=float)
    assert _handover_m(NS(samples=[NS(t=0.9, x=1200.0)]), late, H) is None
    assert _handover_m(NS(samples=[NS(t=0.9, x=1200.0), NS(t=0.9, x=1210.0)]), late, H) is None


def test_stitch_logs_where_and_both_speeds(settings_stitch, caplog):
    # For the night review: where the earlier piece ends and the later starts
    # (an overlap, or a gap, for a trunk split), and the car's speed from its
    # own piece beside the joined one.
    caplog.set_level("INFO", logger="safe_streets.lights")
    night_cars(_pieces((0.0, 0.83, 445.0), (0.8, 1.58, 495.0)), CAL2_ROAD, CAL2_H, settings_stitch)
    [line] = [r.getMessage() for r in caplog.records if "stitched track" in r.getMessage()]
    assert "first ends at x 1220, second starts at x 1220" in line and " mph alone, 22.3 joined" in line


def test_stitch_undone_is_logged(settings_stitch, caplog):
    caplog.set_level("INFO", logger="safe_streets.lights")
    frames = _lights((445.0, 0.0, 0, 1283), (495.0, 0.0, 1213, 1920),
                     (545.0, -5 * PX_PER_M, 870, 1700), (595.0, -10 * PX_PER_M, 870, 1605))
    night_cars(frames, CAL2_ROAD, CAL2_H, settings_stitch)
    assert "stitch undone" in caplog.text and "stitched track" not in caplog.text
