import cv2
import numpy as np

from safe_streets.scan import find_windows, merge_windows

ROAD = [(870, 457), (1920, 465), (1920, 566), (870, 503)]  # full-frame px


def make_clip(path, seconds=30, fps=15, car_at=(15.0, 17.0)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 360))
    rng = np.random.default_rng(0)
    bg = (rng.random((360, 640, 3)) * 60 + 80).astype(np.uint8)
    for i in range(seconds * fps):
        t = i / fps
        f = bg.copy()
        if car_at[0] <= t <= car_at[1]:
            x = int(300 + (t - car_at[0]) / (car_at[1] - car_at[0]) * 300)
            cv2.rectangle(f, (x - 60, 120), (x + 60, 185), (255, 255, 255), -1)
        w.write(f)
    w.release()


def test_moving_car_on_road_gives_one_padded_window(tmp_path):
    clip = tmp_path / "low.mp4"
    make_clip(clip)
    wins = find_windows(clip, ROAD)
    assert len(wins) == 1
    a, b = wins[0]
    assert 10.5 <= a <= 11.5 and 20.5 <= b <= 21.5  # 15–17 s padded by 4 s


def test_motion_during_warmup_ignored(tmp_path):
    clip = tmp_path / "low.mp4"
    make_clip(clip, car_at=(2.0, 4.0))
    assert find_windows(clip, ROAD) == []


def test_merge_windows_merges_close_and_caps_length():
    assert merge_windows([(0, 5), (6, 10), (20, 25)], gap=2, max_len=60) == [(0, 10), (20, 25)]
    assert merge_windows([(0, 50), (51, 90)], gap=2, max_len=60) == [(0, 50), (51, 90)]
