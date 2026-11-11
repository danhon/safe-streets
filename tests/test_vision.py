import os
from pathlib import Path

import numpy as np
import pytest

from safe_streets.vision import Gate, crop_for, duration_from, fps_from, keep_sample, moving_span, wide_crop_for

MODELS = Path(__file__).parent.parent / "models"


def test_edge_truncated_samples_dropped():
    crop = (600, 200, 1920, 600)
    assert keep_sample((800, 300, 1000, 500), crop)
    assert not keep_sample((601, 300, 800, 500), crop)  # touching the left edge
    assert not keep_sample((1700, 300, 1918, 500), crop)  # touching the right edge


def test_crop_for_extends_up_and_down_and_clips():
    road = [(600, 450), (1920, 450), (1920, 570), (600, 570)]
    x0, y0, x1, y1 = crop_for(road)
    assert (x0, x1) == (600, 1920)  # x unchanged: the calibration scale was measured at this crop
    assert y0 == 450 - 180
    assert y1 == 570 + 36  # near-curb cars' box bottoms reach below the outline
    assert crop_for([(0, 50), (100, 50), (100, 100), (0, 100)])[1] == 0
    assert crop_for([(0, 1000), (100, 1000), (100, 1070), (0, 1070)])[3] == 1080


def test_wide_crop_adds_a_car_length_each_side():
    road = [(600, 450), (1920, 450), (1920, 570), (600, 570)]
    x0, y0, x1, y1 = wide_crop_for(road)
    assert (x0, x1) == (200, 1920)
    assert (y0, y1) == crop_for(road)[1::2]


def test_gate_empty_clip(tmp_path):
    from safe_streets.vision import Vision

    clip = tmp_path / "empty.mp4"
    clip.write_bytes(b"")
    v = Vision(MODELS, "intel:cpu", 0.35)
    v._detect = lambda img: pytest.fail("no frame to detect in")
    assert v.gate(clip, (0, 0, 1920, 1080), [(0, 0), (1920, 0), (1920, 1080), (0, 1080)]) == Gate(None, [])


def test_duration_and_fps_from_times():
    assert duration_from([0, .05, .10]) == pytest.approx(.15)
    assert duration_from([]) is None
    assert fps_from([0, .05, .10]) == pytest.approx(20)
    assert fps_from([]) == 0.0


def _frame_by_decode(clip, t):
    """The screenshot rule, decoding from the start: the first frame with
    pts ≥ t − 1 ms, else the last."""
    from safe_streets.vision import _frames

    best = None
    for ft, frame, _ in _frames(clip):
        best = frame
        if ft >= t - 1e-3:
            break
    return best


def test_frame_at_seek_empty_clip(tmp_path):
    from safe_streets.vision import frame_at_seek

    clip = tmp_path / "empty.mp4"
    clip.write_bytes(b"")
    assert frame_at_seek(clip, 1.0) is None


def test_low_res_frames_scaled_to_1080p(tmp_path):
    import cv2
    import numpy as np

    from safe_streets.vision import _frames

    clip = tmp_path / "low.mp4"
    w = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 360))
    for _ in range(3):
        w.write(np.zeros((360, 640, 3), np.uint8))
    w.release()
    frames = list(_frames(clip))
    assert len(frames) == 3
    assert all(f.shape[:2] == (1080, 1920) for _, f, _ in frames)
    assert frames[0][2] is True  # scaled


def _s(t, x0, x1, y=540):
    from safe_streets.speed import Sample

    return Sample(t=t, x=(x0 + x1) / 2, y=y, box=(x0, y - 200, x1, y))


def test_clean_track_follows_the_leading_edge():
    from safe_streets.vision import clean_track

    crop = (860, 200, 1920, 600)
    s = [
        _s(0.0, 1300, 1500),  # rear hidden behind a parked car: front edge still true
        _s(0.1, 1100, 1550),
        _s(0.2, 1150, 1600),
        _s(0.3, 1200, 1650),
        _s(0.4, 1500, 1918),  # front at the frame edge: dropped
    ]
    out = clean_track(s, crop)
    assert [round(q.t, 1) for q in out] == [0.0, 0.1, 0.2, 0.3]
    # Position is the front edge minus half the full width (450 px).
    assert [q.x for q in out] == [1275, 1325, 1375, 1425]


def test_clean_track_leftward_uses_left_edge():
    from safe_streets.vision import clean_track

    crop = (860, 200, 1920, 600)
    s = [_s(0.0, 1500, 1918), _s(0.1, 1400, 1850), _s(0.2, 1300, 1750), _s(0.3, 1200, 1650), _s(0.4, 862, 1400)]
    out = clean_track(s, crop)
    assert [round(q.t, 1) for q in out] == [0.0, 0.1, 0.2, 0.3]
    assert [q.x for q in out] == [1725, 1625, 1525, 1425]


def test_clean_track_needs_three_samples():
    from safe_streets.vision import clean_track

    crop = (860, 200, 1920, 600)
    assert clean_track([_s(0, 1500, 1700), _s(0.1, 1550, 1750)], crop) == []


def test_split_track_at_id_switch():
    from safe_streets.vision import split_track

    parked = [_s(i / 20, 870, 1187) for i in range(10)]  # a parked SUV cut off by the crop
    car = [_s(0.5 + i / 20, 1250 + 30 * i, 1700 + 30 * i) for i in range(10)]  # the car that took its ID
    parts = split_track(parked + car)
    assert [len(p) for p in parts] == [10, 10]


def test_split_track_keeps_a_smooth_track_whole():
    from safe_streets.vision import split_track

    assert len(split_track([_s(i / 20, 1000 + 30 * i, 1450 + 30 * i) for i in range(20)])) == 1


def test_clean_track_drops_a_stationary_lead_in():
    from safe_streets.vision import clean_track

    crop = (860, 200, 1920, 600)
    # A parked car's box whose ID then passes to a car driving away right:
    # the leading edge sits still, then moves.
    still = [_s(i / 20, 870, 1187) for i in range(10)]
    moving = [_s(0.5 + i / 20, 1250 + 30 * i, 1700 + 30 * i) for i in range(10)]
    out = clean_track(still + moving, crop)
    assert out and min(q.t for q in out) >= 0.45


def test_direction_from_overall_motion_not_median_step():
    from safe_streets.vision import clean_track

    crop = (860, 200, 1920, 600)
    # A long lead-in whose box creeps left (a parked car's box merging with the
    # emerging one), then the car drives right: it's moving right.
    creep = [_s(i / 20, 1000 - 2 * i, 1300 - 2 * i) for i in range(15)]
    # The box grows as more of the car comes out from behind the parked one,
    # so its front and rear edges move at different rates.
    car = [_s(0.75 + i / 20, 1300 + 20 * i, 1700 + 40 * i) for i in range(6)]
    out = clean_track(creep + car, crop)
    xs = [q.x for q in out]
    fronts = [q.box[2] for q in out]
    assert len(out) >= 3
    assert [round(b - a) for a, b in zip(xs, xs[1:])] == [round(b - a) for a, b in zip(fronts, fronts[1:])]


def test_one_id_for_parking_then_parked_then_passing_car():
    from safe_streets.vision import clean_track

    crop = (870, 293, 1920, 566)
    # 11:21:14 shape: an SUV slowly reverses left into a space (120 px/s for 2 s),
    # sits parked for 6 s, then its ID passes to a car driving right at 600 px/s.
    parking = [_s(i / 20, 1300 - 6 * i, 1600 - 6 * i) for i in range(40)]
    parked = [_s(2 + i / 20, 1060, 1360) for i in range(120)]
    passing = [_s(8 + i / 20, 1150 + 30 * i, 1600 + 30 * i + 2 * i) for i in range(10)]
    out = clean_track(parking + parked + passing, crop)
    assert out and min(q.t for q in out) >= 8
    fronts = [q.box[2] for q in out]
    xs = [q.x for q in out]
    assert [round(b - a) for a, b in zip(xs, xs[1:])] == [round(b - a) for a, b in zip(fronts, fronts[1:])]


def test_moving_span_ignores_parked_and_spans_the_movers():
    parked = (1500.0, 560.0)
    frames = [(t / 2, [parked]) for t in range(20)]  # 0–9.5 s, a parked car in every frame
    frames[8] = (4.0, [parked, (900.0, 580.0)])
    frames[11] = (5.5, [parked, (1600.0, 585.0)])
    assert moving_span(frames) == (4.0, 5.5)


def test_moving_span_none_when_only_parked_or_empty():
    assert moving_span([(t / 2, [(1500.0, 560.0)]) for t in range(10)]) is None
    assert moving_span([]) is None
    assert moving_span([(0.0, [])]) is None


@pytest.fixture
def recorded_compile(monkeypatch):
    """ov.Core.compile_model replaced by a recorder (and restored after)."""
    import openvino as ov

    calls = []

    def fake(self, model, device_name=None, config=None, **kw):
        calls.append((device_name, dict(config or {})))
        return "compiled"

    monkeypatch.setattr(ov.Core, "compile_model", fake)
    return calls


def test_gpu_throttle_added_to_gpu_compiles_only(recorded_compile):
    """OpenVINO's GPU wait busy-spins the calling thread: 45 of 46 ms of CPU
    per 640x640 inference on the HD 530, 8 ms with GPU_QUEUE_THROTTLE=LOW,
    outputs identical (bench/gpu_wait.py, 2026-11-02). Ultralytics builds its
    own Core, so the hint goes in at compile_model."""
    import openvino as ov

    from safe_streets.vision import gpu_throttle

    gpu_throttle("LOW")
    core = ov.Core.__new__(ov.Core)  # the method is patched; no device needed
    core.compile_model("m", "GPU", {"PERFORMANCE_HINT": "LATENCY"})
    core.compile_model("m", "GPU.0")
    core.compile_model("m", "CPU", {})
    assert recorded_compile == [
        ("GPU", {"PERFORMANCE_HINT": "LATENCY", "GPU_QUEUE_THROTTLE": "LOW"}),
        ("GPU.0", {"GPU_QUEUE_THROTTLE": "LOW"}),
        ("CPU", {}),
    ]


def test_gpu_throttle_idempotent_and_keeps_an_explicit_value(recorded_compile):
    import openvino as ov

    from safe_streets.vision import gpu_throttle

    gpu_throttle("LOW")
    gpu_throttle("LOW")  # Vision may be built more than once
    ov.Core.__new__(ov.Core).compile_model("m", "GPU", {"GPU_QUEUE_THROTTLE": "HIGH"})
    assert recorded_compile == [("GPU", {"GPU_QUEUE_THROTTLE": "HIGH"})]


def test_gpu_throttle_off_leaves_compile_alone(recorded_compile):
    import openvino as ov

    from safe_streets.vision import gpu_throttle

    gpu_throttle("")
    ov.Core.__new__(ov.Core).compile_model("m", "GPU", None)
    assert recorded_compile == [("GPU", {})]


def test_vision_applies_the_throttle_setting(monkeypatch):
    import safe_streets.vision as V

    seen = []
    monkeypatch.setattr(V, "gpu_throttle", seen.append)
    V.Vision(MODELS, "intel:gpu", 0.35, throttle="LOW")
    V.Vision(MODELS, "intel:gpu", 0.35)
    assert seen == ["LOW", "LOW"]  # on by default


def test_compile_cache_added_to_every_compile(recorded_compile, tmp_path):
    """OpenVINO's model cache: a restart loads the compiled model instead of
    compiling it again (~20 s on the GPU for the first event after each
    restart, 2026-11-07). CPU too, for the CPU week."""
    import openvino as ov

    from safe_streets.vision import compile_cache

    d = tmp_path / "ov-cache"
    compile_cache(d)
    compile_cache(d)  # Vision may be built more than once
    core = ov.Core.__new__(ov.Core)
    core.compile_model("m", "GPU", {"PERFORMANCE_HINT": "LATENCY"})
    core.compile_model("m", "CPU")
    core.compile_model("m", "GPU", {"CACHE_DIR": "/elsewhere"})
    assert d.is_dir()
    assert recorded_compile == [
        ("GPU", {"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": str(d)}),
        ("CPU", {"CACHE_DIR": str(d)}),
        ("GPU", {"CACHE_DIR": "/elsewhere"}),
    ]


def test_compile_cache_off_leaves_compile_alone(recorded_compile):
    import openvino as ov

    from safe_streets.vision import compile_cache

    compile_cache(None)
    compile_cache("")
    ov.Core.__new__(ov.Core).compile_model("m", "GPU", None)
    assert recorded_compile == [("GPU", {})]


def test_compile_cache_and_throttle_together(recorded_compile, tmp_path):
    import openvino as ov

    from safe_streets.vision import compile_cache, gpu_throttle

    gpu_throttle("LOW")
    compile_cache(tmp_path)
    gpu_throttle("LOW")  # a second Vision: neither hook doubles up or drops
    compile_cache(tmp_path)
    ov.Core.__new__(ov.Core).compile_model("m", "GPU")
    assert recorded_compile == [("GPU", {"GPU_QUEUE_THROTTLE": "LOW", "CACHE_DIR": str(tmp_path)})]


def test_vision_applies_the_cache_setting(monkeypatch, tmp_path):
    import safe_streets.vision as V

    seen = []
    monkeypatch.setattr(V, "compile_cache", seen.append)
    V.Vision(MODELS, "intel:gpu", 0.35, cache_dir=tmp_path)
    V.Vision(MODELS, "intel:gpu", 0.35)
    assert seen == [tmp_path, None]  # off unless given


def test_settings_ov_cache_dir_default_and_off(monkeypatch):
    from safe_streets.config import Settings

    monkeypatch.setenv("DATA_DIR", "/data")
    assert Settings().ov_cache_dir == Path("/data/ov-cache")
    monkeypatch.setenv("OV_CACHE_DIR", "/x")
    assert Settings().ov_cache_dir == Path("/x")
    monkeypatch.setenv("OV_CACHE_DIR", "")
    assert Settings().ov_cache_dir is None


@pytest.mark.skipif(os.environ.get("SAFE_STREETS_REAL_MODEL") != "1" or not (MODELS / "rtmdet-tiny").exists(),
                    reason="the real model: SAFE_STREETS_REAL_MODEL=1 and models/rtmdet-tiny")
def test_a_real_compile_fills_the_cache(monkeypatch, tmp_path):
    import openvino as ov

    from safe_streets.vision import Vision

    monkeypatch.setattr(ov.Core, "compile_model", ov.Core.compile_model)  # restored after
    v = Vision(MODELS, "cpu", 0.35, cache_dir=tmp_path)
    v._detect(np.zeros((360, 640, 3), np.uint8))
    assert any(tmp_path.iterdir())


def test_settings_gpu_throttle_default_and_off(monkeypatch):
    from safe_streets.config import Settings

    assert Settings().yolo_gpu_throttle == "LOW"
    monkeypatch.setenv("YOLO_GPU_THROTTLE", "")
    assert Settings().yolo_gpu_throttle == ""


def test_settings_track_pads_default_to_todays(monkeypatch):
    from safe_streets.config import Settings

    s = Settings()
    assert (s.track_pad_before_s, s.track_pad_after_s) == (2.5, 2.5)
    monkeypatch.setenv("TRACK_PAD_BEFORE_S", "1.0")
    monkeypatch.setenv("TRACK_PAD_AFTER_S", "0.5")
    s = Settings()
    assert (s.track_pad_before_s, s.track_pad_after_s) == (1.0, 0.5)


def test_settings_motion_gate_defaults_off(monkeypatch):
    from safe_streets.config import Settings

    assert Settings().motion_gate == "off"
    monkeypatch.setenv("MOTION_GATE", "shadow")
    assert Settings().motion_gate == "shadow"


ROAD_BAND = [(0, 400), (1920, 400), (1920, 600), (0, 600)]


def _clip(dest: Path, frames, fps: float = 10) -> Path:
    import cv2

    w = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080))
    for f in frames:
        w.write(f)
    w.release()
    return dest


def _square_frames(n=14, step=80, x0=100, width=300):
    """A white box crossing the road band on grey, step px a frame; a uniform
    box only changes at its edges, so step must beat MOTION_RUN_PX."""
    for i in range(n):
        f = np.full((1080, 1920, 3), 128, np.uint8)
        x = x0 + step * i
        f[450:550, x:x + width] = 255
        yield f


def _find_square(img):
    xs = np.where(img[500, :, 0] > 200)[0]
    return np.array([[xs.min(), 450, xs.max() + 1, 550]], float) if len(xs) else np.zeros((0, 4))


def _offline_vision():
    from safe_streets.vision import Vision

    return Vision(MODELS, "intel:cpu", 0.35)


def test_motion_gate_skips_yolo_without_motion(tmp_path):
    clip = _clip(tmp_path / "still.mp4", (np.full((1080, 1920, 3), 128, np.uint8) for _ in range(30)))
    v = _offline_vision()
    v._detect = lambda img: pytest.fail("nothing moved, so no YOLO")
    g = v.motion_gate(clip, (0, 0, 1920, 1080), ROAD_BAND)
    assert (g.span, g.moving, g.inferences) == (None, [], 0)
    assert len(g.times) == 30 and g.times[0] == 0


def test_motion_gate_moving_needs_motion_in_the_box(tmp_path):
    """A detection counts only where the motion is: the white box does, a
    'parked' box beside its path never does (bench.py's motion_gate)."""
    clip = _clip(tmp_path / "box.mp4", _square_frames())
    parked = np.array([[1500, 450, 1800, 550]], float)  # past the box's path (x ≤ 1440)
    v = _offline_vision()
    v._detect = lambda img: parked
    g = v.motion_gate(clip, (0, 0, 1920, 1080), ROAD_BAND)
    assert g.span is None and g.inferences > 0

    v._detect = _find_square
    g = v.motion_gate(clip, (0, 0, 1920, 1080), ROAD_BAND)
    assert g.span is not None and g.inferences > 0 and len(g.moving) == g.inferences
    assert (min(g.moving), max(g.moving)) == g.span


def test_motion_gate_empty_clip(tmp_path):
    clip = tmp_path / "empty.mp4"
    clip.write_bytes(b"")
    v = _offline_vision()
    v._detect = lambda img: pytest.fail("no frame to detect in")
    assert v.motion_gate(clip, (0, 0, 1920, 1080), ROAD_BAND) == Gate(None, [])


def test_episodes_are_the_union_of_pads_around_moving_samples():
    from safe_streets.vision import episodes

    assert episodes([], (1.0, 0.5)) == []
    assert episodes([5.0], (1.0, 0.5)) == [(4.0, 5.5)]
    assert episodes([5.0, 5.0, 5.5], (1.0, 0.5)) == [(4.0, 6.0)]  # overlapping: one
    assert episodes([9.0, 5.0], (1.0, 0.5)) == [(4.0, 5.5), (8.0, 9.5)]  # apart: two, in order
    assert episodes([5.0, 6.5], (1.0, 0.5)) == [(4.0, 7.0)]  # touching: one


def test_settings_episodes_default_off(monkeypatch):
    from safe_streets.config import Settings

    assert Settings().episodes == "off"
    monkeypatch.setenv("EPISODES", "on")
    assert Settings().episodes == "on"




def _no_yolo_vision():
    v = _offline_vision()
    v._detect = lambda img: np.zeros((0, 4))
    return v


# /healthz's yolo_device (2026-11-08): a fresh ov.Core() lists devices in
# ~0.35-0.45 s of CPU, on the event loop, and Docker's healthcheck asks every
# 30 s. The devices can't change while the process runs, so it's asked once.


class _Core:
    made = 0
    devices: list | Exception = ["CPU", "GPU"]

    def __init__(self):
        type(self).made += 1

    @property
    def available_devices(self):
        if isinstance(self.devices, Exception):
            raise self.devices
        return self.devices


@pytest.fixture
def fake_core(monkeypatch):
    import openvino as ov

    from safe_streets.vision import Vision

    v = Vision(MODELS, "intel:gpu", 0.35, throttle="")
    _Core.made, _Core.devices = 0, ["CPU", "GPU"]
    monkeypatch.setattr(ov, "Core", _Core)
    return v


def test_device_in_use_asks_openvino_once(fake_core):
    assert [fake_core.device_in_use() for _ in range(3)] == ["intel:gpu"] * 3
    assert _Core.made == 1


def test_device_in_use_without_a_gpu(fake_core):
    _Core.devices = ["CPU"]
    assert fake_core.device_in_use() == "cpu (no GPU visible)"


def test_a_failed_device_query_is_asked_again(fake_core):
    _Core.devices = RuntimeError("no plugin")
    assert fake_core.device_in_use() == "unknown (no plugin)"
    _Core.devices = ["CPU", "GPU"]
    assert fake_core.device_in_use() == "intel:gpu"
    assert _Core.made == 2


# Two-wheelers (plan 2026-11-09-two-wheelers, Task 2) ----------------------

def _boxes(*rows):
    """rows: (x0, y0, x1, y1, cls, conf) -> xyxy, cls, conf arrays."""
    import numpy as np

    a = np.array(rows, dtype=float).reshape(-1, 6)
    return a[:, :4], a[:, 4], a[:, 5]


def test_overlapping_bicycle_and_motorcycle_keep_one():
    from safe_streets.vision import two_wheeler_keep

    keep = two_wheeler_keep(*_boxes((100, 100, 200, 200, 1, 0.6), (105, 100, 205, 200, 3, 0.7)))
    assert keep.tolist() == [False, True]


def test_two_riders_side_by_side_keep_both():
    from safe_streets.vision import two_wheeler_keep

    keep = two_wheeler_keep(*_boxes((100, 100, 200, 200, 1, 0.6), (185, 100, 285, 200, 3, 0.7)))
    assert keep.tolist() == [True, True]


def test_two_bicycles_overlapping_keep_both():
    """Only a bicycle against a motorcycle is one rider seen twice: YOLO's
    own NMS already settled two boxes of one class."""
    from safe_streets.vision import two_wheeler_keep

    keep = two_wheeler_keep(*_boxes((100, 100, 200, 200, 1, 0.6), (105, 100, 205, 200, 1, 0.7)))
    assert keep.tolist() == [True, True]


def test_bicycle_on_a_car_rack_dropped():
    from safe_streets.vision import two_wheeler_keep

    keep = two_wheeler_keep(*_boxes((0, 0, 400, 200, 2, 0.9), (300, 20, 400, 100, 1, 0.5)))
    assert keep.tolist() == [True, False]


def test_motorcycle_inside_a_car_box_kept():
    from safe_streets.vision import two_wheeler_keep

    keep = two_wheeler_keep(*_boxes((0, 0, 400, 200, 2, 0.9), (300, 20, 400, 100, 3, 0.5)))
    assert keep.tolist() == [True, True]


def test_bicycle_overlapping_a_car_a_little_kept():
    from safe_streets.vision import two_wheeler_keep

    keep = two_wheeler_keep(*_boxes((0, 0, 400, 200, 7, 0.9), (370, 20, 470, 120, 1, 0.5)))  # 30% inside
    assert keep.tolist() == [True, True]


def test_two_wheeler_keep_empty():
    from safe_streets.vision import two_wheeler_keep

    assert two_wheeler_keep(*_boxes()).tolist() == []


def test_vehicles_include_bicycles():
    from safe_streets.vision import VEHICLES

    assert VEHICLES == {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}




