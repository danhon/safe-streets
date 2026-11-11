"""Settings, read once from the environment (.env in development)."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv

load_dotenv()


# Metres per second in one of each unit.
UNIT_MPS = {"mph": 0.44704, "kmh": 1 / 3.6}


def _env(name: str, default):
    return type(default)(os.environ.get(name, default))


def _opt_float(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else None


def _ov_cache_dir() -> Path | None:
    v = os.environ.get("OV_CACHE_DIR")
    if v is None:
        return Path(os.environ.get("DATA_DIR", "./data")) / "ov-cache"
    return Path(v) if v.strip() else None


# Segment names: the start in local time with its UTC offset. Without the
# offset, the hour the clocks go back repeats its names (the bundled recorder
# would overwrite the first hour, and nothing could tell the two apart).
DEFAULT_SOURCE_PATTERN = "%Y%m%d-%H%M%S%z.mp4"

@dataclass(frozen=True)
class Settings:
    # Where footage comes from (sources/): folder | protect.
    source: str = field(default_factory=lambda: _env("SOURCE", "folder"))
    # The folder source: segment files named by their start time (a strftime
    # pattern; local time in TZ unless it has %z), ignored until unmodified
    # for SOURCE_SETTLE_S (still being written).
    source_dir: Path | None = field(default_factory=lambda: Path(os.environ["SOURCE_DIR"]) if os.environ.get("SOURCE_DIR") else None)
    source_pattern: str = field(default_factory=lambda: _env("SOURCE_PATTERN", DEFAULT_SOURCE_PATTERN))
    source_settle_s: float = field(default_factory=lambda: _env("SOURCE_SETTLE_S", 10.0))

    protect_host: str = field(default_factory=lambda: _env("PROTECT_HOST", ""))
    protect_port: int = field(default_factory=lambda: _env("PROTECT_PORT", 443))
    protect_username: str = field(default_factory=lambda: _env("PROTECT_USERNAME", ""))
    protect_password: str = field(default_factory=lambda: _env("PROTECT_PASSWORD", ""))
    protect_camera: str = field(default_factory=lambda: _env("PROTECT_CAMERA", ""))
    # How far behind now Protect's recording can be exported.
    protect_lag_s: float = field(default_factory=lambda: _env("PROTECT_LAG_S", 30.0))

    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("DATA_DIR", "./data")))
    models_dir: Path = field(default_factory=lambda: Path(os.environ.get("MODELS_DIR", "./models")))

    # The street's limit, in UNITS. Required: there's no sensible default.
    speed_limit: float | None = field(default_factory=lambda: _opt_float("SPEED_LIMIT"))
    # mph | kmh: how speeds are shown (they're stored in m/s).
    units: str = field(default_factory=lambda: _env("UNITS", "mph").strip().lower())
    # The camera's time zone (an IANA name like Europe/London). Required.
    tz: str = field(default_factory=lambda: _env("TZ", "").strip())

    # Batching. Every 1 min since 2026-11-08 (2 min from 2026-11-07, 5
    # before): a batch costs ~0.1 s beyond its events, so the interval sets
    # how late posts go out (median car-to-post 129 s at 2 min, ~100 s
    # expected at 1), not how much work the box does.
    batch_interval_seconds: int = field(default_factory=lambda: _env("BATCH_INTERVAL_SECONDS", 60))
    # Where the kernel can't report CPU pressure (batcher.cpu_pressure).
    max_load: float = field(default_factory=lambda: _env("MAX_LOAD", 4.0))
    # Busy = CPU pressure (`some avg60`, %) over this. Unset (the default):
    # batches never wait for a quiet box. The reference install uses 60 on a
    # 4-core box shared with other services; normal days run 6-40%.
    max_cpu_pressure: float | None = field(default_factory=lambda: _opt_float("MAX_CPU_PRESSURE"))
    # Load holds work back for at most MAX_SKIPS batch intervals (30 x 1 min,
    # ~30 min); then a batch runs anyway. Change it with the interval.
    max_skips: int = field(default_factory=lambda: _env("MAX_SKIPS", 30))
    # With a backlog, a busy box is checked again this often (2026-11-03).
    load_retry_seconds: int = field(default_factory=lambda: _env("LOAD_RETRY_SECONDS", 30))
    # A batch's posts wait for it to end, so a backlog posts every 10 events;
    # the next batch starts at once, so the backlog clears no slower (was 30).
    max_per_run: int = field(default_factory=lambda: _env("MAX_PER_RUN", 10))
    lookback_hours: float = field(default_factory=lambda: _env("LOOKBACK_HOURS", 24.0))
    login_retry_seconds: int = field(default_factory=lambda: _env("LOGIN_RETRY_SECONDS", 300))
    # Every Protect call gives up after this (2026-11-08): after the 11-04
    # reboot the batcher waited ~36 h on one. uiprotect has no timeout on
    # exports and 5 min on the rest. An export is retried (3 tries), each
    # try with EXPORT_TIMEOUT_SECONDS; one normally takes under a second.
    protect_timeout_seconds: float = field(default_factory=lambda: _env("PROTECT_TIMEOUT_SECONDS", 60.0))
    export_timeout_seconds: float = field(default_factory=lambda: _env("EXPORT_TIMEOUT_SECONDS", 120.0))
    # /healthz says `batch_stale` when no batch has finished for this long
    # (since the start, after a restart). A skipped (busy) batch counts as
    # finished, and a full batch takes minutes, so 30 min is well past normal.
    batch_stale_seconds: int = field(default_factory=lambda: _env("BATCH_STALE_SECONDS", 1800))
    camera_check_seconds: int = field(default_factory=lambda: _env("CAMERA_CHECK_SECONDS", 3600))

    # Clips.
    # Motion events start late and end early on a passing car; widen them.
    clip_pad_seconds: int = field(default_factory=lambda: _env("CLIP_PAD_SECONDS", 5))
    clip_max_seconds: int = field(default_factory=lambda: _env("CLIP_MAX_SECONDS", 60))
    clip_retention_days: int = field(default_factory=lambda: _env("CLIP_RETENTION_DAYS", 30))

    # The operator's dashboard (/operator): password (a built-in password,
    # the default) | none (loopback clients only, for trying it out).
    auth_mode: str = field(default_factory=lambda: _env("AUTH_MODE", "password").strip().lower())
    # This app's address as people reach it, for the setup link the log
    # prints (e.g. https://traffic.example.org). Unset: http://localhost:WEB_PORT.
    base_url: str = field(default_factory=lambda: _env("BASE_URL", "").rstrip("/"))
    # The reverse proxies whose X-Forwarded-For/-Proto the app believes
    # (uvicorn's forwarded_allow_ips: comma-separated addresses, or *). Behind
    # a proxy, set it to the proxy's address, so the app sees the visitor's
    # address and the https scheme. Default: only a proxy on this machine.
    forwarded_allow_ips: str = field(default_factory=lambda: _env("FORWARDED_ALLOW_IPS", "127.0.0.1"))

    # Hooks (hooks.py): comma-separated module:attr entry points, each called
    # after a batch with its new passes.
    hooks: str = field(default_factory=lambda: _env("HOOKS", ""))

    # The public views (public_view.py): nothing newer than PUBLIC_DELAY_HOURS,
    # times floored to PUBLIC_TIME_BUCKET_MIN, and a bucket of fewer than
    # PUBLIC_MIN_GROUP passes shown only as a total. Loosen each with care
    # (README, "Privacy by default").
    public_delay_hours: float = field(default_factory=lambda: _env("PUBLIC_DELAY_HOURS", 24.0))
    public_time_bucket_min: int = field(default_factory=lambda: _env("PUBLIC_TIME_BUCKET_MIN", 15))
    public_min_group: int = field(default_factory=lambda: _env("PUBLIC_MIN_GROUP", 5))

    # Speed gates.
    # The camera can follow a car's leading edge for about 9 m at most.
    min_track_m: float = field(default_factory=lambda: _env("MIN_TRACK_M", 4.0))
    min_track_s: float = field(default_factory=lambda: _env("MIN_TRACK_S", 0.5))
    max_uncertainty_mph: float = field(default_factory=lambda: _env("MAX_UNCERTAINTY_MPH", 3.0))
    parked_mph: float = field(default_factory=lambda: _env("PARKED_MPH", 3.0))
    min_fps: float = field(default_factory=lambda: _env("MIN_FPS", 5.0))

    # The detector via OpenVINO: "cpu" (works anywhere) or "intel:gpu".
    yolo_device: str = field(default_factory=lambda: _env("YOLO_DEVICE", "cpu"))
    yolo_confidence: float = field(default_factory=lambda: _env("YOLO_CONFIDENCE", 0.25))
    # GPU_QUEUE_THROTTLE for every GPU compile: LOW stops the waiting thread
    # busy-spinning (~45 -> ~8 ms of CPU per inference). "" switches it off.
    yolo_gpu_throttle: str = field(default_factory=lambda: _env("YOLO_GPU_THROTTLE", "LOW"))
    # OpenVINO's compiled-model cache (vision.compile_cache): a restart loads
    # the compiled model instead of compiling it again. Unset: DATA_DIR/ov-cache
    # (the data volume, so it outlives the container). "" switches it off.
    ov_cache_dir: Path | None = field(default_factory=lambda: _ov_cache_dir())
    # Tracking runs YOLO from this long before the gate's span to this long
    # after it. 2.5/2.5 is the original; the GPU-impact spike measured 1.0/0.5
    # identical on 42 cars (finding 3), but 0.5 s after a single-sample
    # episode lost a car (finding 15); 1.0 s after keeps it (finding 16).
    track_pad_before_s: float = field(default_factory=lambda: _env("TRACK_PAD_BEFORE_S", 2.5))
    track_pad_after_s: float = field(default_factory=lambda: _env("TRACK_PAD_AFTER_S", 2.5))
    # The motion pre-gate (Vision.motion_gate; GPU-impact spike): off (today's
    # gate) | shadow (today's gate decides; the motion gate also runs, both
    # are logged, and a clip they disagree on is kept in data/shadow) | on.
    # Anything else is off.
    motion_gate: str = field(default_factory=lambda: _env("MOTION_GATE", "off"))
    # Episodes (GPU-impact spike): off (track the gate's span, padded) | on
    # (track only within the TRACK_PAD_* of each moving sample the gate saw,
    # a fresh tracker each). Anything else is off.
    episodes: str = field(default_factory=lambda: _env("EPISODES", "off"))

    # Night cars by their headlights (lights.py), tuned on a labelled night of
    # the reference install (docs/decisions/0009).
    # off | log (judge and log every track, store nothing) | on.
    night_lights: str = field(default_factory=lambda: _env("NIGHT_LIGHTS", "on"))
    # Near-lane boundary, image y at x=0 and x=1920: a track is near-lane when
    # its median y is below (greater than) the line. Far-lane lights measured
    # y 410-421 at x 1174-1782 (line 434-439); the nearest near-lane car is
    # 10.7 px below the line, and one high-lit truck falls just above it.
    # Unset (the default), only the calibration's night lane applies, and
    # without one every light on the road counts.
    night_lane_y0: float | None = field(default_factory=lambda: _opt_float("NIGHT_LANE_Y0"))
    night_lane_y1: float | None = field(default_factory=lambda: _opt_float("NIGHT_LANE_Y1"))
    # Fractions of the road outline's x range a light's moving run must cover:
    # to be a car (0.6, ~7 m), and to show its speed.
    night_car_cover: float = field(default_factory=lambda: _env("NIGHT_CAR_COVER", 0.6))
    night_speed_cover: float = field(default_factory=lambda: _env("NIGHT_SPEED_COVER", 0.8))
    # "Steady": the Theil–Sen fit's robust sigma, metres.
    night_max_resid_m: float = field(default_factory=lambda: _env("NIGHT_MAX_RESID_M", 0.3))
    # To be a car at all, a light's moving run must average this (net world
    # distance over its duration): it keeps walkers (1.5 m/s, 3.4 mph) and
    # runners out and slow cyclists in (2026-11-02). The slowest car in
    # the labelled set averages 12.2 mph (n0802), so no labelled car is lost.
    night_car_min_mph: float = field(default_factory=lambda: _env("NIGHT_CAR_MIN_MPH", 7.0))
    # A car track is a far-lane light's reflection on a wet road when a
    # lane-failed track moves with it, overlapping in time, its path within
    # this many metres. A light and its own reflection share an x (0 m); one
    # car's front and rear lights are ≥ 4.0 m apart in the labelled set.
    night_reflection_m: float = field(default_factory=lambda: _env("NIGHT_REFLECTION_M", 1.0))
    night_min_mph: float = field(default_factory=lambda: _env("NIGHT_MIN_MPH", 5.0))
    night_max_mph: float = field(default_factory=lambda: _env("NIGHT_MAX_MPH", 60.0))
    # Two car tracks moving together up to max apart are one car; from min
    # up, the separation is its length (front and rear lights). Measured one
    # car's pairs at 4.0-6.5 m (08:16:02, a long vehicle, at 6.1-6.5), and
    # the distinct cars in a case 100 m or more apart; so 7.0, not 6.0. The
    # cost: two cars following ≤ 7 m apart (slow, close) count as one.
    night_companion_min_m: float = field(default_factory=lambda: _env("NIGHT_COMPANION_MIN_M", 3.5))
    night_companion_max_m: float = field(default_factory=lambda: _env("NIGHT_COMPANION_MAX_M", 7.0))
    # The stitch (plan 2026-11-08-night-split-stitch): the street tree's trunk
    # at x≈1235 cuts a car's glow in two for a frame or two, and each piece
    # covers too little of the road for a speed. "on" joins a car that lacks
    # only coverage with an in-lane piece that continues its path: within
    # gap_s in time and m at the handover. It never changes a shown speed.
    night_stitch: str = field(default_factory=lambda: _env("NIGHT_STITCH", "off"))
    night_stitch_gap_s: float = field(default_factory=lambda: _env("NIGHT_STITCH_GAP_S", 0.3))
    night_stitch_m: float = field(default_factory=lambda: _env("NIGHT_STITCH_M", 1.0))

    # Durable log: a directory bind-mounted from the host (~/safe_streets-logs),
    # rotated at midnight. Unset: stdout only.
    log_dir: str = field(default_factory=lambda: _env("LOG_DIR", ""))
    log_keep_days: int = field(default_factory=lambda: _env("LOG_KEEP_DAYS", 30))

    build_sha: str = field(default_factory=lambda: _env("BUILD_SHA", "dev"))

    def require(self) -> None:
        """Refuse to start without the settings only the operator can give."""
        problems = []
        if self.speed_limit is None:
            problems.append("SPEED_LIMIT is required: the street's limit, in UNITS (e.g. SPEED_LIMIT=20)")
        if self.units not in UNIT_MPS:
            problems.append(f"UNITS must be mph or kmh, not {self.units!r}")
        if self.auth_mode not in ("password", "none"):
            problems.append(f"AUTH_MODE must be password or none, not {self.auth_mode!r}")
        if self.auth_mode == "none" and self.base_url:
            problems.append("AUTH_MODE=none is for trying it on one machine, with no proxy and no BASE_URL: "
                            "unset BASE_URL, or use AUTH_MODE=password")
        if not self.tz:
            problems.append("TZ is required: the camera's time zone (e.g. TZ=America/Los_Angeles)")
        else:
            try:
                ZoneInfo(self.tz)
            except (ZoneInfoNotFoundError, ValueError):
                problems.append(f"TZ {self.tz!r} isn't a time zone (e.g. TZ=America/Los_Angeles)")
        if problems:
            raise SystemExit("safe_streets can't start: " + "; ".join(problems))

    @property
    def unit_label(self) -> str:
        return "km/h" if self.units == "kmh" else "mph"

    def speed(self, mps: float | None) -> float | None:
        """A speed in m/s as shown, in UNITS."""
        return None if mps is None else mps / UNIT_MPS.get(self.units, UNIT_MPS["mph"])

    @property
    def speed_limit_mps(self) -> float:
        return (self.speed_limit or 0.0) * UNIT_MPS.get(self.units, UNIT_MPS["mph"])

    @property
    def speed_limit_mph(self) -> float:
        """The limit in mph, for the analysis, which works in mph inside."""
        return self.speed_limit_mps / UNIT_MPS["mph"]

    @property
    def db_path(self) -> Path:
        return self.data_dir / "safe_streets.db"
