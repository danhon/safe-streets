"""What the Dockerfile and the compose files promise: the source folder is
only read, the GPU only with its override, no reverse proxy's labels in the
default file, a health check on /healthz, and the optional recorder copies
the camera's stream into the folder the app reads (never re-encodes it)."""

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
ROOT = Path(__file__).resolve().parent.parent


def _load(name: str) -> dict:
    return yaml.safe_load((ROOT / name).read_text())


def _app() -> dict:
    return _load("compose.yml")["services"]["app"]


def _mounts(service: dict) -> list[dict]:
    """Each volume entry as {source, target, ro}, short or long syntax."""
    out = []
    for v in service.get("volumes", []):
        if isinstance(v, str):
            # source:/target[:mode]; the source may hold ${VAR:-default}.
            m = re.fullmatch(r"(.*?):(/[^:]*)(?::([\w,]+))?", v)
            out.append({"source": m[1], "target": m[2], "ro": bool(m[3]) and "ro" in m[3].split(",")})
        else:
            out.append({"source": v.get("source"), "target": v["target"], "ro": bool(v.get("read_only"))})
    return out


def test_no_gpu_without_the_override():
    c = _load("compose.yml")
    for svc in c["services"].values():
        assert "devices" not in svc and "group_add" not in svc
    assert "/dev/dri" not in (ROOT / "compose.yml").read_text()
    gpu = _load("compose.gpu.yml")["services"]["app"]
    assert any("/dev/dri" in d for d in gpu["devices"])
    assert gpu["group_add"]
    env = gpu["environment"]
    assert "YOLO_DEVICE=intel:gpu" in env or (isinstance(env, dict) and env.get("YOLO_DEVICE") == "intel:gpu")


def test_the_source_folder_is_mounted_read_only():
    m = [v for v in _mounts(_app()) if v["target"] == "/footage"]
    assert len(m) == 1 and m[0]["ro"]
    assert "SOURCE_DIR=/footage" in _app()["environment"]


def test_the_default_compose_has_no_reverse_proxy_labels():
    for svc in _load("compose.yml")["services"].values():
        assert not [lb for lb in svc.get("labels", []) if "traefik" in str(lb)]
    assert "traefik" in (ROOT / "examples" / "compose.traefik.yml").read_text()


def test_the_app_publishes_one_port_and_checks_healthz():
    app = _app()
    assert app["ports"] == ["8080:8080"]
    hc = " ".join(app["healthcheck"]["test"])
    assert "/healthz" in hc and "8080" in hc


def test_the_recorder_is_opt_in_copies_the_stream_and_feeds_the_app():
    c = _load("compose.yml")
    rec = c["services"]["recorder"]
    assert rec["profiles"] == ["recorder"]
    cmd = " ".join(rec["command"]) if isinstance(rec["command"], list) else rec["command"]
    script = (ROOT / "recorder" / "record.sh").read_text()
    both = cmd + "\n" + script
    assert "-c copy" in both and "-segment_time" in both and "-reset_timestamps 0" in both and "-strftime 1" in both
    assert "-rtsp_transport tcp" in both
    assert not [w for w in ("libx264", "-c:v lib", "-vf ", "-r ") if w in both]
    writes = [v for v in _mounts(rec) if not v["ro"]]
    reads = [v for v in _mounts(_app()) if v["target"] == "/footage"]
    assert writes and writes[0]["source"] == reads[0]["source"]
    assert "RECORDER_KEEP_HOURS" in both and "48" in both


def test_the_image_runs_as_a_user_at_nice_10_with_the_checked_detector():
    d = (ROOT / "Dockerfile").read_text()
    assert "\nUSER " in d and '"nice", "-n", "10"' in d
    assert "fetch_detector.py" in d and "/dev/dri" not in d
    assert "ultralytics" not in d


def _fetch(*args):
    import subprocess
    import sys
    return subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "fetch_detector.py"), *map(str, args)],
                          capture_output=True, text=True)


def test_fetch_detector_refuses_a_model_that_isnt_the_pinned_one(tmp_path):
    src = tmp_path / "in" / "rtmdet-tiny"
    src.mkdir(parents=True)
    (src / "model.xml").write_text("<net/>")
    (src / "model.bin").write_bytes(b"\0" * 16)
    out = _fetch("--from", tmp_path / "in", tmp_path / "out")
    assert out.returncode != 0 and "sha256" in out.stderr
    assert not (tmp_path / "out" / "rtmdet-tiny" / "model.xml").exists()


def test_fetch_detector_with_nothing_to_fetch_says_how(tmp_path):
    out = _fetch("--from", tmp_path / "none", tmp_path / "out")
    assert out.returncode != 0 and "DETECTOR_URL" in out.stderr


@pytest.mark.skipif(not (ROOT / "models" / "rtmdet-tiny" / "model.bin").exists(), reason="needs models/rtmdet-tiny")
def test_fetch_detector_copies_the_pinned_model(tmp_path):
    out = _fetch("--from", ROOT / "models", tmp_path / "out")
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "out" / "rtmdet-tiny" / "model.bin").stat().st_size > 1_000_000
    assert (tmp_path / "out" / "rtmdet-tiny" / "LICENSE").exists()


def test_every_doc_the_repo_points_to_exists():
    import subprocess
    files = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True).stdout.split()
    missing = set()
    for f in files:
        p = ROOT / f
        if p.suffix in (".py", ".md", ".html", ".yml", ".sh", ".example", ".toml") or p.name == ".env.example":
            for ref in re.findall(r"\bdocs/[\w./-]+\.md\b", p.read_text(errors="replace")):
                if not (ROOT / ref).is_file():
                    missing.add(f"{f}: {ref}")
    assert not missing, sorted(missing)


def test_a_fresh_clone_builds_from_the_release_asset():
    """With no models/ folder, the build downloads the release's archive (and
    checks its sha256), so a fresh clone builds with nothing else."""
    d = (ROOT / "Dockerfile").read_text()
    m = re.search(r'^ARG DETECTOR_URL="?([^"\n]*)"?$', d, re.M)
    assert m and m[1].startswith("https://github.com/") and "/releases/download/" in m[1]
    assert m[1].endswith("/rtmdet-tiny.tar.gz")


def test_every_yaml_file_parses():
    """GitHub runs no job at all from a workflow it can't read (an unquoted
    ': ' in a step's name did that)."""
    files = sorted(p for p in ROOT.rglob("*.y*ml") if ".venv" not in p.parts and p.suffix in (".yml", ".yaml"))
    assert ROOT / ".github" / "workflows" / "test.yml" in files
    class Loader(yaml.SafeLoader):  # Compose's own tags (!reset, !override) are valid
        pass
    Loader.add_multi_constructor("!", lambda loader, tag, node: None)
    for p in files:
        yaml.load(p.read_text(), Loader=Loader)


def test_ci_installs_ffmpeg():
    """The folder source's tests need ffmpeg; GitHub's runner hasn't got it."""
    assert "install -y -q ffmpeg" in (ROOT / ".github" / "workflows" / "test.yml").read_text()
