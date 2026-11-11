"""Put the default detector (RTMDet-tiny's OpenVINO IR) in DEST/rtmdet-tiny,
checked against the sha256 pinned in src/safe_streets/detector.py. The
Docker build runs it; nothing but the standard library.

  python scripts/fetch_detector.py [--from DIR] [--url URL] DEST

--from DIR: a folder holding rtmdet-tiny/ (your own export, by
  scripts/export_rtmdet.py). Used when it has the model.
--url URL: else, a .tar.gz of rtmdet-tiny/ (model.xml, model.bin, LICENSE),
  such as a safe-streets release's asset (the build's DETECTOR_URL).

A model whose files don't match the pinned sha256 is refused, never copied.
"""
import argparse
import ast
import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DETECTOR_PY = ROOT / "src" / "safe_streets" / "detector.py"


def pinned(detector_py: Path) -> tuple[str, dict[str, str]]:
    """RTMDET_DIR and RTMDET_SHA256, read from detector.py without importing it."""
    found = {}
    for node in ast.parse(detector_py.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("RTMDET_DIR", "RTMDET_SHA256"):
                found[node.targets[0].id] = ast.literal_eval(node.value)
    return found["RTMDET_DIR"], found["RTMDET_SHA256"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(folder: Path, want: dict[str, str]) -> list[str]:
    bad = []
    for name, digest in want.items():
        p = folder / name
        if not p.is_file():
            bad.append(f"{name} is missing")
        elif (got := sha256(p)) != digest:
            bad.append(f"{name}'s sha256 is {got}, not the pinned {digest}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", type=Path)
    ap.add_argument("--url", default="")
    ap.add_argument("--detector-py", type=Path, default=DETECTOR_PY)
    ap.add_argument("dest", type=Path)
    a = ap.parse_args()
    name, want = pinned(a.detector_py)
    with tempfile.TemporaryDirectory() as tmp:
        if a.src is not None and (a.src / name / "model.xml").is_file():
            folder = a.src / name
        elif a.url:
            archive = Path(tmp) / "model.tar.gz"
            with urllib.request.urlopen(a.url, timeout=120) as r, archive.open("wb") as f:
                shutil.copyfileobj(r, f)
            with tarfile.open(archive) as t:
                t.extractall(tmp, filter="data")
            folder = Path(tmp) / name
        else:
            sys.exit(f"no detector to fetch: put {name}/ in the build's models/ folder "
                     "(scripts/export_rtmdet.py), or pass DETECTOR_URL (a release's .tar.gz)")
        bad = check(folder, want)
        if bad:
            sys.exit("refusing the detector: " + "; ".join(bad))
        out = a.dest / name
        out.mkdir(parents=True, exist_ok=True)
        for p in folder.iterdir():
            if p.is_file():
                shutil.copy2(p, out / p.name)
    print(f"detector checked and copied to {out}")


if __name__ == "__main__":
    main()
