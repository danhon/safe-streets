#!/usr/bin/env python3
"""Check a repo is safe to make public: no footage, nothing that identifies
the reference install, no private addresses, no secrets.

  python3 scripts/prepublish.py [REPO] [--denylist FILE] [--history]

Exits 1 and prints every hit (file:line and what matched, never the
denylist term itself, so the output is safe to paste) when it finds:

- an image or video file anywhere except docs/img/, and in docs/img/ any
  file not named in docs/img/SYNTHETIC (one file name a line: pictures made
  from the synthetic scene, never camera footage);
- a denylist term (default ~/.config/safe-streets/denylist.txt, kept outside
  every repo): plain lines match case-insensitively as substrings, `re:`
  lines are regexes; LICENSE is exempt, as its copyright line names the
  author, and so is the repository's own address (github.com/OWNER/safe-streets);
- a private IPv4 address (10/8, 172.16/12, 192.168/16, 100.64/10);
- a secret, if gitleaks is installed (`gitleaks detect`).

A line carrying the marker `prepublish: allow` is skipped (for this
script's own patterns and its tests' example addresses); use it sparingly.

With --history it checks every blob in the git history too, not only the
working tree: a file deleted later is still public once pushed.
Standard library only.
"""

from __future__ import annotations

import argparse
import ipaddress
import re
import shutil
import subprocess
import sys
from pathlib import Path

MEDIA = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".heic",
         ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".ts", ".h264", ".hevc"}
EXEMPT = {"LICENSE", "LICENSE.md", "LICENSE.txt"}
IPV4 = re.compile(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b")
PRIVATE = [ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10")]  # prepublish: allow
ALLOW = "prepublish: allow"  # a line carrying this marker is skipped
# The repository's own public address names its owner; it isn't a leak.
OWN_ADDRESS = re.compile(r"github\.com/[\w.-]+/safe-streets\b")
DEFAULT_DENYLIST = Path.home() / ".config" / "safe-streets" / "denylist.txt"


def load_denylist(path: Path) -> list[tuple[str, re.Pattern]]:
    """(label, pattern) pairs. The label is the term's line number, so hits
    can be reported without printing the term."""
    out = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pat = re.compile(line[3:]) if line.startswith("re:") else re.compile(re.escape(line), re.I)
        out.append((f"denylist line {n}", pat))
    return out


def private_ips(text: str) -> list[str]:
    hits = []
    for m in IPV4.finditer(text):
        try:
            ip = ipaddress.ip_address(m.group(0))
        except ValueError:
            continue
        if any(ip in net for net in PRIVATE):
            hits.append(m.group(0))
    return hits


def check_media(rel: str, synthetic: set[str]) -> str | None:
    p = Path(rel)
    if p.suffix.lower() not in MEDIA:
        return None
    if p.parts[:2] == ("docs", "img") and p.name in synthetic:
        return None
    where = "not listed in docs/img/SYNTHETIC" if p.parts[:2] == ("docs", "img") else "outside docs/img/"
    return f"{rel}: image or video {where}"


def check_text(rel: str, data: bytes, deny) -> list[str]:
    if b"\0" in data[:4096]:
        return []  # binary
    text = data.decode("utf-8", errors="replace")
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        if ALLOW in line:
            continue
        if Path(rel.split(" (in history")[0]).name not in EXEMPT:
            bare = OWN_ADDRESS.sub("", line)
            for label, pat in deny:
                if pat.search(bare):
                    hits.append(f"{rel}:{i}: matches {label}")
        for ip in private_ips(line):
            hits.append(f"{rel}:{i}: private address {ip}")
    return hits


def tree_files(repo: Path) -> list[tuple[str, bytes]]:
    """Every file git would publish from the working tree (tracked or not
    ignored), with its contents."""
    names = subprocess.run(["git", "-C", str(repo), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                           capture_output=True, check=True).stdout.split(b"\0")
    out = []
    for n in names:
        if n:
            rel = n.decode()
            path = repo / rel
            if path.is_file():
                out.append((rel, path.read_bytes()))
    return out


def history_files(repo: Path) -> list[tuple[str, bytes]]:
    """Every (path, blob) in every commit's tree, each blob once."""
    out, seen = [], set()
    revs = subprocess.run(["git", "-C", str(repo), "rev-list", "--all"], capture_output=True, text=True,
                          check=True).stdout.split()
    for rev in revs:
        listing = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "-z", rev], capture_output=True,
                                 check=True).stdout.split(b"\0")
        for entry in listing:
            if not entry:
                continue
            meta, path = entry.split(b"\t", 1)
            _, kind, sha = meta.split()
            if kind != b"blob" or sha in seen:
                continue
            seen.add(sha)
            blob = subprocess.run(["git", "-C", str(repo), "cat-file", "blob", sha.decode()], capture_output=True,
                                  check=True).stdout
            out.append((f"{path.decode()} (in history, {sha.decode()[:8]})", blob))
    return out


def scan(repo: Path, denylist: Path, history: bool = False) -> list[str]:
    deny = load_denylist(denylist)
    files = tree_files(repo) + (history_files(repo) if history else [])
    syn_path = repo / "docs" / "img" / "SYNTHETIC"
    synthetic = set(syn_path.read_text().split()) if syn_path.is_file() else set()
    hits = []
    for rel, data in files:
        name = rel.split(" (in history")[0]
        m = check_media(name, synthetic)
        if m:
            hits.append(m if name == rel else f"{rel}: image or video")
        hits += check_text(rel, data, deny)
    if shutil.which("gitleaks"):
        r = subprocess.run(["gitleaks", "detect", "--source", str(repo), "--no-banner", "--redact"]
                           + ([] if history else ["--no-git"]), capture_output=True, text=True)
        if r.returncode != 0:
            hits.append("gitleaks found secrets:\n" + (r.stdout + r.stderr).strip())
    else:
        hits.append("gitleaks isn't installed: secrets were not checked")
    return hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", nargs="?", type=Path, default=Path("."))
    ap.add_argument("--denylist", type=Path, default=DEFAULT_DENYLIST)
    ap.add_argument("--history", action="store_true")
    a = ap.parse_args(argv)
    if not a.denylist.is_file():
        print(f"no denylist at {a.denylist}: refusing to call the repo clean", file=sys.stderr)
        return 2
    hits = scan(a.repo.resolve(), a.denylist, a.history)
    for h in hits:
        print(h)
    print(f"{len(hits)} problem(s)" if hits else "clean")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
