#!/usr/bin/env python3
"""Fail if any resolved dependency is GPL or AGPL, or is ultralytics.

  uv run python scripts/check_licences.py

Reads `pip-licenses --format=json` for the current environment. LGPL is
allowed (it's linked, not copied in); GPL and AGPL are not, nor is
ultralytics by name, whatever its metadata says.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

BANNED = {"ultralytics": "AGPL-3.0 upstream"}
COPYLEFT = re.compile(r"\b(A?GPL|GNU (Affero )?General Public License)", re.I)
LESSER = re.compile(r"\b(LGPL|Lesser General Public License|Library General Public License)", re.I)


def problems(pkgs: list[dict]) -> list[str]:
    out = []
    for p in pkgs:
        name, lic = p["Name"], p.get("License", "")
        if name.lower() in BANNED:
            out.append(f"{name}: banned by name ({BANNED[name.lower()]})")
        elif COPYLEFT.search(lic) and not LESSER.search(lic):
            out.append(f"{name}: {lic}")
    return out


def main() -> int:
    raw = subprocess.run(["pip-licenses", "--format=json", "--from=mixed"], capture_output=True, text=True,
                         check=True).stdout
    hits = problems(json.loads(raw))
    for h in hits:
        print(h)
    print(f"{len(hits)} licence problem(s)" if hits else "licences ok")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
