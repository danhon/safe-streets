"""The GPU time this process has used, from the kernel's per-client DRM
stats: each open /dev/dri fd's fdinfo has `drm-engine-<class>: <ns>`, the
time that client's work spent on each engine class (i915, kernel 5.19+).
It's what the host's GPU sampler reads per process (~/freeze-log), so the
event line's *_gpu_ms agree with it. YOLO on the GPU runs on the render
engine; VA-API decode would add the video engine, so every engine counts.

Events run one at a time and nothing else in the app uses the GPU, so the
difference across a stage is that stage's GPU time.
"""

import os
from pathlib import Path

SELF = Path("/proc/self")


def busy_ns(proc: Path = SELF) -> int | None:
    """Nanoseconds of GPU engine time this process's DRM clients have used,
    or None where there is none to read (no GPU open, or a kernel without
    per-client stats). A client open on two fds counts once."""
    clients: dict[str, int] = {}
    try:
        fds = os.listdir(proc / "fd")
    except OSError:
        return None
    for fd in fds:
        try:
            if not os.readlink(proc / "fd" / fd).startswith("/dev/dri/"):
                continue
            text = (proc / "fdinfo" / fd).read_text()
        except OSError:
            continue  # closed since the listing
        cid, ns, seen = None, 0, False
        for line in text.splitlines():
            key, _, value = line.partition(":")
            if key == "drm-client-id":
                cid = value.strip()
            elif key.startswith("drm-engine-") and not key.startswith("drm-engine-capacity-"):
                ns += int(value.split()[0])
                seen = True
        if cid is not None and seen:
            clients[cid] = ns
    return sum(clients.values()) if clients else None
