# 0002: The source is the continuous recording, not the camera's motion events

Status: accepted (reference install, 2026-11).

## Context

The first build analysed the camera system's own motion events (UniFi
Protect's). It missed cars: one of the two calibration drives produced no
motion event at all, and for a time the motion zone didn't cover the whole
road.

## Decision

Scan the continuous recording instead. Export it in chunks on the camera's
low-resolution channel (640×360), run background subtraction inside the
road outline, and queue every window where something moves. Then export
each window at full resolution and analyse it.

Measured on the reference install: 5 minutes of low-res footage is 14 MB,
exports in 1.4 s and scans in about 4–7 s of CPU.

## Consequences

- Needs a recorder that keeps continuous footage and can export arbitrary
  windows. A folder of segment files (Frigate, Blue Iris, ffmpeg) works as
  well as Protect.
- The scan sees everything that moves, so the detector's gate (0006) has
  more to reject.
- Overlapping windows can contain the same car twice; passes are
  de-duplicated by time and lane.
