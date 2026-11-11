# 0001: Speeds come from each frame's own timestamp

Status: accepted (reference install, 2026-11).

## Context

Consumer cameras don't record at a fixed frame rate. The reference camera
nominally records about 25 fps, but its clips ran anywhere from 8.7 to 20
fps, and the rate changes within a clip as the exposure changes (at night
especially). A speed worked out as "frames × 1/25 s" is wrong by the same
factor as the frame rate is.

## Decision

Every sample carries the presentation timestamp (pts) of the frame it came
from, read from the container, and the speed is fitted against those times.
A nominal or average frame rate is never used for a speed. Clips under
`MIN_FPS` (5) are still measured but flagged "low frame rate" (a rough
speed).

The frame times have to be the decoder's, exactly. On night clips a speed
can turn on the last bit of a time: reading the same clip's times through a
second library moved one speed by 0.02 mph. So one decoder (OpenCV's FFmpeg
backend) provides the times everywhere, and any optimisation that decodes a
different way has to reproduce them bit for bit.

## Consequences

- The tests use clips with deliberately uneven frame times.
- A source must supply clips with real per-frame timestamps. Re-encoded
  clips with a forced constant rate destroy the information.
