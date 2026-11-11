# 0009: At night, find cars by their headlights

Status: accepted (reference install, 2026-11).

## Context

At night the reference camera switches to infrared greyscale. A dark car is
then a dark shape in its own glare, and the detector doesn't find it at any
crop or threshold. Every car does carry a saturated headlight blob, which
crosses the frame in a straight, steady line.

## Decision

At night: threshold each frame's grey (≥ 245) in the road band, keep blobs
of a plausible size, link them frame to frame, and judge the tracks with
**precision over recall**: when in doubt, don't store a car. A track that
covers enough of the road (60% of the outline's width) is a car; it shows a
speed, always marked rough, only if it covers more (80%) and moves steadily.
Otherwise the car is counted without a speed. The lane is a line in image
space (0005).

## Consequences

- Night speeds are rough by design: a headlight is not a wheel, and the
  calibration is on the ground.
- Occluders split a light's track: see 0010.
