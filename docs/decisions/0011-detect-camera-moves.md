# 0011: Notice when the camera has moved, and pause speeds

Status: accepted (reference install, 2026-11).

## Context

Cameras get bumped, cleaned and re-aimed. The reference camera moved twice
in its first weeks. A moved camera makes every speed wrong while everything
looks fine.

## Decision

About once an hour, match ORB features between a fresh snapshot and the
calibration's reference frame, fit a shift, rotation and scale with
**RANSAC**, and read the shift. Over a few pixels, the calibration is
marked stale and new passes are stored without a speed until it's
recalibrated; the page says so. The median of all matches isn't used:
moved shadows and cars gave a 9.8 px median on a camera that hadn't moved,
where RANSAC read 0.6 px.

## Consequences

- Stored clips can be re-analysed after a recalibration.
