# 0003: A robust straight-line fit, and its spread as the ±

Status: accepted (reference install, 2026-11).

## Context

A car past a house moves at a near-constant speed over the 10–20 m the
camera sees. The positions are noisy: boxes jitter, wheels are hidden for a
frame, a tracker sometimes jumps.

## Decision

Map each sample's ground point through the calibration's homography onto the
road plane, take the distance along the road, and fit distance against time
with **Theil–Sen**: the median of the slopes between every pair of samples
at least 0.2 s apart. The slope is the speed; its sign is the direction.

The **±** is twice the slope's standard error, using a robust sigma of the
residuals (1.4826 × their median absolute deviation) over the spread of the
sample times. It's shown with every speed.

## Consequences

- One or two bad samples barely move the speed; a track that is mostly bad
  shows a wide ±, and gets flagged (0004).
- Over ~1 s of track, a 1 mph error is about 0.45 m: calibration matters
  more than the fit (see the calibration guide).
