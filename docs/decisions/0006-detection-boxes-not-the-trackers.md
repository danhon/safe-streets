# 0006: Track with the detector's boxes, not the tracker's predictions

Status: accepted (reference install, 2026-11).

## Context

A tracker such as ByteTrack keeps a Kalman filter per object and, by
default, reports the filter's box. The filter lags a car that's still
coming into view and then catches up, which reads as speed. On a 20 mph
known-speed drive, the filter's boxes read 24.5 mph; the detections read
21.9. Across 97 clips, switching to detections lowered speeds by a median
1.85 mph (7.7%).

## Decision

The tracker only links detections into tracks. Each sample keeps that
frame's detection box (the tracker is driven by hand, so the match is
known), and the speed is fitted to those.

Two more rules came from the same kind of evidence:

- **Follow the leading edge**, half a car back. A parked car often hides the
  back of a passing one; the front is visible from entry to exit. On two
  15 mph drives, front edges gave 604 and 608 px/s (0.7% apart); centres of
  partly hidden boxes were off by up to 40%.
- **Keep the run that moves furthest.** A tracker can keep one ID across a
  car parking, sitting, and another car driving past. The passing car is the
  run of steady movement that covers the most ground. Samples whose leading
  edge touches the crop's edge are dropped.
