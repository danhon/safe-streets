# Decisions

Short records of decisions a newcomer would otherwise relearn the hard way. Each says what was decided, why, and the measurement behind it, from the reference install (one side-on camera over a residential street, 2026-11).

- [0001: Speeds come from each frame's own timestamp](0001-speeds-from-frame-timestamps.md)
- [0002: The source is the continuous recording, not the camera's motion events](0002-continuous-recording-not-motion-events.md)
- [0003: A robust straight-line fit, and its spread as the ±](0003-robust-line-fit-and-its-error.md)
- [0004: Say less rather than something wrong](0004-say-less-rather-than-wrong.md)
- [0005: Filter by lane, never by direction alone](0005-filter-by-lane-not-direction.md)
- [0006: Track with the detector's boxes, not the tracker's predictions](0006-detection-boxes-not-the-trackers.md)
- [0007: Run the detector only where the road moved](0007-motion-gate-and-episodes.md)
- [0008: Pace the work by CPU pressure, not the load average](0008-pace-by-cpu-pressure.md)
- [0009: At night, find cars by their headlights](0009-night-by-headlights.md)
- [0010: A change that could move results must reproduce them first](0010-change-results-only-when-proven.md)
- [0011: Notice when the camera has moved, and pause speeds](0011-detect-camera-moves.md)
- [0012: Every call to the recorder has a timeout](0012-every-external-call-has-a-timeout.md)
- [0013: Local models only, and privacy-preserving public views](0013-local-models-and-privacy.md)
- [0014: Fight motion blur with the shutter, not the resolution](0014-shutter-before-resolution.md)
- [0015: Calibrate by driving past at a known speed](0015-calibrate-by-driving-past.md)
