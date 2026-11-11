# 0004: Say less rather than something wrong

Status: accepted (reference install, 2026-11).

## Context

A confidently wrong speed is worse than none: a parked car shown at 40 mph,
or two cars merged into one fast one. People will quote the numbers.

## Decision

Every speed carries a confidence label, and the doubtful ones say why:

- **good**: no caveat and ± ≤ 2 mph;
- **fair**: no caveat, wider ±;
- **rough**: a caveat about the track or clip: a short track (under
  `MIN_TRACK_M` 4 m or `MIN_TRACK_S` 0.5 s), a wide ± (over
  `MAX_UNCERTAINTY_MPH` 3), a low frame rate, a low-res clip, a night track
  (0009), or a position outside where the scale was checked. Rough speeds
  show as "~27".

Tracks slower than `PARKED_MPH` (3) are parked and dropped. Night cars that
can't show a speed are counted without one. When the camera has moved,
speeds pause (0011).

## Consequences

- Nothing is silently hidden; the flags are stored, so thresholds can be
  tuned from real data.
- Public views (0013) use the confidence label too.
