# 0005: Filter by lane, never by direction alone

Status: accepted (reference install, 2026-11).

## Context

A street has two lanes, and from the side the far lane is small, oblique and
partly hidden. Speeds there are much rougher. Dropping "cars going left"
would remove the far lane on a right-hand-traffic street, but it would also
drop a car going the wrong way in the near lane, which is exactly the kind
of pass worth recording.

## Decision

Scope is a lane, defined in image space at calibration: the road outline
where wheels may be (by day) and a lane line (by night, where headlights
stand in for wheels). Direction is reported, never used to filter.

## Consequences

- The calibration has to define the lane well; tall vehicles' headlights
  sit higher than a car's and can cross a night lane line.
- Wrong-way passes in the near lane are kept and shown.
