# 0014: Fight motion blur with the shutter, not the resolution

Status: accepted (reference install, 2026-11).

## Context

A car at 30 mph moves about 13 m a second. With a slow shutter (the
reference camera's low-light mode), a car in low sun is a smear: the box
edges soften, and anything that reads the picture (a vehicle-type model)
gives up.

## Decision

Prefer a camera mode with a faster shutter wherever the light allows. On
the reference camera, its `sport` mode let the vehicle-type model name 80%
of low-sun passes, against 44% in its slow-shutter mode (25 passes,
p ≈ 0.0006), for about 15% more CPU and 17% more GPU per car. Test modes in
matched windows (weekday against weekday) before switching, and know that
switching some cameras' mode reboots them.
