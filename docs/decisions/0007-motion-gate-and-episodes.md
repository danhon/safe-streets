# 0007: Run the detector only where the road moved

Status: accepted (reference install, 2026-11).

## Context

Running the detector on every frame of every window was most of the cost,
and the reference box is a 4-core machine shared with other services.

## Decision

Three steps cut the detector's work by about two thirds without changing
the results:

1. **Pads:** track from 1.0 s before the first moving sample to 1.0 s after
   the last (not 2.5 s).
2. **A motion gate:** run the detector only on gate samples where the
   road's brightness actually changed.
3. **Episodes:** track only near each moving sample, with a fresh tracker
   each, not across the whole window.

Per car event, on the CPU with one thread:

| | detector calls | CPU-s |
|---|---|---|
| everything | 243 | 43.3 |
| + shorter pads | 181 | 27.8 |
| + motion gate | 135 | 21.7 |
| + episodes | 79 | 13.8 |

On 200 replayed clips the stack changed 8 clips; with the 1.0 s after-pad,
on 169 more it changed one slow car. A 0.5 s after-pad lost a car, which is
why it's 1.0 s.

## Consequences

- Settings: `MOTION_GATE` off|shadow|on, `EPISODES` off|on,
  `TRACK_PAD_BEFORE_S`/`TRACK_PAD_AFTER_S`. `shadow` runs both gates and
  keeps the clips they disagree on, to check a change on live data first.
- A cheaper model (INT8) was tried as well and rejected: it changed 79 of
  200 clips. Changing the model moves box edges, so it moves speeds.
