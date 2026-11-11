# 0010: A change that could move results must reproduce them first

Status: accepted (reference install, 2026-11).

## Context

Several changes were meant to make things cheaper or fill gaps without
changing any existing speed: one decode instead of two at night, the
motion gate, joining night tracks split by a tree trunk. Each could
silently move speeds.

## Decision

Before such a change goes live, replay stored clips through old and new
code and compare. The bar is set before the run:

- **Pure optimisations** must reproduce results exactly, bit for bit (one
  decode at night: exact on 110 clips; the merged code: exact on 83).
- **Gap-fillers** may only add: the night stitch joins two pieces of one
  light across an occluder only if the joined track shows a speed, and keeps
  the stitched result only if it finds the same cars with every shown speed
  unchanged. Replayed on 369 night events, it gave 6 passes a speed, each
  within 0.8 mph of its own piece's.
- **Behaviour changes** run in `shadow` first, logging both answers.

## Consequences

- The bench keeps a fixed set of replay clips and production's stored
  results for them.
