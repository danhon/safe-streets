# 0013: Local models only, and privacy-preserving public views

Status: accepted (reference install, 2026-11).

## Context

The camera sees a public street: neighbours' cars, people walking, children
on bikes. Frames leaving the house, or a public page full of identifiable
pictures, would be a real harm to people who never agreed to it.

## Decision

- **Every model runs locally.** No frame is sent to a cloud service, for
  detection, classification or anything else.
- **Public views carry no camera pixels by default:** times to the quarter
  hour, published after a delay, groups of fewer than 5 cars shown only as a
  total, the kind as car or two-wheeler, speeds rounded. Shots, clips and
  logs are for the operator, behind a sign-in.
- **Public pages never depend on who is signed in**, so they can be cached
  without leaking the operator's view.

## Consequences

- No plates, no makes and models, no exact times in public.
- Local laws on recording and publishing differ: check yours.
