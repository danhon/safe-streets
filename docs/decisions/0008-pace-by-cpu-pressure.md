# 0008: Pace the work by CPU pressure, not the load average

Status: accepted (reference install, 2026-11).

## Context

On a shared box, analysis can stall other services (on the reference
install, a home-automation bridge's video). The first gate used the load
average, but on that box it sat at 3–5 all day from disk and GPU waits,
which aren't CPU contention: a 4.0 threshold held events back for about an
hour at a time.

## Decision

Before every event, read Linux's pressure stall information,
`/proc/pressure/cpu` `some avg60`: the share of the last minute that some
task waited for a CPU. Work runs back to back while it's under
`MAX_CPU_PRESSURE` (60%; normal days ran 6–40%), and retries every 30 s
when it isn't. After about 30 busy minutes a batch runs anyway. The app
runs at nice 10, tests at nice 19. `MAX_LOAD` is the fallback where PSI
can't be read.

## Consequences

- Off by default for new installs (a dedicated box doesn't need it).
- The work queue lives in the database, so a backlog survives restarts and
  deploys.
