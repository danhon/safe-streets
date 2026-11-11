# 0012: Every call to the recorder has a timeout

Status: accepted (reference install, 2026-11).

## Context

After a recorder reboot, the reference install's batcher waited about 36
hours on a single clip export: the client library had no timeout on
exports. Nothing failed, so nothing alerted.

## Decision

Every call to the recorder has a timeout (60 s; an export 120 s a try,
three tries). `/healthz` reports when the last batch finished
(`batch_age_s`) and `batch_stale` after 30 minutes without one, so an
outside watchdog can alert. `/healthz` itself always answers 200 while the
process can serve, with the problem in its body.
