# Contributing

Thanks for helping. A few rules keep the project safe to publish and easy to
reuse.

## Sign your commits (DCO)

Every commit carries a `Signed-off-by:` line (`git commit -s`), certifying
the [Developer Certificate of Origin](https://developercertificate.org/):
that you wrote the change or have the right to submit it under the
project's MIT licence. There is no CLA.

## Never commit footage

No clips, frames, snapshots or screenshots of a real street, ever, and
nothing that identifies a real install (addresses, coordinates, camera
names, hostnames). Tests and docs use the synthetic scene
(`safe_streets.synth`). Pictures in `docs/img/` must be synthetic and
listed in `docs/img/SYNTHETIC`. Run `make scan` before you push.

## Licences

The project is MIT. Dependencies must be permissive: no GPL or AGPL
package, and no model weights under a research-only or non-commercial
licence. CI checks the resolved dependencies (`make licences`).

## In scope

Measuring vehicle speeds from a fixed camera's recordings, calibrating
that, and showing the results with privacy-preserving defaults. Out of
scope for now: posting to social networks, live-stream analysis, and
identifying vehicles or people.

## Tests

`make test` runs the suite at low priority. Write the failing test first.
