# Third-Party Notices for the Astrodynamics Tasks

## poliastro-derived routines

The `angles.py`, `elements.py`, `prop.py`, and `lambert.py` files under
`TrajectoryToolKit/OrbDyn/` contain a reduced and modified subset derived from
poliastro 0.17.0, tag commit
`c7d12e9b715d3fd60f2be233af707d5b97617d39`.

poliastro is distributed under the MIT License.  Its preserved license and
copyright notice are in `TrajectoryToolKit/LICENSE`.  Files containing derived
code remain MIT-licensed and identify modifications in this attribution order:

1. Wizard Intelligence Learning Lab (WILL), 2026.
2. Laboratory of Aerospace Dynamics (LAD), Tsinghua University, 2026.

See `TrajectoryToolKit/SOURCE_MAP.md` for the source and function mapping.

`TrajectoryToolKit/OrbDyn/gravity.py` is an independent LAD implementation and
is not included in the poliastro-derived subset.

## Runtime dependencies

NumPy, SciPy, Numba, and SpiceyPy are installed as dependencies and are not
copied into this repository.  Their own package licenses continue to apply.

## NAIF/JPL kernels

The DE430 SPK and NAIF leap-second kernels are not distributed in this
repository.  `download_ephemerides.py` retrieves the exact files from the
NASA/JPL NAIF generic-kernel archive and verifies SHA-256 digests.  Users of
SPICE data should follow NAIF's official attribution and giving-credit guidance.
