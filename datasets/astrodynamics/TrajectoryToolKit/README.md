# TrajectoryToolKit subset

This directory contains only the source closure required by the five released
impulsive MGA tasks.  It intentionally excludes J2, modified-equinoctial,
Galilean-moon, cache, and binary-kernel files.

The `angles.py`, `elements.py`, `prop.py`, and `lambert.py` files include code
derived from and modified after poliastro 0.17.0. Those files remain
MIT-licensed; see `LICENSE` and `SOURCE_MAP.md`.
The primary modification attribution is Wizard Intelligence Learning Lab
(WILL), 2026. Additional modifications are attributed to Laboratory of
Aerospace Dynamics (LAD), Tsinghua University, 2026.

`OrbDyn/gravity.py` is an independent implementation copyrighted solely by
Laboratory of Aerospace Dynamics (LAD), Tsinghua University. It is released
under the repository's AGPL-3.0-or-later license and is not derived from
poliastro.

`Ephs/de430.py` is the task's SPICE integration.  Kernel files are downloaded
separately into `../data/ephemerides/` by `../download_ephemerides.py`.
