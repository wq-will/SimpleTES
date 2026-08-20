# poliastro source map

Upstream reference: poliastro 0.17.0, commit
`c7d12e9b715d3fd60f2be233af707d5b97617d39`, MIT License.

- Repository/tag: https://github.com/poliastro/poliastro/tree/v0.17.0
- License: https://github.com/poliastro/poliastro/blob/v0.17.0/COPYING
- Core sources: https://github.com/poliastro/poliastro/tree/v0.17.0/src/poliastro/core

| Local file | Upstream source | Treatment |
|---|---|---|
| `OrbDyn/angles.py` | `src/poliastro/core/angles.py` | Copied, then adjusted for Numba caching and this package's import context. |
| `OrbDyn/elements.py` | `src/poliastro/core/elements.py`, `src/poliastro/_math/linalg.py`, `src/poliastro/core/util.py` | Modified copy; includes in-tree `norm` and `rotation_matrix`, SMA helpers, vectorized helpers, and caching changes. |
| `OrbDyn/prop.py` | `src/poliastro/core/propagation/farnocchia.py` | Modified copy using package-relative imports; adds `farnocchia_rv_series`. |
| `OrbDyn/lambert.py` | `src/poliastro/core/iod.py`, `src/poliastro/_math/special.py` | Modified/combined copy; adds public dispatch and solution-enumeration helpers. |

Key derived function groups are:

- `angles.py`: all anomaly-conversion and Newton-solver functions.
- `elements.py`: `eccentricity_vector`, `circular_velocity`, `rv_pqw`,
  `coe_rotation_matrix`, `coe2rv`, `coe2rv_many`, `coe2mee`, `rv2coe`,
  `mee2coe`, and `mee2rv`; added and adapted helpers are summarized above.
- `prop.py`: the near-parabolic helpers, `delta_t_from_nu`,
  `nu_from_delta_t`, `farnocchia_coe`, and `farnocchia_rv`.
- `lambert.py`: `vallado`, `izzo`, and their internal reconstruction,
  time-of-flight, initial-guess, Halley, and Householder helpers; Stumpff and
  hypergeometric helpers come from poliastro's math module.

`OrbDyn/gravity.py` is an independent implementation authored by Laboratory of
Aerospace Dynamics (LAD), Tsinghua University. It is not derived from
poliastro and is therefore outside this poliastro source map.
