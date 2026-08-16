"""Thin wrapper exposing the four primitives evaluator/baseline need.

Units:
- ``ephem`` is fixed to km, km/s, MJD (UTC).
- ``propagate_two_body``, ``lambert``, ``gravity_assist`` are unit-free,
  inherited from OrbDyn. The caller must keep ``mu``, ``r``, ``v``, ``tof``,
  ``r_p`` in a self-consistent system.

The vendored, attributed ``TrajectoryToolKit`` subset lives beside this file.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from TrajectoryToolKit.Ephs import planet_state
from TrajectoryToolKit.OrbDyn.prop import farnocchia_rv
from TrajectoryToolKit.OrbDyn.lambert import lambert as _lambert_solve
from TrajectoryToolKit.OrbDyn.gravity import gravity_assists


class Tools:
    """API for MGA trajectory computation, backed by TrajectoryToolKit primitives."""

    def ephem(
        self, planet_id: str | int, mjd_utc: float, *, center: str = "0"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """State of a solar-system body relative to a center body.

        Planet IDs "1"-"8": DE430 planets (Mercury-Neptune).

        Args:
            planet_id: Body whose state is queried.
            mjd_utc: Epoch as Modified Julian Date in UTC.
            center: Body ID whose state is subtracted.
                "0" (Sun, default) for heliocentric.
                "5" (Jupiter) for Jupiter-centric, etc.

        Returns (r km, v km/s) in J2000, relative to *center*.
        """
        pid = str(planet_id)
        cid = str(center)

        # Heliocentric state of the target body.
        if pid == "0":
            r, v = np.zeros(3), np.zeros(3)
        else:
            r, v = planet_state(pid, mjd_utc)

        if cid == "0":
            return r, v

        # Subtract center's heliocentric state
        rc, vc = planet_state(cid, mjd_utc)
        return r - rc, v - vc

    def propagate_two_body(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        tof: float,
        mu: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Keplerian two-body propagation (Farnocchia).

        Args (unit-free, must be consistent):
            r0, v0: state vectors shape (3,).
            tof: time of flight.
            mu: gravitational parameter of the central body.

        Returns (r, v) at t0 + tof.
        """
        r = np.ascontiguousarray(r0, dtype=float)
        v = np.ascontiguousarray(v0, dtype=float)
        r_f, v_f = farnocchia_rv(float(mu), r, v, float(tof))
        return np.asarray(r_f, dtype=float), np.asarray(v_f, dtype=float)

    def lambert(
        self,
        r0: np.ndarray,
        r1: np.ndarray,
        tof: float,
        mu: float,
        *,
        prograde: bool = True,
        lowpath: bool = True,
        M: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve Lambert's problem (Izzo).

        Args (unit-free, must be consistent):
            r0, r1: departure / arrival position vectors.
            tof: time of flight.
            mu: gravitational parameter.
            prograde: True for inclination < 90 deg.
            lowpath: True for short-way (< 180 deg), False for long-way.
            M: number of full revolutions.

        Returns (v0, v1) departure and arrival velocity vectors.
        """
        r0 = np.ascontiguousarray(r0, dtype=float)
        r1 = np.ascontiguousarray(r1, dtype=float)
        v0, v1 = _lambert_solve(float(mu), r0, r1, float(tof), M=M, prograde=prograde, lowpath=lowpath)
        return np.asarray(v0, dtype=float), np.asarray(v1, dtype=float)

    def gravity_assist(
        self,
        v_in: np.ndarray,
        v_planet: np.ndarray,
        mu_p: float,
        r_p: float,
        psi: float,
    ) -> np.ndarray:
        """Unpowered flyby: rotate v_inf about the planet by deflection set by r_p.

        Args (unit-free, must be consistent):
            v_in: incoming heliocentric spacecraft velocity.
            v_planet: heliocentric velocity of the flyby body.
            mu_p: gravitational parameter of the flyby body.
            r_p: periapsis RADIUS (not altitude).
            psi: B-plane orientation angle (rad).

        Returns v_out: outgoing heliocentric spacecraft velocity.
        """
        v_in = np.ascontiguousarray(v_in, dtype=float)
        v_planet = np.ascontiguousarray(v_planet, dtype=float)
        v_out = gravity_assists(v_in, v_planet, float(mu_p), float(r_p), float(psi))
        return np.asarray(v_out, dtype=float)

    def powered_flyby(
        self,
        v_arr: np.ndarray,
        v_dep: np.ndarray,
        v_planet: np.ndarray,
        mu_p: float,
        min_r_p: float,
    ):
        """Compute powered-flyby parameters from arrival and departure velocities.

        Given Lambert-computed incoming/outgoing heliocentric velocities at a
        flyby planet, compute the required periapsis radius, the velocity
        mismatch dv at periapsis, and whether the flyby is geometrically feasible.

        Args:
            v_arr: heliocentric arrival velocity (from previous Lambert leg).
            v_dep: heliocentric departure velocity (to next Lambert leg).
            v_planet: planet's heliocentric velocity at flyby epoch.
            mu_p: planet gravitational parameter (km³/s²).
            min_r_p: minimum allowed periapsis radius (km), i.e.
                     R_planet + min_flyby_altitude.

        Returns:
            (r_p, dv_flyby, feasible) — all scalars.
        """
        vinf_m = np.asarray(v_arr, dtype=float) - np.asarray(v_planet, dtype=float)
        vinf_p = np.asarray(v_dep, dtype=float) - np.asarray(v_planet, dtype=float)
        n_m = float(np.linalg.norm(vinf_m))
        n_p = float(np.linalg.norm(vinf_p))

        if n_m < 1e-9 or n_p < 1e-9:
            return float("inf"), float("inf"), False

        dot = float(np.dot(vinf_m, vinf_p))
        cos_delta = max(-1.0, min(1.0, dot / (n_m * n_p)))
        delta = float(np.arccos(cos_delta))
        sin_half = float(np.sin(delta / 2.0))

        if sin_half < 1e-12:
            return float("inf"), float("inf"), False

        # Conservative estimate: use the smaller v_inf → larger r_p required
        v_inf = min(n_m, n_p)
        r_p = mu_p / (v_inf * v_inf) * (1.0 / sin_half - 1.0)

        # dv at periapsis to bridge the v_inf magnitude mismatch
        v_peri_m = float(np.sqrt(n_m * n_m + 2.0 * mu_p / r_p))
        v_peri_p = float(np.sqrt(n_p * n_p + 2.0 * mu_p / r_p))
        dv = abs(v_peri_p - v_peri_m)

        feasible = r_p >= min_r_p
        return r_p, dv, feasible
