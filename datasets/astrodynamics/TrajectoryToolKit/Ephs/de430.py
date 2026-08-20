"""DE430 planetary ephemeris query via SPICE.

Provides heliocentric position/velocity of solar-system planets given an
epoch in Modified Julian Date (UTC).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import spiceypy as spice

_KERNEL_DIR = Path(__file__).resolve().parents[2] / "data" / "ephemerides"
_SPK_FILE = _KERNEL_DIR / "de430.bsp"
_LSK_FILE = _KERNEL_DIR / "naif0012.tls"

# Guide uses planet_id strings matching NAIF barycenter IDs.
# Earth here means Earth-Moon Barycenter (3); precise enough for patched-conic.
PLANET_IDS = {
    "1": "MERCURY BARYCENTER",
    "2": "VENUS BARYCENTER",
    "3": "EARTH BARYCENTER",
    "4": "MARS BARYCENTER",
    "5": "JUPITER BARYCENTER",
    "6": "SATURN BARYCENTER",
    "7": "URANUS BARYCENTER",
    "8": "NEPTUNE BARYCENTER",
    "9": "PLUTO BARYCENTER",
    "10": "SUN",
}

_kernels_loaded = False


def load_kernels() -> None:
    """Furnish DE430 SPK and leap-second kernels (idempotent)."""
    global _kernels_loaded
    if _kernels_loaded:
        return
    if not _SPK_FILE.exists():
        raise FileNotFoundError(f"Missing SPK kernel: {_SPK_FILE}")
    if not _LSK_FILE.exists():
        raise FileNotFoundError(f"Missing leap-second kernel: {_LSK_FILE}")
    spice.furnsh(str(_LSK_FILE))
    spice.furnsh(str(_SPK_FILE))
    _kernels_loaded = True


def _mjd_utc_to_et(mjd_utc: float) -> float:
    jd_utc = mjd_utc + 2400000.5
    return spice.str2et(f"JD {jd_utc:.15f} UTC")


def planet_state(
    planet_id: Union[str, int],
    mjd_utc: float,
    *,
    frame: str = "J2000",
    center: str = "10",
) -> Tuple[np.ndarray, np.ndarray]:
    """Query heliocentric state of a solar-system planet from DE430.

    Args:
        planet_id: Body ID following the guide convention
            ("3"=Earth, "4"=Mars, "5"=Jupiter, ...). Accepts int or str.
        mjd_utc: Epoch as Modified Julian Date in UTC.
        frame: Reference frame (default "J2000").
        center: Observer body ID. Default "10" (Sun).

    Returns:
        r: ndarray (3,), position in km.
        v: ndarray (3,), velocity in km/s.
    """
    load_kernels()
    pid = str(planet_id)
    if pid not in PLANET_IDS:
        raise ValueError(f"Unknown planet_id {pid!r}; valid: {sorted(PLANET_IDS)}")
    cid = str(center)
    if cid not in PLANET_IDS:
        raise ValueError(f"Unknown center {cid!r}; valid: {sorted(PLANET_IDS)}")

    et = _mjd_utc_to_et(mjd_utc)
    state, _lt = spice.spkezr(PLANET_IDS[pid], et, frame, "NONE", PLANET_IDS[cid])
    state = np.asarray(state, dtype=float)
    return state[:3].copy(), state[3:].copy()
