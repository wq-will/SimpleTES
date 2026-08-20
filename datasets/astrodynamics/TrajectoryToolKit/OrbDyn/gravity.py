# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Laboratory of Aerospace Dynamics (LAD),
# Tsinghua University.

from numpy.linalg import norm
from numpy import cross, arcsin, sin, cos, sqrt, pi
from numba import njit as jit

@jit(cache=True)
def gravity_assists(v_in, v_p, mu_p: float, r_p: float, psi: float):
    """Compute the outgoing velocity after an unpowered gravity assist.

    Parameters
    -------
    v_in : ndarray, shape (3,)
        Incoming spacecraft velocity in the central-body frame.
    v_p : ndarray, shape (3,)
        Flyby-body velocity in the central-body frame.
    mu_p : float
        Standard gravitational parameter of the flyby body.
    r_p : float
        Flyby periapsis radius, measured from the flyby body's center.
    psi : float
        Azimuth of the outgoing excess-velocity vector, in radians.

    Returns
    -------
    ndarray, shape (3,)
        Outgoing spacecraft velocity in the central-body frame.
    """
    v_in_inf = v_in - v_p
    v_in_inf_norm2 = norm(v_in_inf)

    i_axis = v_in_inf / v_in_inf_norm2
    k_axis = cross(i_axis, v_p); k_axis /= norm(k_axis)
    j_axis = cross(k_axis, i_axis)

    delta = 2 * arcsin(
        mu_p /
        (mu_p + r_p * v_in_inf_norm2 **2)
    )

    v_out = v_in_inf_norm2 * (
        i_axis * cos(delta) +
        j_axis * sin(delta) * sin(psi) +
        k_axis * sin(delta) * cos(psi)
    ) + v_p

    return(v_out)


@jit(cache=True)
def ga_rp2delta(r_p: float, mu_p: float, v_in, v_p):
    """Compute the gravity-assist deflection angle from periapsis radius.

    Parameters
    ----------
    r_p : float
        Flyby periapsis radius.
    mu_p : float
        Standard gravitational parameter of the flyby body.
    v_in : ndarray, shape (3,)
        Incoming spacecraft velocity in the central-body frame.
    v_p : ndarray, shape (3,)
        Flyby-body velocity in the central-body frame.

    Returns
    -------
    float
        Gravity-assist deflection angle, in radians.
    """
    v_in_inf = v_in - v_p
    v_inf_norm = norm(v_in_inf)

    delta = 2 * arcsin(
        mu_p /
        (mu_p + r_p * v_inf_norm ** 2)
    )

    return delta


@jit(cache=True)
def ga_delta2rp(delta: float, mu_p: float, v_in, v_p):
    """Compute the flyby periapsis radius from the deflection angle.

    Parameters
    ----------
    delta : float
        Gravity-assist deflection angle, in radians.
    mu_p : float
        Standard gravitational parameter of the flyby body.
    v_in : ndarray, shape (3,)
        Incoming spacecraft velocity in the central-body frame.
    v_p : ndarray, shape (3,)
        Flyby-body velocity in the central-body frame.

    Returns
    -------
    float
        Flyby periapsis radius.
    """
    v_in_inf = v_in - v_p
    v_inf_norm = norm(v_in_inf)

    r_p = mu_p / v_inf_norm / v_inf_norm * (sqrt(2 / (1 - cos(delta))) - 1)

    return r_p


@jit(cache=True)
def ga_v2rp(v_in, v_out, v_p, mu_p: float):
    """Compute periapsis radius from the incoming and outgoing velocities.

    This uses the relationship between the excess-velocity vectors and the
    deflection angle.

    Parameters
    ----------
    v_in : ndarray, shape (3,)
        Incoming spacecraft velocity in the central-body frame.
    v_out : ndarray, shape (3,)
        Outgoing spacecraft velocity in the central-body frame.
    v_p : ndarray, shape (3,)
        Flyby-body velocity in the central-body frame.
    mu_p : float
        Standard gravitational parameter of the flyby body.

    Returns
    -------
    float
        Flyby periapsis radius.

    Raises
    -------
    AssertionError
        If the squared incoming and outgoing excess speeds differ by more
        than 0.1 percent.
    """
    # Compute the incoming and outgoing hyperbolic excess velocities.
    v_in_inf = v_in - v_p
    v_out_inf = v_out - v_p

    # Use dot products to compute the squared magnitudes.
    v_inf_in_square = v_in_inf @ v_in_inf
    v_inf_out_square = v_out_inf @ v_out_inf

    # An unpowered flyby conserves the hyperbolic excess speed.
    assert abs(v_inf_in_square - v_inf_out_square) < 1e-3 * v_inf_in_square, \
        "v_inf magnitude mismatch: energy not conserved in GA"

    v_inf_square = v_inf_in_square

    # Recover the turn angle from the velocity-vector dot product.
    v_inf_dot = v_in_inf @ v_out_inf

    # cos(delta) = v_inf_dot / v_inf_square
    # r_p = mu_p / v_inf² * (sqrt(2 / (1 - cos(delta))) - 1)
    cos_delta = v_inf_dot / v_inf_square
    r_p = mu_p / v_inf_square * (sqrt(2.0 / (1.0 - cos_delta)) - 1.0)

    return r_p


@jit(cache=True)
def gravity_assists_normalized(v_in, v_p, mu_p: float, r_p_min: float,
                                 delta_norm: float, psi_norm: float):
    """Compute a gravity assist from normalized optimization parameters.

    The parameter space is normalized to ``[0, 1] ** 2`` to reduce the
    nonlinearity seen by the optimizer. ``delta_norm`` maps to
    ``[0, delta_max]`` and ``psi_norm`` maps to ``[0, 2 pi]``.

    Parameters
    ----------
    v_in : ndarray, shape (3,)
        Incoming spacecraft velocity in the central-body frame.
    v_p : ndarray, shape (3,)
        Flyby-body velocity in the central-body frame.
    mu_p : float
        Standard gravitational parameter of the flyby body.
    r_p_min : float
        Minimum allowed flyby periapsis radius.
    delta_norm : float
        Normalized deflection parameter in ``[0, 1]``.
    psi_norm : float
        Normalized azimuth parameter in ``[0, 1]``.

    Returns
    -------
    ndarray, shape (3,)
        Outgoing spacecraft velocity in the central-body frame.
    """
    # Incoming hyperbolic excess velocity.
    v_in_inf = v_in - v_p
    v_inf_norm = norm(v_in_inf)

    # Maximum deflection occurs at the minimum allowed periapsis radius.
    delta_max = 2 * arcsin(
        mu_p /
        (mu_p + r_p_min * v_inf_norm ** 2)
    )

    # Map the normalized parameters to physical angles.
    delta = delta_norm * delta_max  # [0,1] -> [0, delta_max]
    psi = psi_norm * 2 * pi         # [0,1] -> [0, 2π]

    # Construct a right-handed encounter frame; i follows incoming v-infinity.
    i_axis = v_in_inf / v_inf_norm

    # k is normal to the plane containing incoming v-infinity and body velocity.
    k_axis = cross(i_axis, v_p)
    k_norm = norm(k_axis)

    # For nearly collinear vectors, choose an arbitrary transverse direction.
    if k_norm < 1e-10:
        if abs(i_axis[0]) < 0.9:
            temp = [1.0, 0.0, 0.0]
        else:
            temp = [0.0, 1.0, 0.0]
        k_axis = cross(i_axis, temp)
        k_axis /= norm(k_axis)
    else:
        k_axis /= k_norm

    # j completes the right-handed frame.
    j_axis = cross(k_axis, i_axis)

    # Rotate outgoing v-infinity by delta with azimuth psi in the encounter frame.
    v_out_inf = v_inf_norm * (
        i_axis * cos(delta) +
        j_axis * sin(delta) * sin(psi) +
        k_axis * sin(delta) * cos(psi)
    )

    # Transform back to the central-body frame.
    v_out = v_out_inf + v_p

    return v_out
