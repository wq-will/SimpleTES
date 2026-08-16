# SPDX-License-Identifier: MIT
# Derived from poliastro 0.17.0.
# Modified by Wizard Intelligence Learning Lab (WILL), 2026.
# Additional modifications by Laboratory of Aerospace Dynamics (LAD),
# Tsinghua University, 2026.
"""Lambert problem solvers.

``lambert`` solves one revolution count and path branch,
``lambert_solutions`` yields feasible solutions in increasing revolution-count
order, and ``lambert_all_solutions`` returns those solutions as a list.
"""

from numba import njit as jit
import numpy as np
from numpy import cross, pi
from math import gamma
from typing import Generator, Literal

def lambert(
    k: float,
    r0: np.ndarray,
    r: np.ndarray,
    tof: float,
    M: int = 0,
    prograde: bool = True,
    lowpath: bool = True,
    numiter: int = 35,
    rtol: float = 1e-8,
    method: Literal["vallado", "izzo"] = "izzo"
):
    """Solves the Lambert problem.

    Parameters
    ----------
    k : float
        Gravitational constant of main attractor.
    r0 : np.ndarray
        Initial position.
    r : np.ndarray
        Final position.
    tof : float
        Time of flight.
    M : int, optional
        Number of full revolutions, default to 0.
    prograde: boolean
        Controls the desired inclination of the transfer orbit.
        True for inc < 90 degrees, False for inc > 90 degrees.
    lowpath: boolean
        If `True` or `False`, gets the transfer orbit whose vacant focus is
        below or above the chord line, respectively.
    numiter : int, optional
        Maximum number of iterations, default to 35.
    rtol : float, optional
        Relative tolerance of the algorithm, default to 1e-8.
    method : {'vallado', 'izzo'}, optional
        Method to use for solving the Lambert problem. Defaults to 'izzo'.

    Returns
    -------
    v0, v : tuple
        Pair of velocity solutions.

    """
    if method == "izzo":
        return izzo(k, r0, r, tof, M, prograde, lowpath, numiter, rtol)
    elif method == "vallado":
        return vallado(k, r0, r, tof, M, prograde, lowpath, numiter, rtol)
    else:
        raise ValueError(f"Method '{method}' is not supported.")

@jit
def vallado(k, r0, r, tof, M, prograde, lowpath, numiter, rtol):
    r"""Solves the Lambert's problem.

    The algorithm returns the initial velocity vector and the final one, these are
    computed by the following expresions:

    .. math::

        \vec{v_{o}} &= \frac{1}{g}(\vec{r} - f\vec{r_{0}}) \\
        \vec{v} &= \frac{1}{g}(\dot{g}\vec{r} - \vec{r_{0}})

    Therefore, the lagrange coefficients need to be computed. For the case of
    Lamber's problem, they can be expressed by terms of the initial and final vector:

    .. math::

        \begin{align}
            f = 1 -\frac{y}{r_{o}} \\
            g = A\sqrt{\frac{y}{\mu}} \\
            \dot{g} = 1 - \frac{y}{r} \\
        \end{align}

    Where y(z) is a function that depends on the :py:mod:`poliastro.core.stumpff` coefficients:

    .. math::

        y = r_{o} + r + A\frac{zS(z)-1}{\sqrt{C(z)}} \\
        A = \sin{(\Delta \nu)}\sqrt{\frac{rr_{o}}{1 - \cos{(\Delta \nu)}}}

    The value of z to evaluate the stump functions is solved by applying a Numerical method to
    the following equation:

    .. math::

        z_{i+1} = z_{i} - \frac{F(z_{i})}{F{}'(z_{i})}

    Function F(z)  to the expression:

    .. math::

        F(z) = \left [\frac{y(z)}{C(z)}  \right ]^{\frac{3}{2}}S(z) + A\sqrt{y(z)} - \sqrt{\mu}\Delta t

    Parameters
    ----------
    k : float
        Gravitational Parameter
    r0 : numpy.ndarray
        Initial position vector
    r : numpy.ndarray
        Final position vector
    tof : float
        Time of flight
    M : int
        Number of revolutions
    prograde: boolean
        Controls the desired inclination of the transfer orbit.
    lowpath: boolean
        If `True` or `False`, gets the transfer orbit whose vacant focus is
        below or above the chord line, respectively.
    numiter : int
        Number of iterations to
    rtol : int
        Number of revolutions

    Returns
    -------
    v1: numpy.ndarray
        Initial velocity vector
    v2: numpy.ndarray
        Final velocity vector

    Examples
    --------
    >>> from poliastro.core.iod import vallado
    >>> from astropy import units as u
    >>> import numpy as np
    >>> from poliastro.bodies import Earth
    >>> k = Earth.k.to(u.km ** 3 / u.s ** 2)
    >>> r1 = np.array([5000, 10000, 2100]) * u.km # Initial position vector
    >>> r2 = np.array([-14600, 2500, 7000]) * u.km # Final position vector
    >>> tof = 3600 * u.s # Time of flight
    >>> v1, v2 = vallado(k.value, r1.value, r2.value, tof.value, M=0, prograde=True, lowpath=True, numiter=35, rtol=1e-8)
    >>> v1 = v1 * u.km / u.s
    >>> v2 = v2 * u.km / u.s
    >>> print(v1, v2)
    [-5.99249503  1.92536671  3.24563805] km / s [-3.31245851 -4.19661901 -0.38528906] km / s

    Notes
    -----
    This procedure can be found in section 5.3 of Curtis, with all the
    theoretical description of the problem. Analytical example can be found
    in the same book under name Example 5.2.

    """
    # This Vallado implementation supports only zero-revolution transfers.
    # Issue: https://github.com/poliastro/poliastro/issues/858
    if M > 0:
        raise NotImplementedError(
            "Multi-revolution scenario not supported for Vallado. See issue https://github.com/poliastro/poliastro/issues/858"
        )

    t_m = 1 if prograde else -1

    norm_r0 = norm(r0)
    norm_r = norm(r)
    norm_r0_times_norm_r = norm_r0 * norm_r
    norm_r0_plus_norm_r = norm_r0 + norm_r

    cos_dnu = (r0 @ r) / norm_r0_times_norm_r

    A = t_m * (norm_r * norm_r0 * (1 + cos_dnu)) ** 0.5

    if A == 0.0:
        raise RuntimeError("Cannot compute orbit, phase angle is 180 degrees")

    psi = 0.0
    psi_low = -4 * np.pi**2
    psi_up = 4 * np.pi**2

    count = 0

    while count < numiter:
        y = norm_r0_plus_norm_r + A * (psi * c3(psi) - 1) / c2(psi) ** 0.5
        if A > 0.0:
            # Readjust xi_low until y > 0.0
            # Translated directly from Vallado
            while y < 0.0:
                psi_low = psi
                psi = (
                    0.8
                    * (1.0 / c3(psi))
                    * (1.0 - norm_r0_times_norm_r * np.sqrt(c2(psi)) / A)
                )
                y = (
                    norm_r0_plus_norm_r
                    + A * (psi * c3(psi) - 1) / c2(psi) ** 0.5
                )

        xi = np.sqrt(y / c2(psi))
        tof_new = (xi**3 * c3(psi) + A * np.sqrt(y)) / np.sqrt(k)

        # Convergence check
        if np.abs((tof_new - tof) / tof) < rtol:
            break
        count += 1
        # Bisection check
        condition = tof_new <= tof
        psi_low = psi_low + (psi - psi_low) * condition
        psi_up = psi_up + (psi - psi_up) * (not condition)

        psi = (psi_up + psi_low) / 2
    else:
        raise RuntimeError("Maximum number of iterations reached")

    f = 1 - y / norm_r0
    g = A * np.sqrt(y / k)

    gdot = 1 - y / norm_r

    v0 = (r - f * r0) / g
    v = (gdot * r - r0) / g

    return v0, v


@jit
def izzo(k, r1, r2, tof, M, prograde, lowpath, numiter, rtol):
    """Aplies izzo algorithm to solve Lambert's problem.

    Parameters
    ----------
    k : float
        Gravitational Constant
    r1 : numpy.ndarray
        Initial position vector
    r2 : numpy.ndarray
        Final position vector
    tof : float
        Time of flight between both positions
    M : int
        Number of revolutions
    prograde: boolean
        Controls the desired inclination of the transfer orbit.
    lowpath: boolean
        If `True` or `False`, gets the transfer orbit whose vacant focus is
        below or above the chord line, respectively.
    numiter : int
        Number of iterations
    rtol : float
        Error tolerance

    Returns
    -------
    v1: numpy.ndarray
        Initial velocity vector
    v2: numpy.ndarray
        Final velocity vector

    """
    # Check preconditions
    assert tof > 0
    assert k > 0

    # Check collinearity of r1 and r2
    if not cross(r1, r2).any():
        raise ValueError(
            "Lambert solution cannot be computed for collinear vectors"
        )

    # Chord
    c = r2 - r1
    c_norm, r1_norm, r2_norm = norm(c), norm(r1), norm(r2)

    # Semiperimeter
    s = (r1_norm + r2_norm + c_norm) * 0.5

    # Versors
    i_r1, i_r2 = r1 / r1_norm, r2 / r2_norm
    i_h = cross(i_r1, i_r2)
    i_h = i_h / norm(i_h)  # Fixed from paper

    # Geometry of the problem
    ll = np.sqrt(1 - min(1.0, c_norm / s))

    # Compute the fundamental tangential directions
    if i_h[2] < 0:
        ll = -ll
        i_t1, i_t2 = cross(i_r1, i_h), cross(i_r2, i_h)
    else:
        i_t1, i_t2 = cross(i_h, i_r1), cross(i_h, i_r2)

    # Correct transfer angle parameter and tangential vectors if required
    ll, i_t1, i_t2 = (ll, i_t1, i_t2) if prograde else (-ll, -i_t1, -i_t2)

    # Nondimensional time of flight
    T = np.sqrt(2 * k / s**3) * tof

    # Find solutions
    x, y = _find_xy(ll, T, M, numiter, lowpath, rtol)

    # Reconstruct
    gamma = np.sqrt(k * s / 2)
    rho = (r1_norm - r2_norm) / c_norm
    sigma = np.sqrt(1 - rho**2)

    # Compute the radial and tangential components at r0 and r
    V_r1, V_r2, V_t1, V_t2 = _reconstruct(
        x, y, r1_norm, r2_norm, ll, gamma, rho, sigma
    )

    # Solve for the initial and final velocity
    v1 = V_r1 * (r1 / r1_norm) + V_t1 * i_t1
    v2 = V_r2 * (r2 / r2_norm) + V_t2 * i_t2

    return v1, v2


@jit
def _reconstruct(x, y, r1, r2, ll, gamma, rho, sigma):
    """Reconstruct solution velocity vectors."""
    V_r1 = gamma * ((ll * y - x) - rho * (ll * y + x)) / r1
    V_r2 = -gamma * ((ll * y - x) + rho * (ll * y + x)) / r2
    V_t1 = gamma * sigma * (y + ll * x) / r1
    V_t2 = gamma * sigma * (y + ll * x) / r2
    return V_r1, V_r2, V_t1, V_t2


@jit
def _find_xy(ll, T, M, numiter, lowpath, rtol):
    """Computes all x, y for given number of revolutions."""
    # For abs(ll) == 1 the derivative is not continuous
    assert abs(ll) < 1
    assert T > 0  # Mistake in the original paper

    M_max = np.floor(T / pi)
    T_00 = np.arccos(ll) + ll * np.sqrt(1 - ll**2)  # T_xM

    # Refine maximum number of revolutions if necessary
    if T < T_00 + M_max * pi and M_max > 0:
        _, T_min = _compute_T_min(ll, M_max, numiter, rtol)
        if T < T_min:
            M_max -= 1

    # Check whether a feasible solution exists for the requested revolutions.
    # This departs from the original paper in that we do not compute all solutions
    if M > M_max:
        raise ValueError("No feasible solution, try lower M")

    # Initial guess
    x_0 = _initial_guess(T, ll, M, lowpath)

    # Start Householder iterations from x_0 and find x, y
    x = _householder(x_0, T, ll, M, rtol, numiter)
    y = _compute_y(x, ll)

    return x, y


@jit
def _compute_y(x, ll):
    """Computes y."""
    return np.sqrt(1 - ll**2 * (1 - x**2))


@jit
def _compute_psi(x, y, ll):
    """Computes psi.

    "The auxiliary angle psi is computed using Eq.(17) by the appropriate
    inverse function"

    """
    if -1 <= x < 1:
        # Elliptic motion
        # Use arc cosine to avoid numerical errors
        return np.arccos(x * y + ll * (1 - x**2))
    elif x > 1:
        # Hyperbolic motion
        # The hyperbolic sine is bijective
        return np.arcsinh((y - x * ll) * np.sqrt(x**2 - 1))
    else:
        # Parabolic motion
        return 0.0


@jit
def _tof_equation(x, T0, ll, M):
    """Time of flight equation."""
    return _tof_equation_y(x, _compute_y(x, ll), T0, ll, M)


@jit
def _tof_equation_y(x, y, T0, ll, M):
    """Time of flight equation with externally computated y."""
    if M == 0 and np.sqrt(0.6) < x < np.sqrt(1.4):
        eta = y - ll * x
        S_1 = (1 - ll - x * eta) * 0.5
        Q = 4 / 3 * hyp2f1b(S_1)
        T_ = (eta**3 * Q + 4 * ll * eta) * 0.5
    else:
        psi = _compute_psi(x, y, ll)
        T_ = np.divide(
            np.divide(psi + M * pi, np.sqrt(np.abs(1 - x**2))) - x + ll * y,
            (1 - x**2),
        )

    return T_ - T0


@jit
def _tof_equation_p(x, y, T, ll):
    # This expression is singular as x approaches 1.
    return (3 * T * x - 2 + 2 * ll**3 * x / y) / (1 - x**2)


@jit
def _tof_equation_p2(x, y, T, dT, ll):
    return (3 * T + 5 * x * dT + 2 * (1 - ll**2) * ll**3 / y**3) / (
        1 - x**2
    )


@jit
def _tof_equation_p3(x, y, _, dT, ddT, ll):
    return (
        7 * x * ddT + 8 * dT - 6 * (1 - ll**2) * ll**5 * x / y**5
    ) / (1 - x**2)


@jit
def _compute_T_min(ll, M, numiter, rtol):
    """Compute minimum T."""
    if ll == 1:
        x_T_min = 0.0
        T_min = _tof_equation(x_T_min, 0.0, ll, M)
    else:
        if M == 0:
            x_T_min = np.inf
            T_min = 0.0
        else:
            # Set x_i > 0 to avoid problems at ll = -1
            x_i = 0.1
            T_i = _tof_equation(x_i, 0.0, ll, M)
            x_T_min = _halley(x_i, T_i, ll, rtol, numiter)
            T_min = _tof_equation(x_T_min, 0.0, ll, M)

    return x_T_min, T_min


@jit
def _initial_guess(T, ll, M, lowpath):
    """Initial guess."""
    if M == 0:
        # Single revolution
        T_0 = np.arccos(ll) + ll * np.sqrt(1 - ll**2) + M * pi  # Equation 19
        T_1 = 2 * (1 - ll**3) / 3  # Equation 21
        if T >= T_0:
            x_0 = (T_0 / T) ** (2 / 3) - 1
        elif T < T_1:
            x_0 = 5 / 2 * T_1 / T * (T_1 - T) / (1 - ll**5) + 1
        else:
            # Corrected initial guess for T_1 <= T < T_0; the piecewise
            # expression after equation (30) in the original paper is incorrect.
            # See https://github.com/poliastro/poliastro/issues/1362
            x_0 = np.exp(np.log(2) * np.log(T / T_0) / np.log(T_1 / T_0)) - 1

        return x_0
    else:
        # Multiple revolutions
        x_0l = (((M * pi + pi) / (8 * T)) ** (2 / 3) - 1) / (
            ((M * pi + pi) / (8 * T)) ** (2 / 3) + 1
        )
        x_0r = (((8 * T) / (M * pi)) ** (2 / 3) - 1) / (
            ((8 * T) / (M * pi)) ** (2 / 3) + 1
        )

        # Select one of the solutions according to desired type of path
        x_0 = (
            np.max(np.array([x_0l, x_0r]))
            if lowpath
            else np.min(np.array([x_0l, x_0r]))
        )

        return x_0


@jit
def _halley(p0, T0, ll, tol, maxiter):
    """Find a minimum of time of flight equation using the Halley method.

    Notes
    -----
    This function is private because it assumes a calling convention specific to
    this module and is not really reusable.

    """
    for ii in range(maxiter):
        y = _compute_y(p0, ll)
        fder = _tof_equation_p(p0, y, T0, ll)
        fder2 = _tof_equation_p2(p0, y, T0, fder, ll)
        if fder2 == 0:
            raise RuntimeError("Derivative was zero")
        fder3 = _tof_equation_p3(p0, y, T0, fder, fder2, ll)

        # Halley step (cubic)
        p = p0 - 2 * fder * fder2 / (2 * fder2**2 - fder * fder3)

        if abs(p - p0) < tol:
            return p
        p0 = p

    raise RuntimeError("Failed to converge")


@jit
def _householder(p0, T0, ll, M, tol, maxiter):
    """Find a zero of time of flight equation using the Householder method.

    Notes
    -----
    This function is private because it assumes a calling convention specific to
    this module and is not really reusable.

    """
    for ii in range(maxiter):
        y = _compute_y(p0, ll)
        fval = _tof_equation_y(p0, y, T0, ll, M)
        T = fval + T0
        fder = _tof_equation_p(p0, y, T, ll)
        fder2 = _tof_equation_p2(p0, y, T, fder, ll)
        fder3 = _tof_equation_p3(p0, y, T, fder, fder2, ll)

        # Householder step (quartic)
        p = p0 - fval * (
            (fder**2 - fval * fder2 / 2)
            / (fder * (fder**2 - fval * fder2) + fder3 * fval**2 / 6)
        )

        if abs(p - p0) < tol:
            return p
        p0 = p

    raise RuntimeError("Failed to converge")

@jit
def norm(arr):
    return np.sqrt(arr @ arr)

@jit
def hyp2f1b(x):
    """Evaluate the hypergeometric function 2F1(3, 1, 5/2, x).

    Notes
    -----
    The series is evaluated iteratively following the formulation used in
    Battin's Lambert algorithm. Background on the function is available at
    https://en.wikipedia.org/wiki/Hypergeometric_function

    """
    if x >= 1.0:
        return np.inf
    else:
        res = 1.0
        term = 1.0
        ii = 0
        while True:
            term = term * (3 + ii) * (1 + ii) / (5 / 2 + ii) * x / (ii + 1)
            res_old = res
            res += term
            if res_old == res:
                return res
            ii += 1

@jit
def stumpff_c2(psi):
    r"""Second Stumpff function.

    For positive arguments:

    .. math::

        c_2(\psi) = \frac{1 - \cos{\sqrt{\psi}}}{\psi}

    """
    eps = 1.0
    if psi > eps:
        res = (1 - np.cos(np.sqrt(psi))) / psi
    elif psi < -eps:
        res = (np.cosh(np.sqrt(-psi)) - 1) / (-psi)
    else:
        res = 1.0 / 2.0
        delta = (-psi) / gamma(2 + 2 + 1)
        k = 1
        while res + delta != res:
            res = res + delta
            k += 1
            delta = (-psi) ** k / gamma(2 * k + 2 + 1)

    return res

@jit
def c2(psi):
    """Alias for stumpff_c2."""
    return stumpff_c2(psi)

@jit
def stumpff_c3(psi):
    r"""Third Stumpff function.

    For positive arguments:

    .. math::

        c_3(\psi) = \frac{\sqrt{\psi} - \sin{\sqrt{\psi}}}{\sqrt{\psi^3}}

    """
    eps = 1.0
    if psi > eps:
        res = (np.sqrt(psi) - np.sin(np.sqrt(psi))) / (psi * np.sqrt(psi))
    elif psi < -eps:
        res = (np.sinh(np.sqrt(-psi)) - np.sqrt(-psi)) / (-psi * np.sqrt(-psi))
    else:
        res = 1.0 / 6.0
        delta = (-psi) / gamma(2 + 3 + 1)
        k = 1
        while res + delta != res:
            res = res + delta
            k += 1
            delta = (-psi) ** k / gamma(2 * k + 3 + 1)

    return res

@jit
def c3(psi):
    """Alias for stumpff_c3."""
    return stumpff_c3(psi)

def _test():
    """Run tests for two lambert solvers."""
    import time

    # Earth's gravitational parameter (km^3/s^2)
    k = 398600.4418

    # Near-Earth satellite example (LEO orbit)
    # Initial position: ISS-like orbit at ~400 km altitude
    r0 = np.array([6778.0, 0.0, 0.0])  # km
    # Final position: after 1/4 orbit
    r = np.array([0.0, 6778.0, 0.0])   # km

    # Time of flight: approximately 1/4 of orbital period
    # For circular orbit at 400 km: T = 2π√(a³/μ) ≈ 5536 seconds
    tof = 1384.0  # seconds (1/4 of orbit period)

    # Test parameters
    M = 0          # Number of revolutions
    prograde = True
    lowpath = True
    numiter = 35
    rtol = 1e-8

    print("=" * 60)
    print("Lambert Problem Test - Near-Earth Satellite Example")
    print("=" * 60)
    print(f"Initial position: {r0} km")
    print(f"Final position: {r} km")
    print(f"Time of flight: {tof} seconds")
    print(f"Gravitational parameter: {k} km³/s²")
    print()

    # Test Vallado method
    print("Testing Vallado method...")
    try:
        start_time = time.time()
        v0_vallado, v_vallado = lambert(k, r0, r, tof, M, prograde, lowpath, numiter, rtol, method="vallado")
        vallado_time = time.time() - start_time

        print(f"✓ Vallado method completed in {vallado_time:.6f} seconds")
        print(f"  Initial velocity: {v0_vallado} km/s")
        print(f"  Final velocity: {v_vallado} km/s")
        print(f"  Initial speed: {norm(v0_vallado):.4f} km/s")
        print(f"  Final speed: {norm(v_vallado):.4f} km/s")

        vallado_success = True
    except Exception as e:
        print(f"✗ Vallado method failed: {e}")
        vallado_success = False

    print()

    # Test Izzo method
    print("Testing Izzo method...")
    try:
        start_time = time.time()
        v0_izzo, v_izzo = lambert(k, r0, r, tof, M, prograde, lowpath, numiter, rtol, method="izzo")
        izzo_time = time.time() - start_time

        print(f"✓ Izzo method completed in {izzo_time:.6f} seconds")
        print(f"  Initial velocity: {v0_izzo} km/s")
        print(f"  Final velocity: {v_izzo} km/s")
        print(f"  Initial speed: {norm(v0_izzo):.4f} km/s")
        print(f"  Final speed: {norm(v_izzo):.4f} km/s")

        izzo_success = True
    except Exception as e:
        print(f"✗ Izzo method failed: {e}")
        izzo_success = False

    print()

    # Compare results if both methods succeeded
    if vallado_success and izzo_success:
        print("Comparing results...")
        v0_diff = norm(v0_vallado - v0_izzo)
        v_diff = norm(v_vallado - v_izzo)

        print(f"Initial velocity difference: {v0_diff:.8f} km/s")
        print(f"Final velocity difference: {v_diff:.8f} km/s")

        if v0_diff < 1e-6 and v_diff < 1e-6:
            print("✓ Results match within tolerance")
        else:
            print("⚠ Results differ significantly")

    print()
    print("=" * 60)
    print("Performance Test - 1,000,000 iterations")
    print("=" * 60)

    # Performance test with 1M iterations
    iterations = 1000000

    if vallado_success:
        print(f"Running Vallado method {iterations:,} times...")
        start_time = time.time()
        for _ in range(iterations):
            lambert(k, r0, r, tof, M, prograde, lowpath, numiter, rtol, method="vallado")
        vallado_total_time = time.time() - start_time

        print(f"Vallado method:")
        print(f"  Total time: {vallado_total_time:.3f} seconds")
        print(f"  Average time per call: {vallado_total_time/iterations*1e6:.3f} microseconds")
        print(f"  Calls per second: {iterations/vallado_total_time:,.0f}")

    print()

    if izzo_success:
        print(f"Running Izzo method {iterations:,} times...")
        start_time = time.time()
        for _ in range(iterations):
            lambert(k, r0, r, tof, M, prograde, lowpath, numiter, rtol, method="izzo")
        izzo_total_time = time.time() - start_time

        print(f"Izzo method:")
        print(f"  Total time: {izzo_total_time:.3f} seconds")
        print(f"  Average time per call: {izzo_total_time/iterations*1e6:.3f} microseconds")
        print(f"  Calls per second: {iterations/izzo_total_time:,.0f}")

    print()

    # Performance comparison
    if vallado_success and izzo_success:
        speedup = vallado_total_time / izzo_total_time
        if speedup > 1:
            print(f"Izzo method is {speedup:.2f}x faster than Vallado method")
        else:
            print(f"Vallado method is {1/speedup:.2f}x faster than Izzo method")

    print("=" * 60)

def _test_prograde():
    """Run tests for prograde Lambert solver."""
    D2PI = pi / 180.0
    k = 398600.4418  # Earth's gravitational parameter (km^3/s^2)
    sma = 6778.0  # Semi-major axis (km)
    ecc = 0.001  # Eccentricity
    inc = 98.7 * D2PI  # Inclination (rad)
    raan = 0.0  # Right ascension of ascending node (rad)
    argp = 0.0  # Argument of perigee (rad)
    ta0 = 0.5; ta1 = 1.5  # True anomaly at t0 and t1 (rad)
    tof = 1384.0  # Time of flight (seconds)

    from .elements import coe2rv_sma, rv2coe_sma
    r0, v0 = coe2rv_sma(k, sma, ecc, inc, raan, argp, ta0)
    r1, v1 = coe2rv_sma(k, sma, ecc, inc, raan, argp, ta1)

    print("Testing prograde Lambert solver...")
    try:
        v0_prograde, v1_prograde = lambert(
            k, r0, r1, tof, M=0, prograde=True
        )
        sma_p, ecc_p, inc_p, raan_p, argp_p, ta0_p = rv2coe_sma(k, r0, v0_prograde)
        print(f"Prograde state vectors:")
        print(f"  semi-major axis: {sma_p:.4f} km")
        print(f"  eccentricity: {ecc_p:.4f}")
        print(f"  inclination: {inc_p * 180.0 / pi:.4f} degrees")
        print(f"  right ascension of ascending node: {raan_p * 180.0 / pi:.4f} degrees")
        print(f"  argument of perigee: {argp_p * 180.0 / pi:.4f} degrees")
        print(f"  true anomaly at t0: {ta0_p * 180.0 / pi:.4f} degrees")
    except Exception as e:
        print(f"Prograde Lambert solver failed: {e}")

    print("Testing retrograde Lambert solver...")
    try:
        v0_retrograde, v1_retrograde = lambert(
            k, r0, r1, tof, M=0, prograde=False
        )
        sma_r, ecc_r, inc_r, raan_r, argp_r, ta0_r = rv2coe_sma(k, r0, v0_retrograde)
        print(f"Retrograde state vectors:")
        print(f"  semi-major axis: {sma_r:.4f} km")
        print(f"  eccentricity: {ecc_r:.4f}")
        print(f"  inclination: {inc_r * 180.0 / pi:.4f} degrees")
        print(f"  right ascension of ascending node: {raan_r * 180.0 / pi:.4f} degrees")
        print(f"  argument of perigee: {argp_r * 180.0 / pi:.4f} degrees")
        print(f"  true anomaly at t0: {ta0_r * 180.0 / pi:.4f} degrees")
    except Exception as e:
        print(f"Retrograde Lambert solver failed: {e}")



def lambert_solutions(
    k: float,
    r0: np.ndarray,
    r: np.ndarray,
    tof: float,
    prograde: bool = True,
    numiter: int = 35,
    rtol: float = 1e-8,
    method: Literal["vallado", "izzo"] = "izzo",
    max_revs: int = 20,
) -> Generator[tuple[int, bool | None, np.ndarray, np.ndarray], None, None]:
    """Yield all feasible Lambert transfers for the boundary conditions.

    There is one solution for ``M=0``, where ``lowpath`` is not applicable.
    For ``M >= 1``, the solver considers distinct low- and high-path branches.

    Parameters
    ----------
    k : float
        Standard gravitational parameter of the central body, in km³/s².
    r0 : np.ndarray
        Departure position vector, in km.
    r : np.ndarray
        Arrival position vector, in km.
    tof : float
        Time of flight, in seconds. Must be positive.
    prograde : bool, optional
        Select a prograde transfer when true and a retrograde transfer when
        false. The default is true.
    numiter : int, optional
        Maximum iteration count. The default is 35.
    rtol : float, optional
        Relative convergence tolerance. The default is ``1e-8``.
    method : {'vallado', 'izzo'}, optional
        Solution algorithm. The default is ``'izzo'``.
    max_revs : int, optional
        Maximum revolution count to enumerate. The default is 20.

    Yields
    ------
    M : int
        Number of complete revolutions in the transfer.
    lowpath : bool or None
        ``None`` for ``M=0``; for ``M >= 1``, true selects the low path and
        false selects the high path.
    v0 : np.ndarray
        Departure velocity vector, in km/s.
    v : np.ndarray
        Arrival velocity vector, in km/s.

    Examples
    --------
    >>> for M, lowpath, v0, v in lambert_solutions(k, r0, r, tof):
    ...     dv = np.linalg.norm(v0 - v_hub)
    ...     print(f"M={M}, low={lowpath}, Δv={dv:.3f} km/s")

    Notes
    -----
    This generator does not filter solutions by mission-level feasibility such
    as velocity budget or periapsis altitude. Callers are responsible for those
    checks.

    For ``M=0``, a mathematical Lambert solution exists for every positive time
    of flight; very short times correspond to very high-energy transfers.
    """
    # M=0 has a single solution; lowpath is not applicable.
    try:
        v0, v = lambert(k, r0, r, tof,
                        M=0, prograde=prograde, lowpath=True,
                        numiter=numiter, rtol=rtol, method=method)
        yield 0, None, v0, v
    except (ValueError, RuntimeError):
        # No solution exists for this time of flight and boundary condition.
        return

    # For M>=1, test both the low- and high-path branches.
    for M in range(1, max_revs + 1):
        m_feasible = False
        for lowpath in (True, False):
            try:
                v0, v = lambert(k, r0, r, tof,
                                M=M, prograde=prograde, lowpath=lowpath,
                                numiter=numiter, rtol=rtol, method=method)
                yield M, lowpath, v0, v
                m_feasible = True
            except ValueError:
                # Higher revolution counts are also infeasible by monotonicity.
                break
            except RuntimeError:
                # Skip a nonconvergent branch and try the other branch.
                continue
        if not m_feasible:
            break


def lambert_all_solutions(
    k: float,
    r0: np.ndarray,
    r: np.ndarray,
    tof: float,
    prograde: bool = True,
    numiter: int = 35,
    rtol: float = 1e-8,
    method: Literal["vallado", "izzo"] = "izzo",
    max_revs: int = 20,
) -> list[tuple[int, bool | None, np.ndarray, np.ndarray]]:
    """Return all feasible Lambert transfers for the boundary conditions.

    This list wrapper around :func:`lambert_solutions` is convenient for
    sorting solutions by delta-v or accessing them repeatedly.

    Parameters
    ----------
    k : float
        Standard gravitational parameter of the central body, in km³/s².
    r0 : np.ndarray
        Departure position vector, in km.
    r : np.ndarray
        Arrival position vector, in km.
    tof : float
        Time of flight, in seconds. Must be positive.
    prograde : bool, optional
        Select a prograde transfer when true and a retrograde transfer when
        false. The default is true.
    numiter : int, optional
        Maximum iteration count. The default is 35.
    rtol : float, optional
        Relative convergence tolerance. The default is ``1e-8``.
    method : {'vallado', 'izzo'}, optional
        Solution algorithm. The default is ``'izzo'``.
    max_revs : int, optional
        Maximum revolution count to enumerate. The default is 20.

    Returns
    -------
    list of (M, lowpath, v0, v)
        Each tuple contains the revolution count, path selector, departure
        velocity, and arrival velocity. ``lowpath`` is ``None`` for ``M=0``.

    Examples
    --------
    >>> solutions = lambert_all_solutions(k, r0, r, tof)
    >>> best = min(solutions, key=lambda s: np.linalg.norm(s[2]))
    """
    return list(lambert_solutions(k, r0, r, tof,
                                  prograde=prograde,
                                  numiter=numiter, rtol=rtol,
                                  method=method, max_revs=max_revs))


if __name__ == "__main__":
    _test()
