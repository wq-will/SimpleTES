# EVOLVE-BLOCK-START
import numpy as np

def construct_h():
    # Random construction matching ttt_discover's create_initial_state logic
    rng = np.random.default_rng()
    n_points = rng.integers(40, 100)

    # Start with uniform 0.5
    h_values = np.ones(n_points) * 0.5

    # Add random perturbation with zero mean
    perturbation = rng.uniform(-0.4, 0.4, n_points)
    perturbation = perturbation - np.mean(perturbation)
    h_values = h_values + perturbation

    return h_values, n_points

# EVOLVE-BLOCK-END


def run_code():
    """Run the Erdős minimum overlap optimization.
    
    Returns:
        tuple: (h_values, c5_bound, n_points)
            h_values: np.ndarray, shape (n_points,), discretized step function h
            c5_bound: float, max overlap computed from this h_values
            n_points: int, number of bins used to discretize [0, 2]
    """
    h_values, n_points = construct_h()

    n = int(n_points)
    target_sum = n / 2.0

    # Keep post-processing fixed and robust:
    # - cast to float64 (avoid float32 bound spillover)
    # - project to the feasible set {0<=h<=1, sum(h)=n/2}
    h_values = np.asarray(h_values, dtype=np.float64).reshape(-1)
    if h_values.shape[0] != n:
        raise ValueError(f"Expected h_values shape ({n},), got {h_values.shape}")

    def _project_box_sum(v: np.ndarray, s: float, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
        if not np.all(np.isfinite(v)):
            raise ValueError("h_values contain NaN or inf values")
        # Bisection on tau for x = clip(v - tau, lo, hi) such that sum(x)=s.
        tau_lo = float(np.min(v) - hi)
        tau_hi = float(np.max(v) - lo)
        for _ in range(80):
            tau = (tau_lo + tau_hi) / 2.0
            x = np.clip(v - tau, lo, hi)
            if float(np.sum(x, dtype=np.float64)) > s:
                tau_lo = tau
            else:
                tau_hi = tau
        return np.clip(v - tau_hi, lo, hi)

    h_values = _project_box_sum(h_values, target_sum)
    
    dx = 2.0 / n_points
    j_values = 1.0 - h_values
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    c5_bound = np.max(correlation)
    
    return h_values, c5_bound, n_points
