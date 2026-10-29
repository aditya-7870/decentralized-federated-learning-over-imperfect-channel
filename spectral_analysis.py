"""
spectral_analysis.py — Spectral analysis utilities for M3.

PDF specification (Section 7 - M3):
    "For each candidate topology (ring, grid, random graph), a row-stochastic
     mixing matrix W is constructed and its eigenvalues computed, yielding the
     spectral gap 1 - |lambda_2(W)|. This value feeds a theoretical convergence
     bound, which is compared against the convergence rate and consensus error
     observed empirically from a gossip simulation run on the same topology."

Key definitions:
    W       = Metropolis-Hastings mixing matrix (row-stochastic)
    lambda_1 = 1.0  (largest eigenvalue, always 1 for row-stochastic W)
    lambda_2 = second-largest eigenvalue by absolute value = SLEM
    SLEM    = Second Largest Eigenvalue Modulus = |lambda_2(W)|
    Spectral gap = 1 - SLEM = 1 - |lambda_2(W)|

Convergence bound:
    ||x(t) - x*||^2 <= C * SLEM^t
    => Error decays geometrically at rate SLEM per round.

Mixing time:
    T_mix(eps) = log(1/eps) / log(1/SLEM) = log(1/eps) / log(1/(1-gamma))
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional

from network import create_topology, get_mixing_matrix


# ── Core spectral computations ────────────────────────────────────────────────

def compute_eigenvalues(W: np.ndarray) -> np.ndarray:
    """
    Compute all eigenvalues of W, sorted by descending absolute value.

    For a symmetric doubly-stochastic W all eigenvalues are real.

    Returns
    -------
    np.ndarray  shape (n,)
        |lambda_1| >= |lambda_2| >= ... >= |lambda_n|
    """
    evals = np.linalg.eigvalsh(W)          # real, sorted ascending
    return np.sort(np.abs(evals))[::-1]    # sort descending by |.|


def compute_slem(W: np.ndarray) -> float:
    """
    SLEM = Second Largest Eigenvalue Modulus = |lambda_2(W)|.

    This is the key quantity governing gossip convergence speed.
    """
    evals = compute_eigenvalues(W)
    return float(evals[1]) if len(evals) > 1 else 0.0


def compute_spectral_gap(W: np.ndarray) -> float:
    """Spectral gap = 1 - SLEM."""
    return 1.0 - compute_slem(W)


def compute_mixing_time(spectral_gap: float, epsilon: float = 1e-3) -> float:
    """
    Theoretical mixing time: rounds to reach epsilon-consensus.

    T_mix(eps) = log(1/eps) / log(1/SLEM) = log(1/eps) / log(1/(1-gamma))

    Returns inf if spectral_gap <= 0.
    """
    if spectral_gap <= 1e-10:
        return float("inf")
    slem = 1.0 - spectral_gap
    if slem >= 1.0 - 1e-10:
        return float("inf")
    return np.log(1.0 / epsilon) / np.log(1.0 / slem)


def compute_theoretical_convergence(
    spectral_gap: float,
    num_rounds: int,
    initial_error: float = 1.0,
) -> np.ndarray:
    """
    Theoretical consensus error at each round:  E_t = E_0 * SLEM^t

    Returns np.ndarray of length (num_rounds + 1).
    """
    slem = max(0.0, 1.0 - spectral_gap)
    t    = np.arange(num_rounds + 1)
    return initial_error * (slem ** t)


# ── Topology profiling ────────────────────────────────────────────────────────

def profile_topology(
    name: str,
    n: int,
    seed: int = 42,
    p: float   = 0.5,
) -> dict:
    """
    Build topology, compute W and all spectral quantities.

    Returns
    -------
    dict with keys:
        topology, n_nodes, n_edges, avg_degree,
        spectral_gap, slem, lambda_min,
        all_eigenvalues, mixing_time_1e2, mixing_time_1e3, mixing_time_1e6
    """
    graph    = create_topology(name, n, p=p, seed=seed)
    W        = get_mixing_matrix(graph)
    evals    = compute_eigenvalues(W)
    slem     = float(evals[1]) if len(evals) > 1 else 0.0
    gap      = 1.0 - slem
    degrees  = [d for _, d in graph.degree()]

    return {
        "topology"        : name,
        "n_nodes"         : n,
        "n_edges"         : graph.number_of_edges(),
        "avg_degree"      : float(np.mean(degrees)),
        "spectral_gap"    : gap,
        "slem"            : slem,
        "lambda_min"      : float(evals[-1]),
        "all_eigenvalues" : evals.tolist(),
        "mixing_time_1e2" : compute_mixing_time(gap, 1e-2),
        "mixing_time_1e3" : compute_mixing_time(gap, 1e-3),
        "mixing_time_1e6" : compute_mixing_time(gap, 1e-6),
    }


def sweep_sizes(
    topology_names: List[str],
    sizes: List[int],
    seed: int = 42,
    p: float   = 0.5,
) -> List[dict]:
    """Profile each topology at each node count."""
    results = []
    for name in topology_names:
        for n in sizes:
            try:
                results.append(profile_topology(name, n, seed=seed, p=p))
            except Exception as e:
                print(f"  Warning: {name} n={n} failed: {e}")
    return results


# ── Empirical gossip simulation ───────────────────────────────────────────────

def simulate_gossip_consensus(
    W: np.ndarray,
    num_rounds: int = 60,
    seed: int       = 42,
) -> np.ndarray:
    """
    Simulate gossip consensus for random scalar initial values.

    Each node starts with a random value; per round all values are
    multiplied by W.  Returns consensus error (variance) at each step.

    Returns
    -------
    np.ndarray of length (num_rounds + 1)
    """
    n   = W.shape[0]
    rng = np.random.default_rng(seed)
    x   = rng.standard_normal(n)

    errors = []
    for _ in range(num_rounds + 1):
        errors.append(float(np.var(x)))
        x = W @ x
    return np.array(errors)
