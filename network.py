"""
network.py — Network topology generation and mixing-matrix construction.

Generates ring, grid, and Erdős–Rényi random graphs via NetworkX and builds
the row-stochastic Metropolis–Hastings weight matrix W used for gossip
averaging (and later for M3 spectral-gap analysis).
"""

import networkx as nx
import numpy as np
from typing import List, Dict


# ── Topology builders ────────────────────────────────────────────────────────


def create_ring(n: int) -> nx.Graph:
    """
    Create a ring (cycle) graph with *n* nodes.

    Each node is connected to exactly 2 neighbors (left & right).
    """
    G = nx.cycle_graph(n)
    G.graph["topology"] = "ring"
    return G


def create_grid(n: int) -> nx.Graph:
    """
    Create a 2-D grid graph that accommodates *n* nodes.

    For non-square *n* the grid is shaped as ``rows × cols`` where
    ``rows = round(sqrt(n))`` and ``cols = ceil(n / rows)``.
    Only the first *n* nodes of the full grid are kept.
    """
    import math
    rows = round(math.sqrt(n))
    cols = math.ceil(n / rows)
    G_full = nx.grid_2d_graph(rows, cols)

    # Relabel nodes from (r, c) tuples to simple integers
    mapping = {}
    node_id = 0
    for r in range(rows):
        for c in range(cols):
            if (r, c) in G_full and node_id < n:
                mapping[(r, c)] = node_id
                node_id += 1

    # Keep only the first n nodes
    nodes_to_keep = set(mapping.keys())
    G_full = G_full.subgraph(nodes_to_keep).copy()
    G = nx.relabel_nodes(G_full, mapping)
    G.graph["topology"] = "grid"
    return G


def create_random(n: int, p: float = 0.5, seed: int = 42) -> nx.Graph:
    """
    Create a connected Erdős–Rényi random graph G(n, p).

    If the initial sample is disconnected the function re-samples
    (up to 100 attempts) with incremented seeds.
    """
    for attempt in range(100):
        G = nx.erdos_renyi_graph(n, p, seed=seed + attempt)
        if nx.is_connected(G):
            G.graph["topology"] = "random"
            return G
    # Fallback: return a complete graph (always connected)
    G = nx.complete_graph(n)
    G.graph["topology"] = "random"
    return G


def create_topology(name: str, n: int, **kwargs) -> nx.Graph:
    """
    Factory function — create a topology by name.

    Parameters
    ----------
    name : str
        One of ``"ring"``, ``"grid"``, ``"random"``.
    n : int
        Number of nodes.
    **kwargs
        Extra arguments forwarded to the builder (e.g. ``p`` for random).
    """
    builders = {
        "ring": create_ring,
        "grid": create_grid,
        "random": create_random,
    }
    if name not in builders:
        raise ValueError(f"Unknown topology '{name}'. Choose from {list(builders)}")
    if name == "random":
        return create_random(n, p=kwargs.get("p", 0.5), seed=kwargs.get("seed", 42))
    return builders[name](n)


# ── Neighbor queries ─────────────────────────────────────────────────────────


def get_neighbors(graph: nx.Graph, node_id: int) -> List[int]:
    """Return the list of neighbor node IDs for *node_id*."""
    return list(graph.neighbors(node_id))


def get_adjacency_dict(graph: nx.Graph) -> Dict[int, List[int]]:
    """Return a dict mapping each node to its list of neighbors."""
    return {node: get_neighbors(graph, node) for node in graph.nodes()}


# ── Mixing matrix ────────────────────────────────────────────────────────────


def get_mixing_matrix(graph: nx.Graph) -> np.ndarray:
    """
    Build the row-stochastic Metropolis–Hastings weight matrix **W**.

    For nodes *i* and *j* that are neighbors::

        W_ij = 1 / (1 + max(deg(i), deg(j)))

    The diagonal is set so each row sums to 1::

        W_ii = 1 − Σ_{j ∈ N(i)} W_ij

    This weighting scheme is standard in decentralized optimization and
    guarantees a doubly-stochastic matrix for undirected graphs.
    """
    n = graph.number_of_nodes()
    W = np.zeros((n, n))
    degrees = dict(graph.degree())

    for i in graph.nodes():
        for j in graph.neighbors(i):
            W[i][j] = 1.0 / (1.0 + max(degrees[i], degrees[j]))
        W[i][i] = 1.0 - np.sum(W[i])

    return W


def get_spectral_gap(W: np.ndarray) -> float:
    """
    Compute the spectral gap ``1 − |λ₂(W)|``.

    A larger spectral gap implies faster information mixing / consensus.
    """
    eigenvalues = np.sort(np.abs(np.linalg.eigvals(W)))[::-1]
    # eigenvalues[0] ≈ 1 (row-stochastic), eigenvalues[1] = |λ₂|
    if len(eigenvalues) < 2:
        return 1.0
    return 1.0 - eigenvalues[1]
