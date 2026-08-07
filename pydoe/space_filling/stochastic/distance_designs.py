"""
Maximin and minimax distance designs.

Distance-based designs select points so that they are well spread out
over the design space according to an inter-point distance criterion.

A **maximin distance design** maximizes the minimum pairwise Euclidean
distance between design points, spreading points apart from each other.
This implementation searches over Latin hypercube designs (each column
is a permutation of cell indices) using a coordinate-exchange local
search.

A **minimax distance design** selects points from a candidate set so
that every point in the candidate set (i.e. every point of the design
space) is close to some selected design point, minimizing the maximum
distance from any candidate to its nearest selected point. This is a
k-center / facility-location style criterion, also refined by local
search.

References
----------
Johnson, M. E., Moore, L. M., & Ylvisaker, D. (1990). Minimax and
    maximin distance designs. *Journal of Statistical Planning and
    Inference*, 26(2), 131-148.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist


__all__ = ["maximin_design", "minimax_design"]


def maximin_design(
    n_points: int,
    n_factors: int,
    *,
    iterations: int = 200,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a maximin distance Latin hypercube design.

    Each factor (column) is a random permutation of the cell indices
    ``0, ..., n_points - 1``, with points placed at cell centers. A
    coordinate-exchange local search swaps pairs of cell indices
    within a randomly chosen column to maximize the minimum pairwise
    Euclidean distance between design points.

    Parameters
    ----------
    n_points : int
        Number of design points, must be at least 2.
    n_factors : int
        Number of factors (dimensions), must be at least 1.
    iterations : int, optional
        Number of local-search iterations, must be at least 0.
        Default is 200.
    seed : int or numpy.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    design : ndarray of shape (n_points, n_factors)
        Design points in :math:`[0, 1)^{\\text{n\\_factors}}`, with
        each column a Latin hypercube permutation of cell centers.

    Raises
    ------
    ValueError
        If ``n_points < 2``, ``n_factors < 1``, or ``iterations < 0``.

    Examples
    --------
    >>> design = maximin_design(5, 2, iterations=50, seed=0)
    >>> design.shape
    (5, 2)
    >>> bool(np.all((design >= 0) & (design < 1)))
    True

    Each column is a Latin hypercube permutation of cell centers:

    >>> cells = np.floor(design[:, 0] * 5).astype(int)
    >>> np.sort(cells)
    array([0, 1, 2, 3, 4])
    """
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2, got {n_points}")
    if n_factors < 1:
        raise ValueError(f"n_factors must be at least 1, got {n_factors}")
    if iterations < 0:
        raise ValueError(f"iterations must be at least 0, got {iterations}")

    rng = np.random.default_rng(seed)
    cells = np.empty((n_points, n_factors), dtype=int)
    for j in range(n_factors):
        cells[:, j] = rng.permutation(n_points)

    design = (cells + 0.5) / n_points
    best_min_dist = pdist(design).min()

    for _ in range(iterations):
        j = rng.integers(n_factors)
        r1, r2 = rng.choice(n_points, size=2, replace=False)
        cells[r1, j], cells[r2, j] = cells[r2, j], cells[r1, j]
        candidate = (cells + 0.5) / n_points
        candidate_min_dist = pdist(candidate).min()
        if candidate_min_dist > best_min_dist:
            design = candidate
            best_min_dist = candidate_min_dist
        else:
            cells[r1, j], cells[r2, j] = cells[r2, j], cells[r1, j]

    return design


def minimax_design(
    n_points: int,
    n_factors: int,
    *,
    n_candidates: int = 1000,
    iterations: int = 200,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a minimax distance design.

    Points are selected from a random candidate set in
    :math:`[0, 1)^{\\text{n\\_factors}}` to minimize the maximum
    distance from any candidate point to its nearest selected design
    point. After a random initialization, a local search swaps a
    selected point with an unselected candidate whenever doing so
    reduces this criterion.

    Parameters
    ----------
    n_points : int
        Number of design points, must be at least 1.
    n_factors : int
        Number of factors (dimensions), must be at least 1.
    n_candidates : int, optional
        Number of candidate points to sample, must be at least
        ``n_points``. Default is 1000.
    iterations : int, optional
        Number of local-search iterations, must be at least 0.
        Default is 200.
    seed : int or numpy.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    design : ndarray of shape (n_points, n_factors)
        Design points in :math:`[0, 1)^{\\text{n\\_factors}}`, selected
        from the candidate set.

    Raises
    ------
    ValueError
        If ``n_points < 1``, ``n_factors < 1``,
        ``n_candidates < n_points``, or ``iterations < 0``.

    Examples
    --------
    >>> design = minimax_design(5, 2, n_candidates=200, seed=0)
    >>> design.shape
    (5, 2)
    >>> bool(np.all((design >= 0) & (design < 1)))
    True
    >>> np.unique(design, axis=0).shape[0]
    5
    """
    if n_points < 1:
        raise ValueError(f"n_points must be at least 1, got {n_points}")
    if n_factors < 1:
        raise ValueError(f"n_factors must be at least 1, got {n_factors}")
    if n_candidates < n_points:
        raise ValueError(
            f"n_candidates must be at least n_points, got "
            f"n_candidates={n_candidates}, n_points={n_points}"
        )
    if iterations < 0:
        raise ValueError(f"iterations must be at least 0, got {iterations}")

    rng = np.random.default_rng(seed)
    candidates = rng.random((n_candidates, n_factors))

    selected = rng.choice(n_candidates, size=n_points, replace=False)

    def criterion(selected_idx: np.ndarray) -> float:
        distances = cdist(candidates, candidates[selected_idx])
        return distances.min(axis=1).max()

    best_criterion = criterion(selected)
    all_indices = np.arange(n_candidates)

    for _ in range(iterations):
        unselected = np.setdiff1d(all_indices, selected, assume_unique=True)
        pos = rng.integers(n_points)
        new_idx = rng.choice(unselected)
        old_idx = selected[pos]
        selected[pos] = new_idx
        new_criterion = criterion(selected)
        if new_criterion < best_criterion:
            best_criterion = new_criterion
        else:
            selected[pos] = old_idx

    return candidates[selected]
