"""
Nearly orthogonal Latin hypercube designs.

A standard random Latin hypercube design (LHS) guarantees uniform
marginal coverage of every factor, but the pairwise correlations
between columns of the design matrix can be substantial. Such
correlations confound the estimated effects of the corresponding
factors when the design is used to fit a regression or surrogate
model. A nearly orthogonal Latin hypercube (NOLH) design is a Latin
hypercube that additionally minimizes the pairwise correlations
between columns, so that factor effects can be estimated with much
less confounding while still preserving the space-filling and
stratification properties of the underlying Latin hypercube.

References
----------
Cioppa, T. M., & Lucas, T. W. (2007). Efficient nearly orthogonal and
    space-filling Latin hypercubes. *Technometrics*, 49(1), 45-55.

Notes
-----
Cioppa & Lucas constructed NOLH designs from precomputed tables for a
fixed set of design sizes. This implementation instead uses an
optimization-based (coordinate-exchange) construction: starting from a
random Latin hypercube, pairs of cell indices within a single column
are repeatedly swapped, keeping the swap only if it strictly reduces
the maximum absolute pairwise Pearson correlation between any two
columns of the design. This directly targets near-orthogonality for
arbitrary numbers of points and factors, at the cost of not matching
the exact tabulated designs of the original paper.
"""

from __future__ import annotations

import numpy as np


__all__ = ["nearly_orthogonal_lhs"]


def _max_abs_corr(design: np.ndarray) -> float:
    """
    Compute the maximum absolute off-diagonal correlation.

    Parameters
    ----------
    design : ndarray
        Design matrix with one column per factor.

    Returns
    -------
    float
        Maximum absolute pairwise Pearson correlation between any
        two columns of ``design``.
    """
    corr = np.corrcoef(design, rowvar=False)
    n_factors = corr.shape[0]
    mask = ~np.eye(n_factors, dtype=bool)
    return float(np.max(np.abs(corr[mask])))


def nearly_orthogonal_lhs(
    n_points: int,
    n_factors: int,
    *,
    iterations: int = 200,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    r"""
    Generate a nearly orthogonal Latin hypercube design.

    A random Latin hypercube design with :math:`n` cells per factor is
    constructed, then refined by a coordinate-exchange local search:
    at each iteration, two cell indices within a randomly chosen
    column are swapped, and the swap is kept only if it strictly
    decreases the maximum absolute pairwise Pearson correlation
    between any two columns of the design matrix.

    Parameters
    ----------
    n_points : int
        Number of points (rows) in the design, must be at least 2.
    n_factors : int
        Number of factors (columns) in the design, must be at least
        1. If ``n_factors == 1`` there is no correlation to optimize
        and a plain one-column Latin hypercube is returned.
    iterations : int, optional
        Number of coordinate-exchange iterations to attempt, must be
        at least 0. Default is 200.
    seed : int or numpy.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    design : ndarray of shape (n_points, n_factors)
        Nearly orthogonal Latin hypercube design in
        :math:`[0, 1)^{\\text{n\\_factors}}`.

    Raises
    ------
    ValueError
        If ``n_points < 2``, ``n_factors < 1``, or ``iterations < 0``.

    Examples
    --------
    >>> design = nearly_orthogonal_lhs(8, 3, iterations=100, seed=0)
    >>> design.shape
    (8, 3)
    >>> bool(np.all((design >= 0) & (design < 1)))
    True

    Each column is a Latin hypercube with 8 cells:

    >>> cells = np.floor(design[:, 0] * 8).astype(int)
    >>> sorted(cells.tolist())
    [0, 1, 2, 3, 4, 5, 6, 7]
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

    if n_factors < 2:
        return (cells + 0.5) / n_points

    design = (cells + 0.5) / n_points
    best_criterion = _max_abs_corr(design)
    for _ in range(iterations):
        j = rng.integers(n_factors)
        r1, r2 = rng.choice(n_points, size=2, replace=False)

        cells[r1, j], cells[r2, j] = cells[r2, j], cells[r1, j]
        design = (cells + 0.5) / n_points
        criterion = _max_abs_corr(design)

        if criterion < best_criterion:
            best_criterion = criterion
        else:
            cells[r1, j], cells[r2, j] = cells[r2, j], cells[r1, j]

    return (cells + 0.5) / n_points
