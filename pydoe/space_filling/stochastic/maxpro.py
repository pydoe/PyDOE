"""
Maximum projection (MaxPro) designs.

A maximum projection design minimizes the MaxPro criterion

.. math::

    \\psi(D) = \\sum_{i < j} \\frac{1}{\\prod_{k=1}^{p}
    (x_{ik} - x_{jk})^2}

over all pairs of points :math:`x_i, x_j` in the design :math:`D`.
Minimizing :math:`\\psi` pushes pairs of points apart in *every*
subset of factors simultaneously, so projections of the design onto
any subspace of factors retain good space-filling properties. This
makes MaxPro designs especially useful for computer experiments where
only a subset of the input factors may be active.

References
----------
Joseph, V. R., Gul, E., & Ba, S. (2015). Maximum projection designs
    for computer experiments. *Biometrika*, 102(2), 371-380.
"""

from __future__ import annotations

import numpy as np


__all__ = ["maxpro_design"]


def _maxpro_criterion(design: np.ndarray) -> float:
    """Compute the MaxPro criterion for a design.

    Returns
    -------
    float
        The MaxPro criterion value.
    """
    diff = design[:, None, :] - design[None, :, :]
    sq = diff**2
    prod = np.prod(sq, axis=2)
    n_points = design.shape[0]
    iu = np.triu_indices(n_points, k=1)
    return float(np.sum(1.0 / prod[iu]))


def maxpro_design(
    n_points: int,
    n_factors: int,
    *,
    iterations: int = 200,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a maximum projection (MaxPro) design.

    The design is initialized as a random Latin hypercube (each
    column is an independent permutation of the cell indices
    :math:`0, \\ldots, n - 1`, with points placed at cell centers) and
    then refined with a coordinate-exchange local search that swaps
    pairs of cell values within a column whenever doing so reduces
    the MaxPro criterion :math:`\\psi`. The Latin hypercube structure
    is preserved by every swap.

    Parameters
    ----------
    n_points : int
        Number of design points, must be at least 2.
    n_factors : int
        Number of factors (dimensions), must be at least 1.
    iterations : int, optional
        Number of coordinate-exchange iterations to attempt, must be
        non-negative. Default is 200.
    seed : int or numpy.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    design : ndarray of shape (n_points, n_factors)
        Design points in :math:`[0, 1)^{\\text{n\\_factors}}`.

    Raises
    ------
    ValueError
        If ``n_points`` is less than 2, ``n_factors`` is less than 1,
        or ``iterations`` is negative.

    Examples
    --------
    >>> design = maxpro_design(5, 2, iterations=50, seed=0)
    >>> design.shape
    (5, 2)
    >>> bool(np.all((design >= 0) & (design < 1)))
    True

    Each column is a Latin hypercube column with ``n_points`` cells:

    >>> cells = np.floor(design[:, 0] * 5).astype(int)
    >>> sorted(cells.tolist())
    [0, 1, 2, 3, 4]
    """
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2, got {n_points}")
    if n_factors < 1:
        raise ValueError(f"n_factors must be at least 1, got {n_factors}")
    if iterations < 0:
        raise ValueError(f"iterations must be non-negative, got {iterations}")

    rng = np.random.default_rng(seed)
    cells = np.empty((n_points, n_factors), dtype=int)
    for j in range(n_factors):
        cells[:, j] = rng.permutation(n_points)

    design = (cells + 0.5) / n_points
    psi = _maxpro_criterion(design)

    for _ in range(iterations):
        j = rng.integers(n_factors)
        r1, r2 = rng.choice(n_points, size=2, replace=False)

        cells[r1, j], cells[r2, j] = cells[r2, j], cells[r1, j]
        new_design = (cells + 0.5) / n_points
        new_psi = _maxpro_criterion(new_design)

        if new_psi < psi:
            design = new_design
            psi = new_psi
        else:
            cells[r1, j], cells[r2, j] = cells[r2, j], cells[r1, j]

    return design
