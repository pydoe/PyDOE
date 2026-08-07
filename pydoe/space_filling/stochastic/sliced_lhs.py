"""
Sliced Latin hypercube designs.

A sliced Latin hypercube design (SLHD) partitions an
:math:`N = m t`-point Latin hypercube design into :math:`t` slices of
:math:`m` points each, such that the *full* design is a Latin
hypercube over :math:`N` cells *and* every individual slice, once
rescaled to its own unit hypercube, is itself a Latin hypercube over
:math:`m` cells. SLHDs are useful for computer experiments with a
mixture of qualitative levels (one per slice) and quantitative
factors, or for batch sequential designs.

References
----------
Qian, P. Z. G. (2012). Sliced Latin hypercube designs. *Journal of the
    American Statistical Association*, 107(497), 393-399.
"""

from __future__ import annotations

import numpy as np


__all__ = ["sliced_lhs"]


def sliced_lhs(
    n_factors: int,
    m: int,
    t: int,
    seed: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Generate a sliced Latin hypercube design.

    For each factor, the :math:`N = mt` cell indices :math:`0, \\ldots,
    N - 1` are split into :math:`t` consecutive blocks of size
    :math:`m`. Within block :math:`s`, the :math:`m` cell indices are
    assigned to the :math:`m` rows of slice :math:`s` in random order.
    The result is jittered uniformly within each cell.

    Parameters
    ----------
    n_factors : int
        Number of factors (dimensions), must be at least 1.
    m : int
        Number of points per slice, must be at least 1.
    t : int
        Number of slices, must be at least 1.
    seed : int or numpy.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    design : ndarray of shape (m * t, n_factors)
        Latin hypercube design in :math:`[0, 1)^{\\text{n\\_factors}}`
        with :math:`m t` cells.
    slices : ndarray of shape (m * t,)
        Slice label (0 to ``t - 1``) for each row of ``design``.

    Raises
    ------
    ValueError
        If ``n_factors``, ``m``, or ``t`` is less than 1.

    Examples
    --------
    >>> design, slices = sliced_lhs(2, 3, 2, seed=0)
    >>> design.shape
    (6, 2)
    >>> slices
    array([0, 0, 0, 1, 1, 1])

    Each slice, rescaled to its own unit hypercube via
    ``design * t - slices[:, None]``, is itself a Latin hypercube
    design with ``m`` cells:

    >>> rescaled = design * 2 - slices[:, None]
    >>> sub = rescaled[slices == 0]
    >>> sorted((sub[:, 0] * 3).astype(int).tolist())
    [0, 1, 2]
    """
    if n_factors < 1 or m < 1 or t < 1:
        raise ValueError(
            f"n_factors, m, and t must all be at least 1, got "
            f"n_factors={n_factors}, m={m}, t={t}"
        )

    rng = np.random.default_rng(seed)
    n_runs = m * t
    cells = np.empty((n_runs, n_factors), dtype=int)
    for j in range(n_factors):
        for s in range(t):
            cells[s * m : (s + 1) * m, j] = s * m + rng.permutation(m)

    jitter = rng.random((n_runs, n_factors))
    design = (cells + jitter) / n_runs
    slices = np.repeat(np.arange(t), m)
    return design, slices
