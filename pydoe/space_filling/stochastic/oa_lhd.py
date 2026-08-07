"""
Orthogonal-array-based Latin hypercube designs.

Tang's (1993) construction turns any symmetric orthogonal array
:math:`OA(N, k, s, 2)` (:math:`N` runs, :math:`k` factors, :math:`s`
levels, strength 2) into a Latin hypercube design with :math:`N`
points in :math:`k` dimensions. Within each level group of a column,
the rows are assigned a random permutation of the corresponding block
of :math:`N/s` Latin hypercube cells, so every column remains a
permutation of all :math:`N` cells while the two-dimensional balance
of the orthogonal array is preserved -- giving better
two-dimensional uniformity than a plain random LHS.

References
----------
Tang, B. (1993). Orthogonal array-based Latin hypercubes. *Journal of
    the American Statistical Association*, 88(424), 1392-1397.
NIST/SEMATECH e-Handbook of Statistical Methods, Section 5.3.3.6 -
    "Latin hypercube designs",
    https://www.itl.nist.gov/div898/handbook/pri/section3/pri336.htm
"""

from __future__ import annotations

import numpy as np


__all__ = ["oa_lhd"]


def oa_lhd(
    oa: np.ndarray, seed: int | np.random.Generator | None = None
) -> np.ndarray:
    r"""
    Build an orthogonal-array-based Latin hypercube design.

    Parameters
    ----------
    oa : array_like of shape (N, k)
        A symmetric orthogonal array with integer levels
        :math:`0, \\ldots, s - 1`, each level appearing exactly
        :math:`N / s` times in every column (e.g. from
        [`get_orthogonal_array`][pydoe.get_orthogonal_array]).
    seed : int or numpy.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    ndarray of shape (N, k)
        Latin hypercube design in :math:`[0, 1)^k`. Every column is a
        random permutation of the :math:`N` cell midpoints (jittered
        within each cell), with cell ranges respecting the level
        structure of ``oa``.

    Raises
    ------
    ValueError
        If ``N`` is not evenly divisible by the number of distinct
        levels in ``oa``.

    Examples
    --------
    >>> from pydoe import get_orthogonal_array
    >>> oa = get_orthogonal_array("L9(3^4)")
    >>> design = oa_lhd(oa, seed=0)
    >>> design.shape
    (9, 4)

    Every column is a Latin hypercube permutation of the 9 cells:

    >>> sorted((design[:, 0] * 9).astype(int).tolist())
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
    """
    oa = np.asarray(oa, dtype=int)
    n_runs, n_factors = oa.shape
    levels = np.unique(oa)
    n_levels = len(levels)
    if n_runs % n_levels != 0:
        raise ValueError(
            f"N = {n_runs} must be divisible by the number of levels "
            f"({n_levels})"
        )

    rng = np.random.default_rng(seed)
    block_size = n_runs // n_levels
    design = np.empty((n_runs, n_factors), dtype=int)
    for j in range(n_factors):
        for level_idx, level in enumerate(levels):
            rows = np.where(oa[:, j] == level)[0]
            design[rows, j] = level_idx * block_size + rng.permutation(
                block_size
            )

    jitter = rng.random((n_runs, n_factors))
    return (design + jitter) / n_runs
