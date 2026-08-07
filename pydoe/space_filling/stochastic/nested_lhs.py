"""
Nested Latin hypercube designs.

A nested Latin hypercube design (NLHD) consists of two Latin hypercube
designs, a "small" design with :math:`n_1` points and a "large" design
with :math:`n_2 = n_1 k` points, such that the small design is nested
within the large one: every cell of the small design corresponds to a
contiguous block of :math:`k` cells in the large design. NLHDs are
useful for multi-fidelity computer experiments, where the small design
is run on an expensive high-fidelity simulator and the large design on
a cheaper low-fidelity simulator.

References
----------
Qian, P. Z. G. (2009). Nested Latin hypercube designs. *Biometrika*,
    96(4), 957-970.
"""

from __future__ import annotations

import numpy as np


__all__ = ["nested_lhs"]


def nested_lhs(
    n_factors: int,
    n1: int,
    k: int,
    seed: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Generate a nested Latin hypercube design.

    For each factor, the :math:`n_1` cells of the small design are
    assigned a random permutation of :math:`0, \\ldots, n_1 - 1`. Each
    small cell :math:`v` is then expanded into a block of :math:`k`
    consecutive cells :math:`vk, \\ldots, vk + k - 1` in the large
    design of :math:`n_2 = n_1 k` cells, with the :math:`k` cells
    within each block assigned in random order. Both designs are
    jittered uniformly within each cell.

    Parameters
    ----------
    n_factors : int
        Number of factors (dimensions), must be at least 1.
    n1 : int
        Number of points in the small design, must be at least 1.
    k : int
        Ratio of large to small design size, must be at least 1. The
        large design has :math:`n_1 k` points.
    seed : int or numpy.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    small_design : ndarray of shape (n1, n_factors)
        Latin hypercube design in :math:`[0, 1)^{\\text{n\\_factors}}`
        with :math:`n_1` cells.
    large_design : ndarray of shape (n1 * k, n_factors)
        Latin hypercube design in :math:`[0, 1)^{\\text{n\\_factors}}`
        with :math:`n_1 k` cells, nested around ``small_design``.

    Raises
    ------
    ValueError
        If ``n_factors``, ``n1``, or ``k`` is less than 1.

    Examples
    --------
    >>> small, large = nested_lhs(2, 3, 2, seed=0)
    >>> small.shape
    (3, 2)
    >>> large.shape
    (6, 2)

    Each cell of the small design corresponds to a block of ``k``
    consecutive cells of the large design:

    >>> small_cells = np.floor(small[:, 0] * 3).astype(int)
    >>> large_cells = np.floor(large[:, 0] * 6).astype(int)
    >>> bool(np.all(large_cells // 2 == np.repeat(small_cells, 2)))
    True
    """
    if n_factors < 1 or n1 < 1 or k < 1:
        raise ValueError(
            f"n_factors, n1, and k must all be at least 1, got "
            f"n_factors={n_factors}, n1={n1}, k={k}"
        )

    rng = np.random.default_rng(seed)
    n2 = n1 * k
    small_cells = np.empty((n1, n_factors), dtype=int)
    large_cells = np.empty((n2, n_factors), dtype=int)
    for j in range(n_factors):
        small_cells[:, j] = rng.permutation(n1)
        for i in range(n1):
            tau = rng.permutation(k)
            base = small_cells[i, j] * k
            large_cells[i * k : (i + 1) * k, j] = base + tau

    small_jitter = rng.random((n1, n_factors))
    large_jitter = rng.random((n2, n_factors))
    small_design = (small_cells + small_jitter) / n1
    large_design = (large_cells + large_jitter) / n2
    return small_design, large_design
