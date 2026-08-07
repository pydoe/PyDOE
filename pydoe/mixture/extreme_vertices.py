"""
Constrained mixture designs (extreme-vertices designs).

When each mixture component is restricted to a sub-range
:math:`l_i \\le x_i \\le u_i` (with :math:`\\sum l_i \\le 1 \\le \\sum
u_i`), the feasible region is no longer the full simplex but a smaller
polytope nested inside it. An extreme-vertices design places a design
point at every vertex of this polytope.

References
----------
Cornell, J. A. (2002). *Experiments with Mixtures: Designs, Models, and the
    Analysis of Mixture Data* (3rd ed.). Wiley.
McLean, R. A., & Anderson, V. L. (1966). Extreme vertices design of mixture
    experiments. *Technometrics*, 8(3), 447-454.
NIST/SEMATECH e-Handbook of Statistical Methods, Section 5.6.3.2 -
    "Constrained mixture designs",
    https://www.itl.nist.gov/div898/handbook/pri/section6/pri63.htm
"""

from __future__ import annotations

from itertools import product

import numpy as np


__all__ = ["extreme_vertices_design"]


def extreme_vertices_design(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    r"""
    Generate an extreme-vertices design for a constrained mixture region.

    Each component *i* is restricted to :math:`l_i \\le x_i \\le u_i`.
    For each choice of a "free" component *f*, every other component is
    fixed at either its lower or upper bound, and the free component is
    set to :math:`x_f = 1 - \\sum_{i \\ne f} x_i` so that the simplex
    constraint :math:`\\sum_i x_i = 1` holds. A candidate point is kept
    only if :math:`l_f \\le x_f \\le u_f`. Duplicate points are removed
    and the result is sorted lexicographically.

    Parameters
    ----------
    lower : array_like of shape (q,)
        Lower bound :math:`l_i` for each of the *q* components, with
        all entries non-negative and :math:`\\sum_i l_i \\le 1`.
    upper : array_like of shape (q,)
        Upper bound :math:`u_i` for each of the *q* components, with
        all entries at most 1 and :math:`\\sum_i u_i \\ge 1`.

    Returns
    -------
    ndarray of shape (n_vertices, q)
        Extreme vertices of the constrained mixture region. Every row
        sums to 1 with entries respecting ``lower`` and ``upper``.

    Raises
    ------
    ValueError
        If ``lower`` and ``upper`` do not have the same length, if any
        entry of ``lower`` exceeds the corresponding entry of
        ``upper``, if :math:`\\sum_i l_i > 1`, or if
        :math:`\\sum_i u_i < 1`.

    Examples
    --------
    >>> extreme_vertices_design([0.1, 0.1, 0.1], [0.6, 0.6, 0.6])
    array([[0.1, 0.3, 0.6],
           [0.1, 0.6, 0.3],
           [0.3, 0.1, 0.6],
           [0.3, 0.6, 0.1],
           [0.6, 0.1, 0.3],
           [0.6, 0.3, 0.1]])

    Every vertex sums to 1:

    >>> verts = extreme_vertices_design([0.0, 0.2, 0.0], [1.0, 0.5, 0.7])
    >>> np.allclose(verts.sum(axis=1), 1.0)
    True

    !!! note
        This algorithm enumerates vertices of the region defined by
        simple per-component bounds only. It does not support
        additional general linear constraints between components.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != upper.shape:
        raise ValueError(
            "lower and upper must have the same shape, got "
            f"{lower.shape} and {upper.shape}"
        )
    if np.any(lower > upper):
        raise ValueError("each lower bound must not exceed its upper bound")
    q = lower.shape[0]
    if np.sum(lower) > 1.0 + 1e-9:
        raise ValueError("sum of lower bounds must not exceed 1")
    if np.sum(upper) < 1.0 - 1e-9:
        raise ValueError("sum of upper bounds must be at least 1")

    tol = 1e-9
    vertices = []
    for free in range(q):
        others = [i for i in range(q) if i != free]
        for bits in product((0, 1), repeat=q - 1):
            x = np.empty(q)
            for bit, other in zip(bits, others, strict=True):
                x[other] = upper[other] if bit else lower[other]
            x[free] = 1.0 - np.sum(x[others])
            if lower[free] - tol <= x[free] <= upper[free] + tol:
                vertices.append(np.clip(x, lower, upper))

    arr = np.round(np.array(vertices), 10)
    return np.unique(arr, axis=0)
