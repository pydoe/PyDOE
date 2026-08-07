"""
Supersaturated designs.

A supersaturated design has more two-level factors than runs
(:math:`k > n`). Such designs cannot estimate all main effects
simultaneously, but are useful in early-stage screening when only a
small fraction of the factors are believed to be active (effect
sparsity). Designs are constructed by random search to minimize
:math:`E(s^2)`, the average squared off-diagonal element of
:math:`X^T X`, which measures the average pairwise correlation between
factor columns -- smaller values give less-confounded estimates of the
active effects.

References
----------
Booth, K. H. V., & Cox, D. R. (1962). Some systematic supersaturated
    designs. *Technometrics*, 4(4), 489-495.
Lin, D. K. J. (1993). A new class of supersaturated designs.
    *Technometrics*, 35(1), 28-31.
NIST/SEMATECH e-Handbook of Statistical Methods, Section 5.3.3.4 -
    "Supersaturated Designs",
    https://www.itl.nist.gov/div898/handbook/pri/section3/pri334.htm
"""

from __future__ import annotations

import numpy as np


__all__ = ["supersaturated_design"]


def supersaturated_design(
    n_factors: int,
    n_runs: int,
    iterations: int = 1000,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    r"""
    Generate a two-level supersaturated design via random search.

    ``iterations`` random :math:`\\pm 1` matrices of shape
    ``(n_runs, n_factors)`` are drawn, and the one minimizing

    .. math::

        E(s^2) = \\frac{1}{k(k-1)} \\sum_{i \\ne j}
            \\left(X^T X\\right)_{ij}^2

    is returned, where :math:`k` = ``n_factors``. Lower :math:`E(s^2)`
    indicates lower average correlation between factor columns.

    Parameters
    ----------
    n_factors : int
        Number of two-level factors :math:`k`, must be greater than
        ``n_runs``.
    n_runs : int
        Number of runs :math:`n`, must be at least 2.
    iterations : int, optional
        Number of random candidate designs to evaluate. Defaults to
        1000.
    seed : int or numpy.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    ndarray of shape (n_runs, n_factors)
        Design matrix with coded levels -1 and 1, with the smallest
        observed :math:`E(s^2)` among the evaluated candidates.

    Raises
    ------
    ValueError
        If ``n_runs < 2`` or ``n_factors <= n_runs``.

    Examples
    --------
    >>> supersaturated_design(6, 4, iterations=200, seed=0)
    array([[-1.,  1.,  1., -1., -1.,  1.],
           [ 1.,  1., -1.,  1.,  1.,  1.],
           [ 1.,  1.,  1., -1.,  1., -1.],
           [ 1.,  1.,  1.,  1., -1., -1.]])
    """
    if n_runs < 2:
        raise ValueError(f"n_runs must be at least 2, got {n_runs}")
    if n_factors <= n_runs:
        raise ValueError(
            f"n_factors ({n_factors}) must be greater than n_runs "
            f"({n_runs}) for a supersaturated design"
        )

    rng = np.random.default_rng(seed)
    best = None
    best_es2 = np.inf
    for _ in range(iterations):
        X = rng.choice([-1.0, 1.0], size=(n_runs, n_factors))
        XtX = X.T @ X
        off_diag = XtX - np.diag(np.diag(XtX))
        es2 = np.sum(off_diag**2) / (n_factors * (n_factors - 1))
        if es2 < best_es2:
            best_es2 = es2
            best = X

    return best
