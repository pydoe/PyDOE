"""
Iman-Conover method for inducing rank correlations.

The Iman-Conover method (Iman & Conover, 1982) rearranges the rows of a
sample matrix so that the resulting columns have approximately a target
rank correlation structure, while leaving the marginal distribution of
each column unchanged. It is widely used in Monte Carlo simulation and
sensitivity analysis to model correlated uncertain parameters that have
been sampled independently from arbitrary marginal distributions.

References
----------
Iman, R. L., and Conover, W. J. (1982). "A distribution-free approach to
    inducing rank correlation among input variables." *Communications in
    Statistics - Simulation and Computation*, 11(3), 311-334.
    https://doi.org/10.1080/03610918208812265
"""

from __future__ import annotations

import numpy as np


__all__ = ["iman_conover"]


def iman_conover(
    data: np.ndarray,
    correlation_matrix: np.ndarray,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """
    Rearrange samples to induce a target rank correlation structure.

    Each column of ``data`` is treated as an independent sample from some
    marginal distribution. The Iman-Conover algorithm reorders the values
    within each column so that the resulting matrix has a Spearman rank
    correlation matrix close to ``correlation_matrix``, without changing
    the set of values (and therefore the marginal distribution) in any
    column.

    The algorithm works by:

    1. Drawing an auxiliary matrix of independent standard normal scores
       with the same shape as ``data``.
    2. Removing the (incidental) correlation among the score columns via
       a Cholesky decomposition of their sample correlation matrix.
    3. Imposing the target correlation structure on the scores via a
       Cholesky decomposition of ``correlation_matrix``.
    4. Sorting each column of ``data`` and reordering it according to the
       rank order of the corresponding transformed score column.

    Parameters
    ----------
    data : array_like of shape (n, k)
        Sample matrix with ``n`` observations of ``k`` variables. Each
        column may come from a different marginal distribution.
    correlation_matrix : array_like of shape (k, k)
        Target correlation matrix. Must be symmetric positive definite.
    seed : int or numpy.random.Generator, optional
        Seed or generator used to draw the auxiliary score matrix
        (default: None).

    Returns
    -------
    reordered : ndarray of shape (n, k)
        A row-permuted version of ``data`` whose columns have
        approximately the rank correlation structure given by
        ``correlation_matrix``.

    Raises
    ------
    ValueError
        If ``data`` is not 2D, if ``correlation_matrix`` is not a square
        matrix matching the number of columns of ``data``, or if
        ``correlation_matrix`` is not positive definite.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> data = rng.normal(size=(1000, 2))
    >>> target = np.array([[1.0, 0.8], [0.8, 1.0]])
    >>> reordered = iman_conover(data, target, seed=0)
    >>> reordered.shape
    (1000, 2)
    >>> rank_corr = np.corrcoef(
    ...     reordered[:, 0].argsort().argsort(),
    ...     reordered[:, 1].argsort().argsort(),
    ... )[0, 1]
    >>> bool(abs(rank_corr - 0.8) < 0.05)
    True
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got {data.ndim}D")

    n, k = data.shape
    correlation_matrix = np.asarray(correlation_matrix, dtype=np.float64)
    if correlation_matrix.shape != (k, k):
        raise ValueError(
            "correlation_matrix must have shape "
            f"({k}, {k}), got {correlation_matrix.shape}"
        )

    try:
        target_factor = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError as err:
        raise ValueError(
            "correlation_matrix must be positive definite"
        ) from err

    rng = np.random.default_rng(seed)
    scores = rng.standard_normal((n, k))

    score_corr = np.corrcoef(scores, rowvar=False)
    score_factor = np.linalg.cholesky(score_corr)

    target_scores = scores @ np.linalg.inv(score_factor).T
    target_scores @= target_factor.T

    reordered = np.empty_like(data)
    for col in range(k):
        sorted_column = np.sort(data[:, col])
        rank_order = np.argsort(np.argsort(target_scores[:, col]))
        reordered[:, col] = sorted_column[rank_order]

    return reordered
