"""
Sequential (adaptive) design driver for expensive black-box functions.

Sequential design iteratively builds a set of evaluation points by
alternating between fitting a Gaussian process surrogate to the points
evaluated so far and selecting the next point to evaluate by
maximizing an acquisition function over candidate points. This is
useful for optimizing expensive black-box objective functions, where
each evaluation of ``objective`` is costly and the goal is to find a
good optimum with as few evaluations as possible.

References
----------
Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global
    optimization of expensive black-box functions. *Journal of Global
    Optimization*, 13(4), 455-492.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from pydoe.sequential.acquisition import (
    expected_improvement,
    probability_of_improvement,
    upper_confidence_bound,
)
from pydoe.sequential.gaussian_process import GaussianProcessRegressor
from pydoe.space_filling.stochastic import lhs
from pydoe.utils import scale_samples


__all__ = ["sequential_design"]


def sequential_design(  # ruff: ignore[too-many-arguments]
    objective: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    n_initial: int,
    n_iter: int,
    *,
    acquisition: str = "ei",
    maximize: bool = True,
    n_candidates: int = 1000,
    length_scale: float = 1.0,
    seed: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run a sequential (Bayesian-optimization-style) adaptive design.

    An initial space-filling Latin hypercube design of ``n_initial``
    points is evaluated, then ``n_iter`` additional points are chosen
    one at a time by fitting a Gaussian process surrogate to all
    points evaluated so far and maximizing an acquisition function
    over randomly sampled candidate points.

    Parameters
    ----------
    objective : Callable[[ndarray], float]
        Black-box objective function. Takes a 1D array of shape
        ``(d,)`` representing a point in the original ``bounds``
        space and returns a scalar float.
    bounds : ndarray of shape (d, 2)
        Lower and upper bounds for each of the ``d`` dimensions, one
        row ``[low, high]`` per dimension.
    n_initial : int
        Number of initial space-filling samples, must be at least 1.
    n_iter : int
        Number of sequential (adaptive) iterations after the initial
        design, must be non-negative.
    acquisition : str, optional
        Acquisition function to use, one of ``"ei"`` (expected
        improvement), ``"pi"`` (probability of improvement), or
        ``"ucb"`` (upper confidence bound). Default is ``"ei"``.
    maximize : bool, optional
        Whether ``objective`` is being maximized. Default is True.
    n_candidates : int, optional
        Number of random candidate points evaluated by the
        acquisition function at each iteration. Default is 1000.
    length_scale : float, optional
        Length scale passed to the internal
        :class:`~pydoe.sequential.gaussian_process.\
GaussianProcessRegressor`. Default is 1.0.
    seed : int or numpy.random.Generator, optional
        Seed or generator for the initial design and candidate
        sampling.

    Returns
    -------
    X : ndarray of shape (n_initial + n_iter, d)
        All evaluated input points, in the original ``bounds`` space.
    y : ndarray of shape (n_initial + n_iter,)
        Objective values at each point in ``X``.

    Raises
    ------
    ValueError
        If ``bounds`` does not have shape ``(d, 2)``, if any lower
        bound is not strictly less than the corresponding upper bound,
        if ``n_initial < 1``, if ``n_iter < 0``, or if ``acquisition``
        is not one of ``"ei"``, ``"pi"``, or ``"ucb"``.

    Examples
    --------
    >>> import numpy as np
    >>> def neg_quadratic(x):
    ...     return -float((x[0] - 0.3) ** 2)
    >>> bounds = np.array([[0.0, 1.0]])
    >>> X, y = sequential_design(
    ...     neg_quadratic, bounds, n_initial=4, n_iter=5, seed=0
    ... )
    >>> X.shape
    (9, 1)
    >>> y.shape
    (9,)
    >>> bool(y.max() > y[:4].max())
    True
    """
    bounds = np.asarray(bounds, dtype=float)

    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError(f"bounds must have shape (d, 2), got {bounds.shape}")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError(
            "each row of bounds must satisfy low < high, got "
            f"bounds={bounds.tolist()}"
        )
    if n_initial < 1:
        raise ValueError(f"n_initial must be at least 1, got {n_initial}")
    if n_iter < 0:
        raise ValueError(f"n_iter must be non-negative, got {n_iter}")

    acquisitions = {"ei", "pi", "ucb"}
    if acquisition not in acquisitions:
        raise ValueError(
            f"acquisition must be one of {sorted(acquisitions)}, got "
            f"{acquisition!r}"
        )

    rng = np.random.default_rng(seed)
    d = bounds.shape[0]
    low = bounds[:, 0]
    high = bounds[:, 1]
    span = high - low

    bound_pairs = [(float(lo), float(hi)) for lo, hi in bounds]
    unit_design = lhs(d, samples=n_initial, seed=rng)
    X = scale_samples(unit_design, bound_pairs)
    y = np.array([objective(x) for x in X])

    for _ in range(n_iter):
        X_norm = (X - low) / span

        gp = GaussianProcessRegressor(length_scale=length_scale)
        gp.fit(X_norm, y)

        candidates = rng.uniform(low, high, size=(n_candidates, d))
        candidates_norm = (candidates - low) / span

        mean, std = gp.predict(candidates_norm, return_std=True)
        best_f = y.max() if maximize else y.min()

        if acquisition == "ei":
            scores = expected_improvement(mean, std, best_f, maximize=maximize)
        elif acquisition == "pi":
            scores = probability_of_improvement(
                mean, std, best_f, maximize=maximize
            )
        else:
            scores = upper_confidence_bound(mean, std, maximize=maximize)

        next_point = candidates[np.argmax(scores)]
        y_next = objective(next_point)

        X = np.vstack([X, next_point])
        y = np.append(y, y_next)

    return X, y
