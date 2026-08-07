"""
Acquisition functions for Bayesian optimization.

These functions score candidate points based on the predictive
distribution (mean and standard deviation) of a Gaussian process
surrogate, balancing exploitation of promising regions against
exploration of uncertain regions.

For all three functions, **higher returned values indicate more
promising candidate points**, regardless of whether the underlying
objective is being maximized or minimized.

References
----------
Mockus, J. (1989). *Bayesian Approach to Global Optimization*. Kluwer
    Academic Publishers.
Kushner, H. J. (1964). A new method of locating the maximum point of
    an arbitrary multipeak curve in the presence of noise. *Journal of
    Basic Engineering*, 86(1), 97-106.
Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. (2010). Gaussian
    process optimization in the bandit setting: No regret and
    experimental design. *ICML*.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


__all__ = [
    "expected_improvement",
    "probability_of_improvement",
    "upper_confidence_bound",
]


def _improvement(
    mean: np.ndarray, best_f: float, *, maximize: bool
) -> np.ndarray:
    """
    Compute the raw improvement of ``mean`` over ``best_f``.

    Parameters
    ----------
    mean : ndarray
        Predicted means.
    best_f : float
        Best observed objective value so far.
    maximize : bool
        Whether the objective is being maximized.

    Returns
    -------
    ndarray
        Improvement values, ``mean - best_f`` if maximizing, else
        ``best_f - mean``.
    """
    if maximize:
        return mean - best_f
    return best_f - mean


def expected_improvement(
    mean: np.ndarray, std: np.ndarray, best_f: float, *, maximize: bool = True
) -> np.ndarray:
    """
    Compute the expected improvement (EI) acquisition function.

    Parameters
    ----------
    mean : ndarray of shape (n,)
        Predicted means at the candidate points.
    std : ndarray of shape (n,)
        Predicted standard deviations at the candidate points.
    best_f : float
        Best objective value observed so far.
    maximize : bool, optional
        Whether the objective is being maximized. Default is True.

    Returns
    -------
    ndarray of shape (n,)
        Expected improvement at each candidate point, clipped to be
        non-negative.

    Raises
    ------
    ValueError
        If ``mean`` and ``std`` have different shapes, or if ``std``
        contains negative values.

    Examples
    --------
    >>> import numpy as np
    >>> mean = np.array([0.5, 1.5])
    >>> std = np.array([0.1, 0.2])
    >>> ei = expected_improvement(mean, std, best_f=1.0)
    >>> ei.shape
    (2,)
    >>> bool(ei[1] > ei[0])
    True
    """
    if mean.shape != std.shape:
        raise ValueError(
            f"mean and std must have the same shape, got "
            f"{mean.shape} and {std.shape}"
        )
    if np.any(std < 0):
        raise ValueError("std must be non-negative")

    imp = _improvement(mean, best_f, maximize=maximize)

    nonzero = std > 0
    z = np.zeros_like(mean)
    z[nonzero] = imp[nonzero] / std[nonzero]

    ei = np.where(
        nonzero, imp * norm.cdf(z) + std * norm.pdf(z), np.maximum(imp, 0.0)
    )
    return np.clip(ei, 0.0, None)


def probability_of_improvement(
    mean: np.ndarray, std: np.ndarray, best_f: float, *, maximize: bool = True
) -> np.ndarray:
    """
    Compute the probability of improvement (PI) acquisition function.

    Parameters
    ----------
    mean : ndarray of shape (n,)
        Predicted means at the candidate points.
    std : ndarray of shape (n,)
        Predicted standard deviations at the candidate points.
    best_f : float
        Best objective value observed so far.
    maximize : bool, optional
        Whether the objective is being maximized. Default is True.

    Returns
    -------
    ndarray of shape (n,)
        Probability of improvement at each candidate point, in
        ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``mean`` and ``std`` have different shapes, or if ``std``
        contains negative values.

    Examples
    --------
    >>> import numpy as np
    >>> mean = np.array([0.5, 1.5])
    >>> std = np.array([0.1, 0.2])
    >>> pi = probability_of_improvement(mean, std, best_f=1.0)
    >>> pi.shape
    (2,)
    >>> bool(pi[1] > pi[0])
    True
    """
    if mean.shape != std.shape:
        raise ValueError(
            f"mean and std must have the same shape, got "
            f"{mean.shape} and {std.shape}"
        )
    if np.any(std < 0):
        raise ValueError("std must be non-negative")

    imp = _improvement(mean, best_f, maximize=maximize)

    nonzero = std > 0
    z = np.zeros_like(mean)
    z[nonzero] = imp[nonzero] / std[nonzero]

    pi = np.where(nonzero, norm.cdf(z), np.where(imp > 0, 1.0, 0.0))
    return pi


def upper_confidence_bound(
    mean: np.ndarray,
    std: np.ndarray,
    *,
    kappa: float = 1.96,
    maximize: bool = True,
) -> np.ndarray:
    """
    Compute the (generalized) upper confidence bound acquisition.

    For ``maximize=True`` this is the usual upper confidence bound
    ``mean + kappa * std``. For ``maximize=False`` this is the
    *negated* lower confidence bound, ``-(mean - kappa * std)``, so
    that in both cases higher returned values indicate more promising
    candidate points.

    Parameters
    ----------
    mean : ndarray of shape (n,)
        Predicted means at the candidate points.
    std : ndarray of shape (n,)
        Predicted standard deviations at the candidate points.
    kappa : float, optional
        Exploration-exploitation trade-off parameter, must be
        strictly positive. Default is 1.96.
    maximize : bool, optional
        Whether the objective is being maximized. Default is True.

    Returns
    -------
    ndarray of shape (n,)
        Acquisition score at each candidate point, where higher
        values indicate more promising points.

    Raises
    ------
    ValueError
        If ``mean`` and ``std`` have different shapes, if ``std``
        contains negative values, or if ``kappa`` is not strictly
        positive.

    Examples
    --------
    >>> import numpy as np
    >>> mean = np.array([0.5, 1.5])
    >>> std = np.array([0.1, 0.2])
    >>> ucb = upper_confidence_bound(mean, std)
    >>> ucb.shape
    (2,)
    >>> bool(ucb[1] > ucb[0])
    True
    """
    if mean.shape != std.shape:
        raise ValueError(
            f"mean and std must have the same shape, got "
            f"{mean.shape} and {std.shape}"
        )
    if np.any(std < 0):
        raise ValueError("std must be non-negative")
    if kappa <= 0:
        raise ValueError(f"kappa must be strictly positive, got {kappa}")

    if maximize:
        return mean + kappa * std
    return -(mean - kappa * std)
