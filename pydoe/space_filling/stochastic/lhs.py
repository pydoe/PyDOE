"""
This code was originally published by the following individuals for use with
Scilab:
    Copyright (C) 2012 - 2013 - Michael Baudin
    Copyright (C) 2012 - Maria Christopoulou
    Copyright (C) 2010 - 2011 - INRIA - Michael Baudin
    Copyright (C) 2009 - Yann Collette
    Copyright (C) 2009 - CEA - Jean-Marc Martinez

    website: forge.scilab.org/index.php/p/scidoe/sourcetree/master/macros

Much thanks goes to these individuals. It has been converted to Python by
Abraham Lee.
"""

from __future__ import annotations

from typing import Literal
from warnings import warn

import numpy as np
from numpy import ma
from scipy import linalg, spatial, stats


__all__ = ["lhs"]


def lhs(  # ruff: ignore[too-many-branches, too-many-arguments, too-many-positional-arguments]
    n: int,
    samples: int | None = None,
    criterion: Literal[
        "center",
        "c",
        "maximin",
        "m",
        "centermaximin",
        "cm",
        "correlation",
        "corr",
    ]
    | None = None,
    iterations: int | None = None,
    random_state: int | np.random.RandomState | None = None,
    correlation_matrix: np.ndarray = None,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a latin-hypercube design

    Parameters
    ----------
    n : int
        The number of factors to generate samples for

    Optional
    --------
    samples : int
        The number of samples to generate for each factor (Default: n)
    criterion : str
        Allowable values are "center" or "c", "maximin" or "m",
        "centermaximin" or "cm", and "correlation" or "corr". If no value
        given, the design is simply randomized.
    iterations : int
        The number of iterations in the maximin and correlations algorithms
        (Default: 5).
    random_state : np.random.RandomState, int
         Random state (or seed-number) which controls the seed and random draws
         DEPRECATED! It will be removed in a future release.
         Use seed parameter instead.
    correlation_matrix : ndarray
         Enforce correlation between factors (only used in lhsmu)
    seed : int or np.random.Generator
         Seed or np.random.Generator which controls random draws

    Returns
    -------
    H : 2d-array
        An n-by-samples design matrix that has been normalized so factor values
        are uniformly spaced between zero and one.

    Raises
    ------
    ValueError
        If input parameters are invalid.

    Examples
    --------
    A 3-factor design (defaults to 3 samples)::

        >>> lhs(3, seed=42)
        array([[0.25798535, 0.14629281, 0.65854078],
               [0.56578934, 0.9286881 , 0.28619931],
               [0.9203799 , 0.36472578, 0.70937121]])

    A 4-factor design with 6 samples::

        >>> lhs(4, samples=6, seed=42)
        array([[0.60731085, 0.32927039, 0.8046052 , 0.98218685],
               [0.35468561, 0.67730288, 0.14309965, 0.77194407],
               [0.18236289, 0.89242099, 0.29352328, 0.53787312],
               [0.75909746, 0.63712694, 0.395133  , 0.48779416],
               [0.95968129, 0.40839766, 0.99511634, 0.116228  ],
               [0.12899267, 0.07314641, 0.57390237, 0.29767738]])

    A 2-factor design with 5 centered samples::

        >>> lhs(2, samples=5, criterion='center', seed=42)
        array([[0.9, 0.1],
               [0.1, 0.7],
               [0.5, 0.9],
               [0.7, 0.5],
               [0.3, 0.3]])

    A 3-factor design with 4 samples where the minimum distance between
    all samples has been maximized::

        >>> lhs(3, samples=4, criterion='maximin', seed=42)
        array([[0.91547913, 0.46335077, 0.05364617],
               [0.51457569, 0.10917935, 0.94597455],
               [0.35213216, 0.57034597, 0.30848487],
               [0.00770446, 0.88925804, 0.57339844]])

    A 4-factor design with 5 samples where the samples are as uncorrelated
    as possible (within 10 iterations)::

        >>> lhs(4, samples=5, criterion='correlation', iterations=10, seed=42)
        array([[0.15479121, 0.49007719, 0.68868284, 0.35721286],
               [0.21883547, 0.39512447, 0.4741596 , 0.64544774],
               [0.42562273, 0.08777569, 0.35222794, 0.92633288],
               [0.91091696, 0.76455232, 0.96552623, 0.585353  ],
               [0.72877302, 0.81276345, 0.17171958, 0.13947361]])

    """
    H = None

    if random_state is None:
        random_state = np.random.default_rng()
    elif isinstance(random_state, np.random.RandomState):
        warn(
            "Using random_state is deprecated "
            "and will raise an error in a future version. Please "
            "use seed parameter and pass a np.random.Generator",
            DeprecationWarning,
            stacklevel=2,
        )
    elif not isinstance(random_state, np.random.RandomState):
        warn(
            "Passing a seed or integer to random_state is deprecated "
            "and will raise an error in a future version. Please "
            "use seed parameter",
            DeprecationWarning,
            stacklevel=2,
        )
        random_state = np.random.RandomState(random_state)

    if seed is not None:
        if isinstance(seed, np.random.Generator):
            random_state = seed
        else:
            random_state = np.random.default_rng(seed)

    if samples is None:
        samples = n

    if criterion is not None:
        if criterion.lower() not in {
            "center",
            "c",
            "maximin",
            "m",
            "centermaximin",
            "cm",
            "correlation",
            "corr",
            "lhsmu",
        }:
            raise ValueError(f'Invalid value for "criterion": {criterion}')

    else:
        H = _lhsclassic(n, samples, random_state)

    if criterion is None:
        criterion = "center"
    if iterations is None:
        iterations = 5

    if H is None:
        if criterion.lower() in {"center", "c"}:
            H = _lhscentered(n, samples, random_state)
        elif criterion.lower() in {"maximin", "m"}:
            H = _lhsmaximin(n, samples, iterations, "maximin", random_state)
        elif criterion.lower() in {"centermaximin", "cm"}:
            H = _lhsmaximin(
                n, samples, iterations, "centermaximin", random_state
            )
        elif criterion.lower() in {"correlation", "corr"}:
            H = _lhscorrelate(n, samples, iterations, random_state)
        elif criterion.lower() in ("lhsmu"):
            # as specified by the paper. M is set to 5
            H = _lhsmu(n, samples, correlation_matrix, random_state, M=5)

    return H


################################################################################


def _lhsclassic(
    n: int,
    samples: int,
    randomstate: np.random.RandomState | np.random.Generator,
) -> np.ndarray:
    # Generate the intervals
    cut = np.linspace(0, 1, samples + 1)

    # Fill points uniformly in each interval
    u = (
        randomstate.rand(samples, n)
        if isinstance(randomstate, np.random.RandomState)
        else randomstate.random((samples, n))
    )

    a = cut[:samples]
    b = cut[1 : samples + 1]

    # Make the random pairings
    H = np.empty_like(u)

    for j in range(n):
        rdpoints = u[:, j] * (b - a) + a
        H[:, j] = rdpoints[randomstate.permutation(samples)]

    return H


################################################################################


def _lhscentered(
    n: int,
    samples: int,
    randomstate: np.random.RandomState | np.random.Generator,
) -> np.ndarray:
    # Generate the intervals
    cut = np.linspace(0, 1, samples + 1)

    # Fill points uniformly in each interval
    u = (
        randomstate.rand(samples, n)
        if isinstance(randomstate, np.random.RandomState)
        else randomstate.random((samples, n))
    )
    a = cut[:samples]
    b = cut[1 : samples + 1]
    center = (a + b) / 2

    # Make the random pairings
    H = np.zeros_like(u)
    for j in range(n):
        H[:, j] = randomstate.permutation(center)

    return H


################################################################################


def _lhsmaximin(
    n: int,
    samples: int,
    iterations: int,
    lhstype: str,
    randomstate: np.random.RandomState | np.random.Generator,
) -> np.ndarray:
    maxdist = 0

    # Maximize the minimum distance between points
    for _ in range(iterations):
        if lhstype == "maximin":
            H_candidate = _lhsclassic(n, samples, randomstate)
        else:
            H_candidate = _lhscentered(n, samples, randomstate)

        d = spatial.distance.pdist(H_candidate, "euclidean")
        if maxdist < np.min(d):
            maxdist = np.min(d)
            H = H_candidate.copy()

    return H


################################################################################


def _lhscorrelate(
    n: int,
    samples: int,
    iterations: int,
    randomstate: np.random.RandomState | np.random.Generator,
) -> np.ndarray:
    mincorr = np.inf

    # Minimize the components correlation coefficients
    for _ in range(iterations):
        # Generate a random LHS
        H_candidate = _lhsclassic(n, samples, randomstate)
        R = np.corrcoef(H_candidate.T)
        if np.max(np.abs(R[R != 1])) < mincorr:
            mincorr = np.max(np.abs(R - np.eye(R.shape[0])))
            H = H_candidate.copy()

    return H


################################################################################


def _lhsmu(
    N: int,
    samples: int | None = None,
    corr: np.ndarray = None,
    randomstate: np.random.RandomState | np.random.Generator | None = None,
    M: int = 5,
) -> np.ndarray:
    if samples is None:
        samples = N

    I = M * samples  # ruff: ignore[ambiguous-variable-name]

    rdpoints = randomstate.uniform(size=(I, N))

    dist = spatial.distance.cdist(rdpoints, rdpoints, metric="euclidean")
    D_ij = ma.masked_array(dist, mask=np.identity(I))

    index_rm = np.zeros(I - samples, dtype=int)
    i = 0
    while i < I - samples:
        order = ma.sort(D_ij, axis=1)

        avg_dist = ma.mean(order[:, 0:2], axis=1)
        min_l = ma.argmin(avg_dist)

        D_ij[min_l, :] = ma.masked
        D_ij[:, min_l] = ma.masked

        index_rm[i] = min_l
        i += 1

    rdpoints = np.delete(rdpoints, index_rm, axis=0)

    if corr is not None:
        # check if covariance matrix is valid
        if type(corr) is not np.ndarray:
            raise TypeError("corr must be a numpy array")
        if corr.ndim != 2:
            raise ValueError("corr must be 2-dimensional")
        if corr.shape[0] != corr.shape[1]:
            raise ValueError("corr must be square matrix")
        if corr.shape[0] != N:
            raise ValueError(f"corr shape {corr.shape[0]} does not match N={N}")

        norm_u = stats.norm().ppf(rdpoints)
        L = linalg.cholesky(corr, lower=True)

        norm_u = np.matmul(norm_u, L)

        H = stats.norm().cdf(norm_u)
    else:
        H = np.zeros_like(rdpoints, dtype=float)
        rank = np.argsort(rdpoints, axis=0)

        for l in range(samples):  # ruff: ignore[ambiguous-variable-name]
            low = float(l) / samples
            high = float(l + 1) / samples

            l_pos = rank == l
            H[l_pos] = randomstate.uniform(low, high, size=N)
    return H
