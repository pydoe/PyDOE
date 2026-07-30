"""
This module implements the Sobol' sequence generator for quasi-random sampling.

The Sobol' sequence is a type of low-discrepancy sequence widely used in
global optimization, numerical integration (e.g., Monte Carlo methods),
machine learning, and design of experiments. Compared to purely random
samples, Sobol' sequences exhibit better uniformity in multi-dimensional
space.

The sequence generation is based on direction numbers and bitwise operations.
Scrambling can optionally be applied to enhance uniformity and reduce
correlation artifacts. The implementation relies on SciPy's `qmc.Sobol`.

References
----------
Sobol', I. M. (1967). “Distribution of points in a cube and approximate
evaluation of integrals.” *Zh. Vych. Mat. Mat. Fiz.*, 7: 784-802 (in Russian);
*U.S.S.R. Comput. Maths. Math. Phys.*, 7: 86-112 (in English).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import qmc


__all__ = ["sobol_sequence"]


def sobol_sequence(  # ruff: ignore[too-many-arguments]
    n: int,
    d: int,
    *,
    scramble: bool = False,
    seed: int | None = None,
    bounds: np.ndarray = None,
    skip: int = 0,
    use_pow_of_2: bool = True,
) -> np.ndarray:
    """
    Generate a Sobol' sequence (quasi-random design matrix).

    The Sobol' sequence is a low-discrepancy sequence used to
    generate quasi-random samples in the unit hypercube [0, 1)^d.
    Optionally, the sequence can be scrambled for improved uniformity,
    and scaled to arbitrary bounds per dimension.

    Parameters
    ----------
    n : int
        Number of points to generate.
    d : int
        Dimension of the space (must be <= 21201).
    scramble : bool, optional
        Whether to apply Owen scrambling. Default is False.
    seed : int, optional
        Seed for the random number generator (used only when `scramble=True`).
    bounds : array_like of shape (d, 2), optional
        Bounds for each dimension. Each element must be a (min, max) pair.
        If provided, the output will be scaled accordingly.
    skip : int, optional
        Number of initial points to skip (i.e., fast-forward in the sequence).
        Default is 0.
    use_pow_of_2 : bool, optional
        If True, ensures `n` is a power of 2 for best balance and coverage.
        Non-power-of-two `n` values will be rounded **up** to the next power
        of 2.

    Returns
    -------
    design : ndarray of shape (`n`, `d`)
        Array of Sobol' points in [0, 1)^d, or scaled to `bounds` if provided.

    Raises
    ------
    ValueError
        If `bounds` shape does not match dimension `d`.
    """
    # Create sampler (common for both paths)
    sampler = qmc.Sobol(d=d, scramble=scramble, seed=seed)
    if skip > 0:
        sampler.fast_forward(skip)

    if use_pow_of_2:
        # Ensure n is power of 2 for best balance properties
        if not (n > 0 and (n & (n - 1)) == 0):
            n = 2 ** int(np.ceil(np.log2(n)))
        m = int(np.log2(n))
        samples = sampler.random_base2(m)
    else:
        # Generate exactly n samples. Suppress scipy's power-of-2 advisory:
        # callers that pass use_pow_of_2=False (e.g. saltelli_sampling) have
        # already validated n and issued their own user-facing warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            samples = sampler.random(n)

    if bounds is not None:
        bounds = np.asarray(bounds)
        if bounds.shape != (d, 2):
            raise ValueError(
                f"`bounds` must be a (d, 2) array, got shape {bounds.shape}"
            )
        samples = qmc.scale(samples, bounds[:, 0], bounds[:, 1])

    return samples
