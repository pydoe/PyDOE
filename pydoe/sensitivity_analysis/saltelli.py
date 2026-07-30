"""
This module implements Saltelli's sampling scheme based on Sobol' sequences
for global sensitivity analysis. It enables estimation of first-order,
total-order, and (second-order optional) Sobol' sensitivity indices. The
implementation relies on a custom Sobol' sequence generator based on
`scipy.stats.qmc.Sobol`.

Compared to random or Latin Hypercube sampling, this method provides better
convergence for variance-based sensitivity analysis using quasi-random
low-discrepancy sequences.

References
----------
1. Sobol', I.M., 2001.
   Global sensitivity indices for nonlinear mathematical models and
   their Monte Carlo estimates.
   Mathematics and Computers in Simulation,
   The Second IMACS Seminar on Monte Carlo Methods 55, 271-280.
   https://doi.org/10.1016/S0378-4754(00)00270-6

2. Saltelli, A., 2002.
   Making best use of model evaluations to compute sensitivity indices.
   Computer Physics Communications, 145(2), 280-297.
   https://doi.org/10.1016/S0010-4655(02)00280-1

3. Campolongo, F., Saltelli, A., Cariboni, J., 2011.
   From screening to quantitative sensitivity analysis. A unified approach.
   Computer Physics Communications 182, 978-988.
   https://doi.org/10.1016/j.cpc.2010.12.039

4. Owen, A. B., 2020.
   On dropping the first Sobol' point.
   arXiv:2008.08051 [cs, math, stat].
   http://arxiv.org/abs/2008.08051
   (Accessed: 20 April 2021)
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from pydoe.space_filling.quasi_random.sobol import sobol_sequence


__all__ = ["saltelli_sampling"]


def saltelli_sampling(  # ruff: ignore[too-many-arguments]
    num_vars: int,
    N: int,
    *,
    calc_second_order: bool = True,
    skip_values: int | None = None,
    scramble: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate Saltelli samples using Sobol' sequences for sensitivity analysis.

    Parameters
    ----------
    num_vars : int
        Number of input variables (dimensions).
    N : int
        Base sample size (ideally a power of 2).
    calc_second_order : bool, optional
        If True, include second-order interaction terms. Default is True.
    skip_values : int, optional
        Number of Sobol' points to skip. If None, set automatically.
    scramble : bool, optional
        Whether to use scrambling for Sobol' sequence. Default is False.
    seed : int, optional
        Random seed (only used if scramble=True).

    Returns
    -------
    np.ndarray
        Matrix of shape (N * (2 * num_vars + 2), num_vars)
        if calc_second_order=True, or (N * (num_vars + 2), num_vars)
        otherwise. Contains Saltelli samples in [0, 1].
    """
    D = num_vars

    if not ((N & (N - 1)) == 0 and N != 0):
        warnings.warn(
            f"N = {N} is not a power of 2. "
            + "This may affect Sobol sequence convergence.",
            stacklevel=2,
        )

    if skip_values is None:
        skip_values = max(2 ** math.ceil(math.log2(N)), 16)

    # Generate base Sobol samples: shape (N + skip_values, 2*D)
    base = sobol_sequence(
        n=N + skip_values,
        d=2 * D,
        scramble=scramble,
        seed=seed,
        skip=0,
        use_pow_of_2=False,
    )

    if calc_second_order:
        total_samples = N * (2 * D + 2)
    else:
        total_samples = N * (D + 2)

    saltelli_matrix = np.zeros((total_samples, D))
    idx = 0

    for i in range(skip_values, skip_values + N):
        A = base[i, :D]
        B = base[i, D : 2 * D]

        # Matrix A
        saltelli_matrix[idx] = A
        idx += 1

        # Cross A_Bi
        for j in range(D):
            C = A.copy()
            C[j] = B[j]
            saltelli_matrix[idx] = C
            idx += 1

        # Cross B_Ai (only if calc_second_order)
        if calc_second_order:
            for j in range(D):
                C = B.copy()
                C[j] = A[j]
                saltelli_matrix[idx] = C
                idx += 1

        # Matrix B
        saltelli_matrix[idx] = B
        idx += 1

    return saltelli_matrix
