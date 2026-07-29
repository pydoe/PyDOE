r"""
Design matrix formation for Optimal Experimental Design (OED).

This module provides functionality to build design matrices
for polynomial models of various degrees, which are essential
for optimal design of experiments.

References:
    - https://en.wikipedia.org/wiki/Design_matrix

The design matrix encodes the relationship between model parameters and input
variables for regression and optimal design.

Model formation:
    - Linear: $y = \beta_0 + \sum_{i=1}^k \beta_i x_i$

    - Quadratic: $y = \beta_0 + \sum_{i=1}^k \beta_i x_i + \sum_{i=1}^k \beta_{ii} x_i^2 + \sum_{i<j} \beta_{ij} x_i x_j$

where $k$ is the number of factors, $x_i$ are the input variables, and $\beta$
are the model parameters.

For a linear model with 2 factors x1, x2:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2$$

The corresponding design matrix X:
$$X =
\begin{bmatrix}
1 & x_1 & x_2 \\
1 & -1 & -1 \\
1 & -1 & 1 \\
1 & 1 & -1 \\
1 & 1 & 1 \\
\end{bmatrix}$$
For a quadratic model with 2 factors:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_{11} x_1^2 + \beta_{22} x_2^2 + \beta_{12} x_1 x_2$$

The corresponding quadratic design matrix X:
$$X =
\begin{bmatrix}
1 & x_1 & x_2 & x_1^2 & x_2^2 & x_1 x_2 \\
1 & -1 & -1 & 1 & 1 & 1 \\
1 & -1 & 1 & 1 & 1 & -1 \\
1 & 1 & -1 & 1 & 1 & -1 \\
1 & 1 & 1 & 1 & 1 & 1 \\
\end{bmatrix}$$
"""  # ruff: ignore[line-too-long]

from __future__ import annotations

import itertools

import numpy as np


def build_design_matrix(candidates: np.ndarray, degree: int) -> np.ndarray:
    r"""
    Build design matrix X from candidate points for a polynomial model
    of given degree.

    Parameters
    ----------
    candidates : ndarray of shape (n_points, n_factors)
        Candidate points in the design space.
    degree : int
        Polynomial degree (1=linear, 2=quadratic, etc.).

    Returns
    -------
    X : ndarray of shape (n_points, n_parameters)
        Design matrix with polynomial terms.

    Raises
    ------
    ValueError
        If degree is less than 1.

    Notes
    -----
    The design matrix encodes the relationship between model parameters
    and input variables.

    For the **linear** case ($\text{degree}=1$), the model is:

        $y = \beta_0 + \sum_{i=1}^k \beta_i x_i$

    For the **quadratic** case ($\text{degree}=2$), the model is:

        $y = \beta_0 + \sum_{i=1}^k \beta_i x_i + \sum_{i=1}^k \beta_{ii} x_i^2 + \sum_{i<j} \beta_{ij} x_i x_j$

    where $k$ is the number of factors, $x_i$ are the input variables, and $\beta$ are
    the model parameters.
    """  # ruff: ignore[line-too-long]
    candidates = np.asarray(candidates, dtype=float)
    n_points, n_factors = candidates.shape

    if degree < 1:
        raise ValueError("Degree must be at least 1")

    # Start with intercept term
    X_cols = [np.ones(n_points)]

    # Linear terms
    for i in range(n_factors):
        X_cols.append(candidates[:, i])

    if degree >= 2:
        # Quadratic terms (pure squares)
        for i in range(n_factors):
            X_cols.append(candidates[:, i] ** 2)

        # Interaction terms (cross products)
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                X_cols.append(candidates[:, i] * candidates[:, j])

    # Higher order terms if needed
    if degree >= 3:
        # Cubic terms and higher order interactions
        for d in range(3, degree + 1):
            for combo in itertools.combinations_with_replacement(
                range(n_factors), d
            ):
                term = np.ones(n_points)
                for factor_idx in combo:
                    term *= candidates[:, factor_idx]
                X_cols.append(term)

    return np.column_stack(X_cols)


def build_uniform_moment_matrix(X0: np.ndarray) -> np.ndarray:
    """
    Compute the moment matrix for I-optimality under uniform
    measure over the candidate region.

    Parameters
    ----------
    X0 : ndarray of shape (N0, p)
        Design matrix for the candidate region.

    Returns
    -------
    ndarray of shape (p, p)
        Moment matrix.
    """
    N0 = X0.shape[0]
    return (X0.T @ X0) / max(N0, 1)


def generate_candidate_set(
    n_factors: int,
    bounds: tuple | list | None = None,
    n_levels: int = 3,
    grid_type: str = "full_factorial",
) -> np.ndarray:
    """
    Generate candidate set for optimal design.

    Parameters
    ----------
    n_factors : int
        Number of factors/variables.
    bounds : tuple or list of tuples, optional
        Bounds for each factor. If single tuple, applies to all factors.
        Default is (-1, 1) for all factors.
    n_levels : int, optional
        Number of levels per factor for grid (default is 3).
    grid_type : {'full_factorial', 'uniform_random'}, optional
        Type of grid to generate (default is 'full_factorial').

    Returns
    -------
    candidates : ndarray of shape (n_candidates, n_factors)
        Candidate points for the design.

    Raises
    ------
    ValueError
        If bounds are not compatible with n_factors or grid_type is unknown.
    """
    if bounds is None:
        bounds = [(-1, 1)] * n_factors
    elif isinstance(bounds, tuple) and len(bounds) == 2:
        bounds = [bounds] * n_factors
    elif len(bounds) != n_factors:
        raise ValueError(f"bounds must have length {n_factors}")

    if grid_type == "full_factorial":
        # Generate full factorial grid
        levels = []
        for low, high in bounds:
            levels.append(np.linspace(low, high, n_levels))

        candidates = np.array(list(itertools.product(*levels)))

    elif grid_type == "uniform_random":
        # Generate random uniform points
        n_candidates = n_levels**n_factors
        candidates = np.random.uniform(size=(n_candidates, n_factors))

        # Scale to bounds
        for i, (low, high) in enumerate(bounds):
            candidates[:, i] = candidates[:, i] * (high - low) + low
    else:
        raise ValueError(f"Unknown grid_type: {grid_type}")

    return candidates
