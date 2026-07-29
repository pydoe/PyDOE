from __future__ import annotations

from typing import Literal

import numpy as np

from pydoe.optimal.algorithms import (
    detmax,
    fedorov,
    modified_fedorov,
    sequential_dykstra,
    simple_exchange_wynn_mitchell,
)
from pydoe.optimal.efficiency import a_efficiency, d_efficiency
from pydoe.optimal.model import build_design_matrix, build_uniform_moment_matrix
from pydoe.optimal.utils import criterion_value


def optimal_design(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    candidates: np.ndarray,
    n_points: int,
    degree: int,
    criterion: Literal["D", "A", "I", "C", "E", "G", "V", "S", "T"] = "D",
    method: Literal[
        "sequential", "simple_exchange", "fedorov", "modified_fedorov", "detmax"
    ] = "sequential",
    alpha: float = 0.0,
    max_iter: int = 200,
) -> tuple[np.ndarray, dict]:
    """
    Generate an optimal experimental design using a specified
    algorithm and criterion.

    Parameters
    ----------
    candidates : ndarray of shape (N0, k)
        Candidate set (region R).
    n_points : int
        Requested design size (n >= p is recommended).
    degree : int
        Polynomial degree of the model.
    criterion : {'D', 'A', 'I'}, optional
        Optimality criterion to maximize (default is 'D').
    method : {'sequential', 'simple_exchange', 'fedorov', 'modified_fedorov',
              'detmax'}, optional. Algorithm to use for design generation
              (default is 'detmax').
    alpha : float, optional
        Augmentation parameter for information matrix (default is 0.0).
    max_iter : int, optional
        Maximum number of iterations for iterative methods (default is 200).

    Returns
    -------
    design : ndarray of shape (n_points, k)
        Selected design points.
    info : dict
        Dictionary with scores and efficiencies, including:
        - 'criterion': criterion used
        - 'method': algorithm used
        - 'alpha': augmentation parameter
        - 'score': criterion value
        - 'det_XtX': determinant of X^T X
        - 'D_eff': D-efficiency
        - 'A_eff': A-efficiency
        - 'p_columns': number of model parameters
        - 'n_runs': number of runs in the design

    Raises
    ------
    ValueError
        If an unknown method or criterion is specified.
    """
    # Choose algorithm
    if method == "sequential":
        design = sequential_dykstra(
            candidates, n_points, degree, criterion, alpha
        )
    elif method == "simple_exchange":
        design = simple_exchange_wynn_mitchell(
            candidates, n_points, degree, criterion, alpha, max_iter
        )
    elif method == "fedorov":
        design = fedorov(
            candidates, n_points, degree, criterion, alpha, max_iter
        )
    elif method == "modified_fedorov":
        design = modified_fedorov(
            candidates, n_points, degree, criterion, alpha, max_iter
        )
    elif method == "detmax":
        design = detmax(
            candidates, n_points, degree, criterion, alpha, max_iter
        )
    else:
        raise ValueError("Unknown method.")

    # Scores & efficiencies
    X = build_design_matrix(design, degree)
    X0 = build_design_matrix(candidates, degree)
    M_moment = build_uniform_moment_matrix(X0)
    score = criterion_value(X, criterion, X0, alpha, M_moment)
    info = {
        "criterion": criterion,
        "method": method,
        "alpha": alpha,
        "score": float(score),
        "det_XTX": float(max(np.linalg.det(X.T @ X), 0.0)),
        "D_eff": float(d_efficiency(X)),
        "A_eff": float(a_efficiency(X)),
        "p_columns": int(X.shape[1]),
        "n_runs": int(X.shape[0]),
    }
    return design, info
