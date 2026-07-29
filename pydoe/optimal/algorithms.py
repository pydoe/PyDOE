"""
Optimal Experimental Design (OED)
=================================

Algorithms for constructing optimal experimental designs.

This module implements several classical algorithms for generating
D-optimal and other optimal designs from a candidate set. Each algorithm
iteratively selects or exchanges points to optimize a chosen criterion
(e.g., D-, A-, I-optimality).

References:
    - Atkinson, A. C., Donev, A. N., & Tobias, R. D. (2007).
    "Optimum Experimental Designs, with SAS." Oxford University Press.
    - Montgomery D. C. (2017). "Design and Analysis of Experiments." Wiley.
    - Mitchell, T. J. (1974). "An Algorithm for the Construction of 'D-Optimal'
    Experimental Designs." Technometrics, 16(2), 203-210.
    - Wynn, H. P. (1972). "The Sequential Generation of D-Optimal Experimental
    Designs." Annals of Mathematical Statistics, 41(5), 1655-1664.
    - Fedorov, V. V. (1972). "Theory of Optimal Experiments." Academic Press.

Algorithms
----------
Sequential Dykstra Algorithm
    The Sequential Dykstra algorithm is a method used for constructing D-optimal
    designs. It operates by iteratively selecting points from a candidate set to
    maximize the determinant of the information matrix, which corresponds to
    minimizing the variance of the estimated model parameters. This algorithm is
    known for its speed but may sometimes converge to local optima.

Simple Exchange (Wynn-Mitchell) Algorithm
    The Simple Exchange algorithm, attributed to Mitchell and Miller (1970) and
    Wynn (1972), starts with an initial design and iteratively exchanges one
    point at a time. In each iteration, the algorithm removes a point that
    contributes least to the chosen optimality criterion and adds a new point
    from the candidate set that improves the criterion. This process continues
    until no further improvements can be made.

Fedorov Algorithm
    The Fedorov algorithm (1972) is a simultaneous exchange method. In each
    iteration, the algorithm evaluates all possible exchanges between points
    in the current design and points in the candidate set. It then selects the
    exchange that most improves the optimality criterion. This approach can be
    computationally intensive but is effective for finding globally optimal
    designs.

Modified Fedorov Algorithm
    The Modified Fedorov algorithm is a variant of the Fedorov algorithm that
    allows for the inclusion of pre-specified points in the design. This
    modification ensures that certain points, which may be of particular
    interest or importance, are included in the final design while still
    optimizing the overall efficiency. It is particularly useful when some
    experimental runs are mandatory.

DETMAX Algorithm
    The DETMAX algorithm, developed by Mitchell (1974), is designed to construct
    D-optimal designs by maximizing the determinant of the information matrix.
    It starts with an initial design and iteratively adds or removes points to
    improve the determinant. This method is widely used due to its effectiveness
    in producing highly efficient designs.
"""

from typing import Literal

import numpy as np

from pydoe.optimal.model import build_design_matrix, build_uniform_moment_matrix
from pydoe.optimal.utils import (
    _best_single_add,
    _best_single_drop,
    criterion_value,
)


# ------------------- Algorithms -----------------


def sequential_dykstra(
    candidates: np.ndarray,
    n_points: int,
    degree: int,
    criterion: Literal["D", "A", "I", "C", "E", "G", "V", "S", "T"] = "D",
    alpha: float = 0.0,
) -> np.ndarray:
    """
    Construct an optimal design using the sequential (Dykstra) greedy algorithm.

    Parameters
    ----------
    candidates : ndarray of shape (N, k)
        Candidate points in the design space.
    n_points : int
        Number of points to select for the design.
    degree : int
        Polynomial degree of the model.
    criterion : {'D', 'A', 'I', 'C', 'E', 'G', 'V', 'S', 'T'}, optional
        Optimality criterion to maximize (default is 'D').
    alpha : float, optional
        Augmentation parameter for information matrix (default is 0.0).

    Returns
    -------
    design : ndarray of shape (n_points, k)
        Selected design points.

    Raises
    ------
    ValueError
        If n_points is less than 1.
    """
    candidates = np.asarray(candidates, dtype=float)
    if n_points < 1:
        raise ValueError("n_points must be at least 1")
    # Precompute for I- and A-opt
    X0 = build_design_matrix(candidates, degree)
    M_moment = build_uniform_moment_matrix(X0)

    design = np.empty((0, candidates.shape[1]), dtype=float)
    remaining = candidates.copy()

    for _ in range(n_points):
        j_best, _ = _best_single_add(
            design, remaining, degree, criterion, X0, alpha, M_moment
        )
        j_best = max(j_best, 0)
        design = np.vstack([design, remaining[j_best]])
        remaining = np.delete(remaining, j_best, axis=0)

    return design


def simple_exchange_wynn_mitchell(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    candidates: np.ndarray,
    n_points: int,
    degree: int,
    criterion: Literal["D", "A", "I", "C", "E", "G", "V", "S", "T"] = "D",
    alpha: float = 0.0,
    max_iter: int = 200,
) -> np.ndarray:
    """
    Construct an optimal design using the simple exchange (Wynn-Mitchell)
    algorithm.

    Parameters
    ----------
    candidates : ndarray of shape (N, k)
        Candidate points in the design space.
    n_points : int
        Number of points to select for the design.
    degree : int
        Polynomial degree of the model.
    criterion : {'D', 'A', 'I', 'C', 'E', 'G', 'V', 'S', 'T'}, optional
        Optimality criterion to maximize (default is 'D').
    alpha : float, optional
        Augmentation parameter for information matrix (default is 0.0).
    max_iter : int, optional
        Maximum number of iterations (default is 200).

    Returns
    -------
    design : ndarray of shape (n_points, k)
        Selected design points.
    """
    candidates = np.asarray(candidates, dtype=float)
    X0 = build_design_matrix(candidates, degree)
    M_moment = build_uniform_moment_matrix(X0)

    design = sequential_dykstra(candidates, n_points, degree, criterion, alpha)

    def score(des: np.ndarray) -> float:
        X = build_design_matrix(des, degree)
        return criterion_value(X, criterion, X0, alpha, M_moment)

    best_score = score(design)

    for _ in range(max_iter):
        # Drop
        drop_idx, _ = _best_single_drop(
            design, degree, criterion, X0, alpha, M_moment
        )
        # Remove from pool those already in design except the one
        # we plan to drop. Using exact match; for numeric safety,
        # use all rows then filter after equality check
        design_wo = np.delete(design, drop_idx, axis=0)

        # Add
        mask = np.ones(len(candidates), dtype=bool)
        # Mark rows equal to rows in design_wo as not
        # available (basic filtering)
        for r in design_wo:
            eq = np.all(np.isclose(candidates, r, atol=1e-12), axis=1)
            mask &= ~eq
        pool_avail = candidates[mask]

        add_idx, cand_score = _best_single_add(
            design_wo, pool_avail, degree, criterion, X0, alpha, M_moment
        )

        if add_idx >= 0 and cand_score > best_score + 1e-12:
            design = np.vstack([design_wo, pool_avail[add_idx]])
            best_score = cand_score
        else:
            break

    return design


def fedorov(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    candidates: np.ndarray,
    n_points: int,
    degree: int,
    criterion: Literal["D", "A", "I", "C", "E", "G", "V", "S", "T"] = "D",
    alpha: float = 0.0,
    max_iter: int = 200,
) -> np.ndarray:
    """
    Construct an optimal design using the Fedorov exchange algorithm.

    Parameters
    ----------
    candidates : ndarray of shape (N, k)
        Candidate points in the design space.
    n_points : int
        Number of points to select for the design.
    degree : int
        Polynomial degree of the model.
    criterion : {'D', 'A', 'I', 'C', 'E', 'G', 'V', 'S', 'T'}, optional
        Optimality criterion to maximize (default is 'D').
    alpha : float, optional
        Augmentation parameter for information matrix (default is 0.0).
    max_iter : int, optional
        Maximum number of iterations (default is 200).

    Returns
    -------
    design : ndarray of shape (n_points, k)
        Selected design points.
    """
    candidates = np.asarray(candidates, dtype=float)
    X0 = build_design_matrix(candidates, degree)
    M_moment = build_uniform_moment_matrix(X0)

    design = sequential_dykstra(candidates, n_points, degree, criterion, alpha)

    def score(des: np.ndarray) -> float:
        X = build_design_matrix(des, degree)
        return criterion_value(X, criterion, X0, alpha, M_moment)

    best = score(design)

    for _ in range(max_iter):
        improved = False
        # Build availability mask
        mask = np.ones(len(candidates), dtype=bool)
        for r in design:
            eq = np.all(np.isclose(candidates, r, atol=1e-12), axis=1)
            mask &= ~eq
        pool = candidates[mask]

        # evaluate all swaps (i in design, j in pool)
        best_i, best_j, best_val = -1, -1, best
        for i in range(design.shape[0]):
            for j in range(pool.shape[0]):
                trial = design.copy()
                trial[i] = pool[j]
                s = score(trial)
                if s > best_val + 1e-12:
                    best_i, best_j, best_val = i, j, s

        if best_i >= 0:
            design[best_i] = pool[best_j]
            best = best_val
            improved = True

        if not improved:
            break

    return design


def modified_fedorov(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    candidates: np.ndarray,
    n_points: int,
    degree: int,
    criterion: Literal["D", "A", "I", "C", "E", "G", "V", "S", "T"] = "D",
    alpha: float = 0.0,
    max_iter: int = 100,
) -> np.ndarray:
    """
    Construct an optimal design using the modified Fedorov algorithm.

    Parameters
    ----------
    candidates : ndarray of shape (N, k)
        Candidate points in the design space.
    n_points : int
        Number of points to select for the design.
    degree : int
        Polynomial degree of the model.
    criterion : {'D', 'A', 'I', 'C', 'E', 'G', 'V', 'S', 'T'}, optional
        Optimality criterion to maximize (default is 'D').
    alpha : float, optional
        Augmentation parameter for information matrix (default is 0.0).
    max_iter : int, optional
        Maximum number of iterations (default is 100).

    Returns
    -------
    design : ndarray of shape (n_points, k)
        Selected design points.
    """
    candidates = np.asarray(candidates, dtype=float)
    X0 = build_design_matrix(candidates, degree)
    M_moment = build_uniform_moment_matrix(X0)

    design = sequential_dykstra(candidates, n_points, degree, criterion, alpha)

    def score(des: np.ndarray) -> float:
        X = build_design_matrix(des, degree)
        return criterion_value(X, criterion, X0, alpha, M_moment)

    for _ in range(max_iter):
        base = score(design)
        any_change = False

        # Build pool availability once (exclude exact duplicates of current)
        mask_all = np.ones(len(candidates), dtype=bool)
        for r in design:
            mask_all &= ~np.all(np.isclose(candidates, r, atol=1e-12), axis=1)
        pool = candidates[mask_all]

        # Try replace each point with best candidate for that position
        new_design = design.copy()
        for i in range(design.shape[0]):
            best_val = base
            best_row = None
            for j in range(pool.shape[0]):
                trial = new_design.copy()
                trial[i] = pool[j]
                s = score(trial)
                if s > best_val + 1e-12:
                    best_val = s
                    best_row = pool[j]
            if best_row is not None:
                new_design[i] = best_row
                base = best_val
                any_change = True

        design = new_design
        if not any_change:
            break

    return design


def detmax(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    candidates: np.ndarray,
    n_points: int,
    degree: int,
    criterion: Literal["D", "A", "I", "C", "E", "G", "V", "S", "T"] = "D",
    alpha: float = 0.0,
    max_iter: int = 100,
) -> np.ndarray:
    """
    Construct an optimal design using the DETMAX algorithm
    (exchange with excursions).

    Parameters
    ----------
    candidates : ndarray of shape (N, k)
        Candidate points in the design space.
    n_points : int
        Number of points to select for the design.
    degree : int
        Polynomial degree of the model.
    criterion : {'D', 'A', 'I', 'C', 'E', 'G', 'V', 'S', 'T'}, optional
        Optimality criterion to maximize (default is 'D').
    alpha : float, optional
        Augmentation parameter for information matrix (default is 0.0).
    max_iter : int, optional
        Maximum number of iterations (default is 100).

    Returns
    -------
    design : ndarray of shape (n_points, k)
        Selected design points.
    """
    candidates = np.asarray(candidates, dtype=float)
    X0 = build_design_matrix(candidates, degree)
    M_moment = build_uniform_moment_matrix(X0)

    design = simple_exchange_wynn_mitchell(
        candidates, n_points, degree, criterion, alpha
    )

    def score(des: np.ndarray) -> float:
        X = build_design_matrix(des, degree)
        return criterion_value(X, criterion, X0, alpha, M_moment)

    best = score(design)

    for _ in range(max_iter):
        # Try a Fedorov-style single swap first
        improved = False
        mask = np.ones(len(candidates), dtype=bool)
        for r in design:
            mask &= ~np.all(np.isclose(candidates, r, atol=1e-12), axis=1)
        pool = candidates[mask]

        best_i, best_j, best_val = -1, -1, best
        for i in range(design.shape[0]):
            for j in range(pool.shape[0]):
                trial = design.copy()
                trial[i] = pool[j]
                s = score(trial)
                if s > best_val + 1e-12:
                    best_i, best_j, best_val = i, j, s

        if best_i >= 0:
            design[best_i] = pool[best_j]
            best = best_val
            improved = True

        if improved:
            continue

        # Excursion: add best extra, then drop worst
        add_idx, _add_score = _best_single_add(
            design, pool, degree, criterion, X0, alpha, M_moment
        )
        if add_idx < 0:
            break  # nothing to add

        expanded = np.vstack([design, pool[add_idx]])
        # Drop worst from expanded
        drop_idx, final_score = _best_single_drop(
            expanded, degree, criterion, X0, alpha, M_moment
        )

        if final_score > best + 1e-12:
            design = np.delete(expanded, drop_idx, axis=0)
            best = final_score
            improved = True
        else:
            break

    return design
