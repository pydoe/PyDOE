from __future__ import annotations

from typing import Literal

import numpy as np

from pydoe.optimal.criterion import (
    a_optimality,
    c_optimality,
    d_optimality,
    e_optimality,
    g_optimality,
    i_optimality,
    s_optimality,
    t_optimality,
    v_optimality,
)
from pydoe.optimal.model import build_design_matrix, build_uniform_moment_matrix


def _xtx_augmented(
    X: np.ndarray, alpha: float = 0.0, X0: np.ndarray = None
) -> np.ndarray:
    r"""
    Compute augmented information matrix:

    $$ X^T X_{\text{aug}} = X^T X + \alpha \left( \frac{X_0^T X_0}{N_0} \right) $$

    If $$\alpha = 0$$ or $$X_0$$ is None, returns plain $$X^T X$$.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Design matrix.
    alpha : float, optional
        Augmentation parameter (default is 0.0).
    X0 : ndarray, optional
        Candidate set for augmentation.

    Returns
    -------
    ndarray of shape (p, p)
        Augmented information matrix.
    """  # ruff: ignore[line-too-long]
    H = X.T @ X
    if alpha and X0 is not None and len(X0) > 0:
        N0 = X0.shape[0]
        H0 = (X0.T @ X0) / N0
        H = H + alpha * H0
    return H


def information_matrix(
    X: np.ndarray,
    *,
    normalized: bool = True,
    alpha: float = 0.0,
    X0: np.ndarray = None,
) -> np.ndarray:
    r"""
    Compute the information matrix for a design matrix,
    with optional augmentation.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Design matrix.
    normalized : bool, optional
        If True, returns $(1/n) \cdot (X^T X_{\text{aug}})$; else returns $X^T X_{\text{aug}}$.
    alpha : float, optional
        Augmentation parameter (default is 0.0).
    X0 : ndarray, optional
        Candidate set for augmentation.

    Returns
    -------
    ndarray of shape (p, p)
        Information matrix.
    """  # ruff: ignore[line-too-long]
    n = X.shape[0]
    H_aug = _xtx_augmented(X, alpha=alpha, X0=X0)
    return H_aug / n if normalized else H_aug


def criterion_value(  # ruff: ignore[too-many-return-statements]
    X: np.ndarray,
    criterion: Literal["D", "A", "I", "C", "E", "G", "V", "S", "T"],
    X0: np.ndarray = None,
    alpha: float = 0.0,
    M_moment: np.ndarray = None,
    **kwargs: dict,
) -> float:
    """
    Compute the value of a specified optimality criterion for a design matrix.

    Parameters
    ----------
    X : ndarray
        Design matrix.
    criterion : str
        Criterion type: 'D', 'A', 'I', 'C', 'E', 'G', 'V', 'S', 'T'.
    X0 : ndarray, optional
        Full candidate set for augmentation.
    alpha : float, optional
        Augmentation parameter (default is 0.0).
    M_moment : ndarray, optional
        Moment matrix for I-optimality.
    **kwargs : dict, optional
        Additional arguments for specific criteria:
        - c_vector : for C-optimality
        - test_points : for V-optimality
        - model_diff_subset : for T-optimality
        - candidates : for G-optimality

    Returns
    -------
    float
        Value of the specified criterion.

    Raises
    ------
    ValueError
        If an unknown criterion is specified.
    """
    M = information_matrix(X, normalized=True, alpha=alpha, X0=X0)
    p = X.shape[1]

    if criterion.upper() == "D":
        return d_optimality(M)
    elif criterion.upper() == "A":
        return a_optimality(M)
    elif criterion.upper() == "I":
        if M_moment is None and X0 is not None:
            M_moment = build_uniform_moment_matrix(X0)
        return i_optimality(M, M_moment)
    elif criterion.upper() == "C":
        c_vector = kwargs.get("c_vector", np.ones(p))
        return c_optimality(M, c_vector)
    elif criterion.upper() == "E":
        return e_optimality(M)
    elif criterion.upper() == "G":
        candidates = kwargs.get("candidates", X0 if X0 is not None else X)
        return g_optimality(M, candidates)
    elif criterion.upper() == "V":
        test_points = kwargs.get("test_points", X)
        return v_optimality(M, test_points)
    elif criterion.upper() == "S":
        return s_optimality(M)
    elif criterion.upper() == "T":
        model_diff = kwargs.get("model_diff_subset", np.zeros(X.shape[0]))
        return t_optimality(X, model_diff)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")


def _best_single_add(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    current: np.ndarray,
    pool: np.ndarray,
    degree: int,
    criterion: str,
    X0_model: np.ndarray | None,
    alpha: float,
    M_moment: np.ndarray | None,
) -> tuple[int, float]:
    """
    Among candidates in 'pool', find index that maximizes criterion
    if added to 'current'.

    Parameters
    ----------
    current : ndarray
        Current design points.
    pool : ndarray
        Pool of candidate points to consider adding.
    degree : int
        Polynomial degree of the model.
    criterion : str
        Optimality criterion.
    X0_model : ndarray, optional
        Candidate set for augmentation.
    alpha : float
        Augmentation parameter.
    M_moment : ndarray, optional
        Moment matrix for I-optimality.

    Returns
    -------
    best_idx : int
        Index in pool of the best candidate to add (or -1 if none improves).
    best_val : float
        Best criterion value achieved.
    """
    if len(pool) == 0:
        X_cur = build_design_matrix(current, degree)
        base = criterion_value(X_cur, criterion, X0_model, alpha, M_moment)
        return -1, base

    best_idx, best_val = -1, -np.inf
    for j in range(pool.shape[0]):
        trial = np.vstack([current, pool[j]])
        X_trial = build_design_matrix(trial, degree)
        val = criterion_value(X_trial, criterion, X0_model, alpha, M_moment)
        if val > best_val:
            best_val = val
            best_idx = j
    return best_idx, best_val


def _best_single_drop(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    current: np.ndarray,
    degree: int,
    criterion: str,
    X0_model: np.ndarray | None,
    alpha: float,
    M_moment: np.ndarray | None,
) -> tuple[int, float]:
    """
    Among points in 'current', find index whose removal gives best criterion.

    Parameters
    ----------
    current : ndarray
        Current design points.
    degree : int
        Polynomial degree of the model.
    criterion : str
        Optimality criterion.
    X0_model : ndarray, optional
        Candidate set for augmentation.
    alpha : float
        Augmentation parameter.
    M_moment : ndarray, optional
        Moment matrix for I-optimality.

    Returns
    -------
    best_idx : int
        Index in current of the best point to drop (or -1 if none improves).
    best_val : float
        Best criterion value achieved after drop.
    """
    if current.shape[0] <= 1:
        X_cur = build_design_matrix(current, degree)
        return -1, criterion_value(X_cur, criterion, X0_model, alpha, M_moment)

    best_idx, best_val = 0, -np.inf
    for i in range(current.shape[0]):
        trial = np.delete(current, i, axis=0)
        X_trial = build_design_matrix(trial, degree)
        val = criterion_value(X_trial, criterion, X0_model, alpha, M_moment)
        if val > best_val:
            best_val = val
            best_idx = i
    return best_idx, best_val
