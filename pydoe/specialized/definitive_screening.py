"""
Copyright (C) 2015 - Jacob Albrecht, Bristol-Myers Squibb
Copyright (C) 2022 - Daniele Ongari

Definitive screening designs (DSD).

A definitive screening design (Jones & Nachtsheim, 2011) is a
three-level design for continuous factors that estimates main effects
independently of two-factor interactions and can detect curvature. The
Jones & Nachtsheim (2013) extension adds two-level categorical factors.

The generalized construction in this module is adapted from
``danieleongari/definitive_screening_design`` 0.5.0. Jacob Albrecht
originally ported it from JMP to MATLAB, and Daniele Ongari subsequently
ported it to Python.

The adapted source is distributed under the BSD 3-Clause License.

References
----------
Jones, B., & Nachtsheim, C. J. (2011). A class of three-level designs
    for definitive screening in the presence of second-order effects.
    *Journal of Quality Technology*, 43(1), 1-15.
Xiao, L., Lin, D. K. J., & Bai, F. (2012). Constructing definitive
    screening designs using conference matrices. *Journal of Quality
    Technology*, 44(1), 2-8.
Jones, B., & Nachtsheim, C. J. (2013). Definitive screening designs
    with added two-level categorical factors. *Journal of Quality
    Technology*, 45(2), 121-129.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


__all__ = ["definitive_screening_design"]

DSDMethod = Literal["dsd", "orth"]


def definitive_screening_design(
    k: int,
    n_cat: int = 0,
    *,
    method: DSDMethod = "dsd",
    min_13: bool = False,
    n_fake_factors: int = 0,
) -> np.ndarray:
    r"""
    Generate a definitive screening design.

    ``k`` continuous factors occupy the first columns and use coded
    levels -1, 0, and 1. The following ``n_cat`` two-level categorical
    factors use coded levels -1 and 1.

    With the default options, calls supported by earlier PyDOE releases
    retain their exact center-first Paley design. Other factor counts use
    the next suitable conference construction. Optional fake continuous
    factors increase the number of runs but are removed from the returned
    matrix.

    Parameters
    ----------
    k : int
        Number of continuous factors. May be zero when ``n_cat`` is
        positive.
    n_cat : int, optional
        Number of two-level categorical factors. Defaults to zero.
    method : {"dsd", "orth"}, optional
        ``"dsd"`` de-aliases two-factor interactions involving
        categorical factors. ``"orth"`` constructs an orthogonal
        main-effects plan. Defaults to ``"dsd"``.
    min_13 : bool, optional
        If ``True``, augment designs with fewer than six effective
        factors so that they contain at least 13 runs. Defaults to
        ``False`` for backward compatibility.
    n_fake_factors : int, optional
        Number of temporary continuous factors used to enlarge the
        design. Their columns are omitted from the result. Defaults to
        zero.

    Returns
    -------
    ndarray of shape (n_runs, k + n_cat)
        Floating-point design matrix. Continuous columns contain -1, 0,
        and 1; categorical columns contain -1 and 1.

    Raises
    ------
    ValueError
        If a factor count is not a nonnegative integer, no real factors
        are requested, ``method`` is invalid, or ``min_13`` is not a
        boolean.

    Notes
    -----
    The generalized construction is adapted from version 0.5.0 of
    ``danieleongari/definitive_screening_design`` [4]_. Jacob Albrecht
    originally ported it from JMP to MATLAB in 2015, and Daniele Ongari
    subsequently ported it to Python in 2022. The adapted source is
    distributed under the BSD 3-Clause License.

    References
    ----------
    .. [1] Jones, B., & Nachtsheim, C. J. (2011). A class of three-level
       designs for definitive screening in the presence of second-order
       effects. *Journal of Quality Technology*, 43(1), 1-15.
    .. [2] Xiao, L., Lin, D. K. J., & Bai, F. (2012). Constructing
       definitive screening designs using conference matrices. *Journal
       of Quality Technology*, 44(1), 2-8.
    .. [3] Jones, B., & Nachtsheim, C. J. (2013). Definitive screening
       designs with added two-level categorical factors. *Journal of
       Quality Technology*, 45(2), 121-129.
    .. [4] Ongari, D. (2022). ``definitive_screening_design`` 0.5.0.
       https://github.com/danieleongari/definitive_screening_design/tree/v0.5.0

    Examples
    --------
    The historical one-argument result is unchanged:

    >>> definitive_screening_design(4)
    array([[ 0.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  1.],
           [-1.,  0.,  1., -1.],
           [-1., -1.,  0.,  1.],
           [-1.,  1., -1.,  0.],
           [ 0., -1., -1., -1.],
           [ 1.,  0., -1.,  1.],
           [ 1.,  1.,  0., -1.],
           [ 1., -1.,  1.,  0.]])

    Mixed designs return continuous columns first and categorical
    columns last:

    >>> D = definitive_screening_design(3, n_cat=2)
    >>> D.shape
    (14, 5)
    >>> np.unique(D[:, :3])
    array([-1.,  0.,  1.])
    >>> np.unique(D[:, 3:])
    array([-1.,  1.])

    ``min_13`` enlarges a small continuous-only design:

    >>> definitive_screening_design(4, min_13=True).shape
    (13, 4)
    """
    _validate_count("k", k)
    _validate_count("n_cat", n_cat)
    _validate_count("n_fake_factors", n_fake_factors)
    if k + n_cat == 0:
        raise ValueError(
            "at least one continuous or categorical factor is required"
        )
    if not isinstance(method, str) or method not in {"dsd", "orth"}:
        raise ValueError(f"method must be 'dsd' or 'orth', got {method!r}")
    if not isinstance(min_13, bool):
        raise ValueError(f"min_13 must be a boolean, got {min_13!r}")

    effective_fake_factors = n_fake_factors
    if min_13:
        effective_fake_factors = max(effective_fake_factors, 6 - (k + n_cat))

    n_continuous = k + effective_fake_factors
    n_factors = n_continuous + n_cat
    historical_paley = (
        n_cat == 0
        and effective_fake_factors == 0
        and k >= 4
        and _is_prime(k - 1)
    )
    base = _conference_design(n_factors, historical_paley=historical_paley)
    design = _add_center_runs_and_code_categories(
        base, n_continuous, n_cat, method
    )

    keep = [*range(k), *range(n_continuous, n_factors)]
    return design[:, keep] + 0.0


def _validate_count(name: str, value: int) -> None:
    """
    Validate a factor-count argument.

    Parameters
    ----------
    name : str
        Argument name for the error message.
    value : int
        Value to validate.

    Raises
    ------
    ValueError
        If ``value`` is not a nonnegative integer.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer, got {value!r}")


def _conference_design(n_factors: int, *, historical_paley: bool) -> np.ndarray:
    """
    Construct the center-free generalized DSD.

    Parameters
    ----------
    n_factors : int
        Effective number of continuous and categorical factors.
    historical_paley : bool
        Whether to use PyDOE's historical Paley orientation.

    Returns
    -------
    ndarray
        Center-free design with ``n_factors`` columns.
    """
    if n_factors in {9, 10}:
        return _DSD_10[:, :n_factors].copy()
    if n_factors in {15, 16}:
        return _DSD_16[:, :n_factors].copy()
    if n_factors in {25, 26}:
        return _dsd_26()[:, :n_factors]

    q = n_factors - 1 if n_factors % 2 == 0 else n_factors
    while q != 1 and not _is_prime(q):
        q += 2

    if historical_paley:
        C = _paley_conference_matrix(q)
    else:
        C = _generalized_paley_conference_matrix(q)
    return np.vstack([C, -C])[:, :n_factors]


def _add_center_runs_and_code_categories(
    base: np.ndarray, n_continuous: int, n_cat: int, method: DSDMethod
) -> np.ndarray:
    """
    Add center runs and convert categorical zeros to two coded levels.

    Parameters
    ----------
    base : ndarray
        Center-free generalized DSD.
    n_continuous : int
        Number of real and fake continuous-factor columns.
    n_cat : int
        Number of categorical-factor columns.
    method : {"dsd", "orth"}
        Categorical construction method.

    Returns
    -------
    ndarray
        Center-first design with coded categorical columns.
    """
    if n_cat == 0:
        return np.vstack([np.zeros((1, base.shape[1])), base])

    base = base.copy()
    half = base.shape[0] // 2
    categorical = base[:, n_continuous:]

    zero_first = categorical[:half] == 0
    zero_second = categorical[half:] == 0
    if method == "dsd":
        categorical[:half][zero_first] = 1
        categorical[half:][zero_second] = -1
    else:
        categorical[categorical == 0] = 1

    if n_cat == 1:
        n_centers = 2
        centers = np.zeros((n_centers, base.shape[1]))
        if method == "dsd":
            centers[:, n_continuous:] = np.array([[1.0], [-1.0]])
        else:
            centers[:, n_continuous:] = 1
    elif method == "dsd":
        centers = np.zeros((2, base.shape[1]))
        centers[0, n_continuous:] = -1
        centers[1, n_continuous:] = 1
    else:
        centers = np.zeros((4, base.shape[1]))
        patterns = np.array([
            [-1.0, -1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, -1.0],
        ])
        for column in range(n_cat):
            centers[:, n_continuous + column] = patterns[:, column % 4]

    return np.vstack([centers, base])


def _is_prime(n: int) -> bool:
    """
    Check whether ``n`` is a prime number.

    Parameters
    ----------
    n : int
        The number to check.

    Returns
    -------
    bool
        ``True`` if ``n`` is prime, ``False`` otherwise.
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for divisor in range(3, int(n**0.5) + 1, 2):
        if n % divisor == 0:
            return False
    return True


def _paley_conference_matrix(q: int) -> np.ndarray:
    """
    Construct a Paley conference matrix of order ``q + 1``.

    Parameters
    ----------
    q : int
        One or an odd prime number.

    Returns
    -------
    ndarray of shape (q + 1, q + 1)
        Conference matrix with zero diagonal and off-diagonal levels
        -1 and 1.
    """
    squares = {(i * i) % q for i in range(1, q)}
    chi = np.zeros(q)
    for a in range(1, q):
        chi[a] = 1.0 if a in squares else -1.0

    Q = np.array([[chi[(j - i) % q] for j in range(q)] for i in range(q)])

    n = q + 1
    C = np.zeros((n, n))
    C[0, 1:] = 1.0
    C[1:, 0] = 1.0 if q % 4 == 1 else -1.0
    C[1:, 1:] = Q
    return C


def _generalized_paley_conference_matrix(q: int) -> np.ndarray:
    """
    Construct the Paley matrix orientation used by the generalized DSD.

    Parameters
    ----------
    q : int
        One or an odd prime number.

    Returns
    -------
    ndarray of shape (q + 1, q + 1)
        Conference matrix used by the upstream generalized construction.
    """
    squares = {(i * i) % q for i in range(1, q)}
    Q = np.zeros((q, q))
    for i in range(q):
        for j in range(q):
            difference = (j - i) % q
            if difference:
                Q[i, j] = -1.0 if difference in squares else 1.0

    C = np.zeros((q + 1, q + 1))
    C[0, 1:] = 1
    C[1:, 0] = 1
    C[1:, 1:] = Q
    return C


def _dsd_26() -> np.ndarray:
    """
    Construct the efficient 26-factor center-free design.

    Returns
    -------
    ndarray of shape (52, 26)
        Center-free 26-factor generalized DSD.
    """
    q = 13
    squares = {(i * i) % q for i in range(1, q)}
    A = np.zeros((q, q))
    for i in range(q):
        for j in range(q):
            difference = (j - i) % q
            if difference:
                A[i, j] = -1.0 if difference in squares else 1.0

    starter = np.array([-1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 1], dtype=float)
    B = np.vstack([np.roll(starter, -shift) for shift in range(q)])
    C = np.block([[A, B], [B.T, -A]])
    return np.vstack([C, -C])


_DSD_10 = np.array(
    [
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, -1, -1, -1, -1, 1, 1, 1, 1],
        [1, -1, 0, -1, 1, 1, -1, -1, 1, 1],
        [1, -1, -1, 0, 1, 1, 1, 1, -1, -1],
        [1, -1, 1, 1, 0, -1, -1, 1, -1, 1],
        [1, -1, 1, 1, -1, 0, 1, -1, 1, -1],
        [1, 1, -1, 1, -1, 1, 0, -1, -1, 1],
        [1, 1, -1, 1, 1, -1, -1, 0, 1, -1],
        [1, 1, 1, -1, -1, 1, -1, 1, 0, -1],
        [1, 1, 1, -1, 1, -1, 1, -1, -1, 0],
        [0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, 0, 1, 1, 1, 1, -1, -1, -1, -1],
        [-1, 1, 0, 1, -1, -1, 1, 1, -1, -1],
        [-1, 1, 1, 0, -1, -1, -1, -1, 1, 1],
        [-1, 1, -1, -1, 0, 1, 1, -1, 1, -1],
        [-1, 1, -1, -1, 1, 0, -1, 1, -1, 1],
        [-1, -1, 1, -1, 1, -1, 0, 1, 1, -1],
        [-1, -1, 1, -1, -1, 1, 1, 0, -1, 1],
        [-1, -1, -1, 1, 1, -1, 1, -1, 0, 1],
        [-1, -1, -1, 1, -1, 1, -1, 1, 1, 0],
    ],
    dtype=float,
)

_DSD_16_HALF = np.array(
    [
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [-1, 0, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1],
        [-1, -1, 0, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1],
        [-1, -1, -1, 0, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1],
        [-1, 1, -1, -1, 0, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1],
        [-1, -1, 1, -1, -1, 0, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1],
        [-1, 1, -1, 1, -1, -1, 0, 1, 1, 1, -1, 1, -1, -1, -1, 1],
        [-1, 1, 1, -1, 1, -1, -1, 0, 1, 1, 1, -1, 1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1, 1, 1],
        [-1, 1, 1, 1, -1, 1, -1, -1, -1, 0, -1, -1, 1, -1, 1, 1],
        [-1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 0, -1, -1, 1, -1, 1],
        [-1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 0, -1, -1, 1, -1],
        [-1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 0, -1, -1, 1],
        [-1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 0, -1, -1],
        [-1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 0, -1],
        [-1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, 0],
    ],
    dtype=float,
)
_DSD_16 = np.vstack([_DSD_16_HALF, -_DSD_16_HALF])
