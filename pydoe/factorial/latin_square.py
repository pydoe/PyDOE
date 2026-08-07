"""
Latin square, Graeco-Latin square, and hyper-Graeco-Latin square designs.

A Latin square of order *n* is an *n* x *n* array filled with *n* different
symbols, each occurring exactly once in each row and exactly once in each
column. Latin square designs are used to remove the effect of two nuisance
factors (rows and columns) while studying a single treatment factor with
*n* levels.

A Graeco-Latin square superimposes two orthogonal Latin squares, allowing a
third nuisance factor to be removed simultaneously. A hyper-Graeco-Latin
square extends this idea to four or more mutually orthogonal Latin squares.

References
----------
NIST/SEMATECH e-Handbook of Statistical Methods, Section 5.3.3.2 -
    "Latin square and related designs",
    https://www.itl.nist.gov/div898/handbook/pri/section3/pri333.htm
Fisher, R. A. (1935). *The Design of Experiments*. Oliver and Boyd.
"""

from __future__ import annotations

import numpy as np


__all__ = ["graeco_latin_square", "hyper_graeco_latin_square", "latin_square"]


def latin_square(n: int) -> np.ndarray:
    """
    Construct a cyclic Latin square of order ``n``.

    The square is built using the cyclic construction
    :math:`L[i, j] = (i + j) \\bmod n`, which is a valid Latin square for
    any :math:`n \\ge 2`: every row and every column is a permutation of
    the symbols :math:`0, 1, \\ldots, n - 1`.

    Parameters
    ----------
    n : int
        Order of the Latin square (number of levels), must be at least 2.

    Returns
    -------
    square : ndarray of shape (n, n)
        A Latin square with integer entries in ``range(n)``.

    Raises
    ------
    ValueError
        If ``n`` is less than 2.

    Examples
    --------
    >>> latin_square(4)
    array([[0, 1, 2, 3],
           [1, 2, 3, 0],
           [2, 3, 0, 1],
           [3, 0, 1, 2]])
    """
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")

    i = np.arange(n).reshape(-1, 1)
    j = np.arange(n).reshape(1, -1)
    return (i + j) % n


def graeco_latin_square(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct a pair of orthogonal (Graeco-)Latin squares of order ``n``.

    Two Latin squares are *orthogonal* if, when superimposed, every
    ordered pair of symbols occurs exactly once. The squares are built
    using the modular construction :math:`L_a[i, j] = (i + a j) \\bmod n`
    with :math:`a \\in \\{1, 2\\}`. For prime :math:`n`, this guarantees
    orthogonality because :math:`\\gcd(a, n) = 1` for both squares and
    :math:`\\gcd(a_1 - a_2, n) = \\gcd(1, n) = 1`.

    Parameters
    ----------
    n : int
        Order of the squares, must be a prime number greater than 2.

    Returns
    -------
    latin : ndarray of shape (n, n)
        The "Latin" square, with integer entries in ``range(n)``.
    graeco : ndarray of shape (n, n)
        The "Graeco" square, orthogonal to ``latin``, with integer
        entries in ``range(n)``.

    Examples
    --------
    >>> latin, graeco = graeco_latin_square(3)
    >>> latin
    array([[0, 1, 2],
           [1, 2, 0],
           [2, 0, 1]])
    >>> graeco
    array([[0, 2, 1],
           [1, 0, 2],
           [2, 1, 0]])

    !!! note
        Every ordered pair ``(latin[i, j], graeco[i, j])`` is unique,
        which can be verified with:

        >>> pairs = set(zip(latin.ravel().tolist(), graeco.ravel().tolist()))
        >>> len(pairs) == 9
        True
    """
    squares = hyper_graeco_latin_square(n, 2)
    return squares[0], squares[1]


def hyper_graeco_latin_square(n: int, k: int) -> np.ndarray:
    """
    Construct ``k`` mutually orthogonal Latin squares of order ``n``.

    Each square is built using the modular construction
    :math:`L_a[i, j] = (i + a j) \\bmod n` for :math:`a = 1, \\ldots, k`.
    For prime :math:`n`, every pair of these squares is orthogonal because
    :math:`\\gcd(a, n) = 1` for all :math:`a \\in \\{1, \\ldots, n - 1\\}`
    and :math:`\\gcd(a_1 - a_2, n) = 1` for any two distinct
    :math:`a_1, a_2 \\in \\{1, \\ldots, n - 1\\}`.

    A complete set of :math:`n - 1` mutually orthogonal Latin squares
    exists for prime :math:`n`, so :math:`k` must satisfy
    :math:`2 \\le k \\le n - 1`.

    Parameters
    ----------
    n : int
        Order of the squares, must be a prime number greater than 2.
    k : int
        Number of mutually orthogonal Latin squares to construct, must
        satisfy :math:`2 \\le k \\le n - 1`.

    Returns
    -------
    squares : ndarray of shape (k, n, n)
        ``k`` mutually orthogonal Latin squares, each with integer
        entries in ``range(n)``.

    Raises
    ------
    ValueError
        If ``n`` is not a prime number greater than 2, or if ``k`` is
        not between 2 and ``n - 1`` (inclusive).

    Examples
    --------
    >>> squares = hyper_graeco_latin_square(5, 3)
    >>> squares.shape
    (3, 5, 5)
    >>> squares[0]
    array([[0, 1, 2, 3, 4],
           [1, 2, 3, 4, 0],
           [2, 3, 4, 0, 1],
           [3, 4, 0, 1, 2],
           [4, 0, 1, 2, 3]])
    >>> squares[2]
    array([[0, 3, 1, 4, 2],
           [1, 4, 2, 0, 3],
           [2, 0, 3, 1, 4],
           [3, 1, 4, 2, 0],
           [4, 2, 0, 3, 1]])

    !!! note
        For :math:`k = 4` (a hyper-Graeco-Latin square), four nuisance
        factors plus the treatment factor can each have :math:`n` levels
        while remaining orthogonal to one another.
    """
    if not _is_prime(n) or n <= 2:
        raise ValueError(f"n must be a prime number greater than 2, got {n}")
    if not (2 <= k <= n - 1):
        raise ValueError(f"k must satisfy 2 <= k <= n - 1 = {n - 1}, got {k}")

    i = np.arange(n).reshape(-1, 1)
    j = np.arange(n).reshape(1, -1)
    return np.stack([(i + a * j) % n for a in range(1, k + 1)])


def _is_prime(n: int) -> bool:
    """
    Check whether ``n`` is a prime number.

    Parameters
    ----------
    n : int
        The number to check.

    Returns
    -------
    is_prime : bool
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
