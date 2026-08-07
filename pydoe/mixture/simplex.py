"""
Mixture designs for experiments where factors are proportions summing to 1.

Simplex-lattice and simplex-centroid designs are the two canonical classes
of mixture experiment designs described in NIST Handbook Section 5.6.

Both designs live on the q-component simplex where all component proportions
are non-negative and sum to 1.

References
----------
Cornell, J. A. (2002). *Experiments with Mixtures: Designs, Models, and the
    Analysis of Mixture Data* (3rd ed.). Wiley.
Scheffé, H. (1958). Experiments with mixtures. *Journal of the Royal
    Statistical Society B*, 20(2), 344-360.
NIST/SEMATECH e-Handbook of Statistical Methods, Sections 5.6.1 and 5.6.2,
    https://www.itl.nist.gov/div898/handbook/pri/section6/pri61.htm
"""

from __future__ import annotations

from itertools import combinations, combinations_with_replacement
from math import comb

import numpy as np


__all__ = ["simplex_centroid_design", "simplex_lattice_design"]


def simplex_lattice_design(q: int, m: int) -> np.ndarray:
    r"""
    Generate a simplex-lattice design for *q* mixture components of degree *m*.

    A :math:`\{q, m\}` simplex-lattice design places design points at all
    lattice points of the *q*-component simplex at resolution :math:`1/m`.
    Each component takes values in :math:`\{0, 1/m, 2/m, \ldots, 1\}`,
    subject to the mixture constraint :math:`\sum_{i=1}^q x_i = 1`.

    The total number of design points is

    .. math::

        N = \binom{q + m - 1}{m} = \binom{q + m - 1}{q - 1}.

    Special cases:

    - *m* = 1: the *q* pure-component vertices of the simplex
      (identity matrix).
    - *m* = 2: vertices plus all binary blends at the midpoints of edges.
    - *m* → ∞: approaches a continuous uniform grid on the simplex.

    Parameters
    ----------
    q : int
        Number of mixture components. Must be at least 2.
    m : int
        Lattice degree (order). Must be at least 1. Higher *m* produces
        more points with finer spacing.

    Returns
    -------
    ndarray of shape ``(comb(q + m - 1, m), q)``
        Design matrix. Each row is a mixture proportion vector with all
        entries in :math:`[0, 1]` summing to 1. Rows are ordered
        lexicographically by component index (higher-indexed components
        cycle fastest).

    Raises
    ------
    ValueError
        If ``q < 2`` or ``m < 1``.

    Examples
    --------
    Two-component lattice of degree 3 — four evenly-spaced blends:

    >>> simplex_lattice_design(2, 3)
    array([[1.        , 0.        ],
           [0.66666667, 0.33333333],
           [0.33333333, 0.66666667],
           [0.        , 1.        ]])

    Three-component lattice of degree 2 — six points (vertices and edge
    midpoints):

    >>> simplex_lattice_design(3, 2)
    array([[1. , 0. , 0. ],
           [0.5, 0.5, 0. ],
           [0.5, 0. , 0.5],
           [0. , 1. , 0. ],
           [0. , 0.5, 0.5],
           [0. , 0. , 1. ]])

    Shape follows the stars-and-bars formula:

    >>> simplex_lattice_design(3, 2).shape
    (6, 3)
    >>> simplex_lattice_design(4, 3).shape
    (20, 4)
    """
    if q < 2:
        raise ValueError(f"q must be at least 2, got {q}")
    if m < 1:
        raise ValueError(f"m must be at least 1, got {m}")

    n = comb(q + m - 1, m)
    out = np.empty((n, q))
    for row_idx, combo in enumerate(combinations_with_replacement(range(q), m)):
        counts = [0] * q
        for idx in combo:
            counts[idx] += 1
        out[row_idx] = [c / m for c in counts]
    return out


def simplex_centroid_design(q: int) -> np.ndarray:
    r"""
    Generate a simplex-centroid design for *q* mixture components.

    A simplex-centroid design consists of the centroids of every non-empty
    subset of components.  For a subset :math:`S \subseteq \{1,\ldots,q\}`
    of size :math:`s`, the corresponding design point sets each component
    in :math:`S` to :math:`1/s` and all others to 0.

    The design contains :math:`2^q - 1` points, ordered by increasing
    subset size:

    - *s* = 1: the *q* pure-component vertices.
    - *s* = 2: the :math:`\binom{q}{2}` binary edge midpoints.
    - ...
    - *s* = *q*: the single overall centroid :math:`(1/q, \ldots, 1/q)`.

    Parameters
    ----------
    q : int
        Number of mixture components. Must be at least 2.

    Returns
    -------
    ndarray of shape ``(2**q - 1, q)``
        Design matrix. Each row is a mixture proportion vector with all
        entries in :math:`[0, 1]` summing to 1. Rows are ordered by
        increasing subset size, with subsets of the same size listed in
        lexicographic order.

    Raises
    ------
    ValueError
        If ``q < 2``.

    Examples
    --------
    Three-component simplex-centroid — 7 points:

    >>> simplex_centroid_design(3)
    array([[1.        , 0.        , 0.        ],
           [0.        , 1.        , 0.        ],
           [0.        , 0.        , 1.        ],
           [0.5       , 0.5       , 0.        ],
           [0.5       , 0.        , 0.5       ],
           [0.        , 0.5       , 0.5       ],
           [0.33333333, 0.33333333, 0.33333333]])

    Shape is always :math:`2^q - 1` rows:

    >>> simplex_centroid_design(3).shape
    (7, 3)
    >>> simplex_centroid_design(4).shape
    (15, 4)
    """
    if q < 2:
        raise ValueError(f"q must be at least 2, got {q}")

    n = 2**q - 1
    out = np.zeros((n, q))
    row_idx = 0
    for s in range(1, q + 1):
        for subset in combinations(range(q), s):
            out[row_idx, list(subset)] = 1.0 / s
            row_idx += 1
    return out
