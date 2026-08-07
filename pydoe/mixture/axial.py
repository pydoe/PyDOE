"""
Axial (mixture screening) designs for mixture experiments.

Axial designs place one design point at the overall centroid of the
simplex and one additional point along each "axis" running from the
centroid towards each pure-component vertex. They are useful for
screening which mixture components have the largest effect on the
response, since each axial point perturbs a single component relative
to the centroid blend.

References
----------
Cornell, J. A. (2002). *Experiments with Mixtures: Designs, Models, and the
    Analysis of Mixture Data* (3rd ed.). Wiley.
NIST/SEMATECH e-Handbook of Statistical Methods, Section 5.6.3 -
    "Screening Designs",
    https://www.itl.nist.gov/div898/handbook/pri/section6/pri63.htm
"""

from __future__ import annotations

import numpy as np


__all__ = ["mixture_axial_design"]


def mixture_axial_design(q: int, delta: float = 0.5) -> np.ndarray:
    r"""
    Generate an axial (screening) design for *q* mixture components.

    The design consists of the overall centroid
    :math:`x_i = 1/q \\ \\forall i` plus, for each component *i*, one
    axial point obtained by moving a fraction ``delta`` of the way from
    the centroid towards the pure-component vertex :math:`e_i`:

    .. math::

        x^{(i)} = (1 - \\delta) \\, c + \\delta \\, e_i

    where :math:`c` is the centroid and :math:`e_i` is the *i*-th unit
    vector. Every row sums to 1 and has non-negative entries for any
    :math:`0 < \\delta \\le 1`, so all points lie on the simplex.

    Parameters
    ----------
    q : int
        Number of mixture components, must be at least 2.
    delta : float, optional
        Fraction of the distance from the centroid to each vertex,
        must satisfy :math:`0 < \\delta \\le 1`. Defaults to 0.5.

    Returns
    -------
    ndarray of shape ``(q + 1, q)``
        Design matrix. The first row is the overall centroid; row
        ``i + 1`` is the axial point for component *i*. Every row sums
        to 1 with all entries in :math:`[0, 1]`.

    Raises
    ------
    ValueError
        If ``q < 2`` or ``delta`` is not in :math:`(0, 1]`.

    Examples
    --------
    >>> mixture_axial_design(3)
    array([[0.33333333, 0.33333333, 0.33333333],
           [0.66666667, 0.16666667, 0.16666667],
           [0.16666667, 0.66666667, 0.16666667],
           [0.16666667, 0.16666667, 0.66666667]])

    Each row sums to 1:

    >>> np.allclose(mixture_axial_design(4, delta=0.25).sum(axis=1), 1.0)
    True
    """
    if q < 2:
        raise ValueError(f"q must be at least 2, got {q}")
    if not (0 < delta <= 1):
        raise ValueError(f"delta must satisfy 0 < delta <= 1, got {delta}")

    centroid = np.full(q, 1.0 / q)
    out = np.empty((q + 1, q))
    out[0] = centroid
    for i in range(q):
        vertex = np.zeros(q)
        vertex[i] = 1.0
        out[i + 1] = (1 - delta) * centroid + delta * vertex
    return out
