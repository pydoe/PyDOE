"""
Hartley's small composite design.

A standard central composite design augments a full :math:`2^k`
factorial with axial (star) points and center runs. Hartley's small
composite design instead augments a resolution-III fractional
factorial — which still allows every main effect to be estimated free
of two-factor interactions — with star points and center runs,
substantially reducing the run count while still supporting a full
quadratic model for 4 or more factors.

References
----------
Hartley, H. O. (1959). Smallest composite designs for quadratic response
    surfaces. *Biometrics*, 15(4), 611-624.
NIST/SEMATECH e-Handbook of Statistical Methods, Section 5.3.3.6.2 -
    "Constructing response surface designs",
    https://www.itl.nist.gov/div898/handbook/pri/section3/pri336.htm
"""

from __future__ import annotations

import numpy as np

from pydoe.factorial.factorial import fracfact_by_res
from pydoe.response_surface.center_design import repeat_center
from pydoe.response_surface.union import union


__all__ = ["small_composite_design"]


def small_composite_design(
    n: int, center: np.ndarray[int] | tuple[int, int] = (4, 4)
) -> np.ndarray:
    r"""
    Generate Hartley's small composite design for ``n`` factors.

    The "cube" portion is a resolution-III fractional factorial from
    [`fracfact_by_res`][pydoe.fracfact_by_res] with :math:`n_c` runs
    (rather than the :math:`2^n` runs of a full factorial). Star
    points are placed at distance ``alpha`` along each axis, with
    :math:`\\alpha` chosen by the same orthogonal-blocking formula used
    by [`star`][pydoe.star], but with :math:`n_c` substituted for
    :math:`2^n`:

    .. math::

        \\alpha = \\left(n \\,
            \\frac{1 + n_{ao} / n_a}{1 + n_{co} / n_c}\\right)^{1/2}

    where :math:`n_a = 2n` is the number of axial points, and
    :math:`n_{co}`, :math:`n_{ao}` are the number of center runs added
    to the cube and star portions respectively (``center``).

    Parameters
    ----------
    n : int
        Number of factors, must be at least 3.
    center : array_like of 2 ints, optional
        Number of center runs to add to the cube portion and the star
        portion, respectively. Defaults to ``(4, 4)``.

    Returns
    -------
    ndarray of shape (n_c + 2*n + sum(center), n)
        Design matrix with coded levels, cube portion first followed
        by the star portion, where :math:`n_c` is the number of runs
        returned by ``fracfact_by_res(n, 3)``.

    Raises
    ------
    ValueError
        If ``n < 3`` or ``center`` does not have exactly 2 elements.

    Examples
    --------
    >>> small_composite_design(4, center=(2, 2))
    array([[-1., -1., -1.,  1.],
           [-1., -1.,  1.,  1.],
           [-1.,  1., -1., -1.],
           [-1.,  1.,  1., -1.],
           [ 1., -1., -1., -1.],
           [ 1., -1.,  1., -1.],
           [ 1.,  1., -1.,  1.],
           [ 1.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.],
           [-2.,  0.,  0.,  0.],
           [ 2.,  0.,  0.,  0.],
           [ 0., -2.,  0.,  0.],
           [ 0.,  2.,  0.,  0.],
           [ 0.,  0., -2.,  0.],
           [ 0.,  0.,  2.,  0.],
           [ 0.,  0.,  0., -2.],
           [ 0.,  0.,  0.,  2.],
           [ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.]])

    For 4 factors this needs only 20 runs, versus 30 for a standard
    central composite design built on the full :math:`2^4` factorial:

    >>> small_composite_design(4, center=(2, 2)).shape[0]
    20
    """
    if n < 3:
        raise ValueError(f"n must be at least 3, got {n}")
    if len(center) != 2:
        raise ValueError(
            f'Invalid number of values for "center" (expected 2, but got '
            f"{len(center)})"
        )

    H1 = fracfact_by_res(n, 3)
    nc = H1.shape[0]
    na = 2 * n
    nco, nao = center
    alpha = (n * (1 + nao / na) / (1 + nco / nc)) ** 0.5

    H2 = np.zeros((2 * n, n))
    for i in range(n):
        H2[2 * i : 2 * i + 2, i] = [-1, 1]
    H2 *= alpha

    H1 = union(H1, repeat_center(n, nco))
    H2 = union(H2, repeat_center(n, nao))
    return union(H1, H2)
