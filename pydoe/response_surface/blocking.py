"""
Orthogonal blocking of central composite designs.

A central composite design (CCD) naturally separates into a factorial
portion (the corner points plus center points) and an axial portion
(the star points plus center points). When these two portions are run
in separate blocks, choosing the star-point distance ``alpha`` so that
the block effect is orthogonal to every model term keeps the block
variable from biasing the estimated factor effects.

References
----------
NIST/SEMATECH e-Handbook of Statistical Methods, Section 5.3.3.4.2 -
    "Blocking a response surface design",
    https://www.itl.nist.gov/div898/handbook/pri/section3/pri3342.htm
Box, G. E. P., Hunter, W. G., & Hunter, J. S. (2005). *Statistics for
    Experimenters* (2nd ed.). Wiley.
"""

from __future__ import annotations

import numpy as np

from pydoe.factorial.factorial import ff2n
from pydoe.response_surface.center_design import repeat_center
from pydoe.response_surface.star import star
from pydoe.response_surface.union import union


__all__ = ["block_ccdesign"]


def block_ccdesign(
    n: int, center: np.ndarray[int] | tuple[int, int] = (4, 4)
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Orthogonally-blocked circumscribed central composite design.

    The design is split into two blocks:

    1. The :math:`2^n` factorial points plus ``center[0]`` center runs.
    2. The :math:`2n` axial (star) points plus ``center[1]`` center
       runs, with the axial distance ``alpha`` chosen via
       [`star`][pydoe.star]'s ``"orthogonal"`` formula so that the
       block indicator is orthogonal to every linear and quadratic
       model term.

    Parameters
    ----------
    n : int
        Number of factors, must be at least 2.
    center : array_like of 2 ints, optional
        Number of center runs to add to the factorial block and the
        axial block, respectively. Defaults to ``(4, 4)``.

    Returns
    -------
    design : ndarray of shape (2**n + 2*n + sum(center), n)
        The combined design matrix with coded levels, factorial block
        first followed by the axial block.
    blocks : ndarray of shape (2**n + 2*n + sum(center),)
        Block label for each row: 0 for the factorial block, 1 for
        the axial block.

    Raises
    ------
    ValueError
        If ``n < 2`` or ``center`` does not have exactly 2 elements.

    Examples
    --------
    >>> design, blocks = block_ccdesign(2, center=(2, 2))
    >>> design
    array([[-1.        , -1.        ],
           [-1.        ,  1.        ],
           [ 1.        , -1.        ],
           [ 1.        ,  1.        ],
           [ 0.        ,  0.        ],
           [ 0.        ,  0.        ],
           [-1.41421356,  0.        ],
           [ 1.41421356,  0.        ],
           [ 0.        , -1.41421356],
           [ 0.        ,  1.41421356],
           [ 0.        ,  0.        ],
           [ 0.        ,  0.        ]])
    >>> blocks
    array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    """
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    if len(center) != 2:
        raise ValueError(
            f'Invalid number of values for "center" (expected 2, but got '
            f"{len(center)})"
        )

    H1 = union(ff2n(n), repeat_center(n, center[0]))
    H2, _a = star(n, alpha="orthogonal", center=center)
    H2 = union(H2, repeat_center(n, center[1]))

    design = union(H1, H2)
    blocks = np.concatenate([
        np.zeros(H1.shape[0], dtype=int),
        np.ones(H2.shape[0], dtype=int),
    ])
    return design, blocks
