"""
John's 3/4 fractional factorial designs.

Three-quarter (3/4) designs are two-level factorial designs that use exactly
3/4 of the runs of the full 2^k factorial. They were introduced by Professor
Peter W. M. John of the University of Texas.

Construction (semifoldover approach):

1. Take the 2^(k-1) half-fraction defined by the generator
   X_k = X_1 * X_2 * ... * X_{k-1}  (highest-resolution defining word).
2. From that half-fraction, select the 2^(k-2) runs where factor *fold_on*
   is at its low level (-1).
3. Flip the sign of *fold_on* in those runs (semifoldover).
4. Append them to the original half-fraction.

Result: 2^(k-1) + 2^(k-2) = 3 * 2^(k-2) = (3/4) * 2^k runs.

The semifoldover de-aliases all two-factor interactions that involve the
chosen factor from the confounded pairs in the original half-fraction,
effectively raising the resolution for those interactions.

References
----------
John, P. W. M. (1971). *Statistical Design and Analysis of Experiments*.
    Macmillan, New York.
Mee, R. W. and Peralta, M. (2000). Semifolding 2^(k-p) designs.
    *Technometrics*, 42(2), 122-134.
NIST/SEMATECH e-Handbook of Statistical Methods, Section 5.5.7,
    https://www.itl.nist.gov/div898/handbook/pri/section5/pri57.htm
"""

from __future__ import annotations

import numpy as np

from pydoe.factorial.factorial import ff2n


__all__ = ["john_three_quarter_design"]


def john_three_quarter_design(k: int, fold_on: int = 1) -> np.ndarray:
    r"""
    Generate John's 3/4 fractional factorial design for *k* two-level factors.

    The design uses the *semifoldover* construction described by John (1971)
    and detailed in NIST Handbook Section 5.5.7:

    1. **Half-fraction** — Build the :math:`2^{k-1}` design with generator
       :math:`X_k = X_1 X_2 \cdots X_{k-1}`.  This is the highest-resolution
       half-fraction of the full :math:`2^k` factorial.
    2. **Select** the :math:`2^{k-2}` rows where factor *fold_on* is at ``-1``.
    3. **Flip** the sign of *fold_on* in those rows (``-1 → +1``).
    4. **Append** the flipped rows to the original half-fraction.

    The resulting design has :math:`3 \times 2^{k-2} = \tfrac{3}{4} \times 2^k`
    runs and de-aliases every two-factor interaction that involves *fold_on*
    from its alias partner in the original half-fraction.

    Parameters
    ----------
    k : int
        Number of two-level factors. Must be at least 3.
    fold_on : int, optional
        1-based column index of the factor to fold over (default ``1``).
        All two-factor interactions involving this factor will be de-aliased.

    Returns
    -------
    ndarray of shape ``(3 * 2**(k-2), k)``
        Design matrix with entries in ``{-1, +1}``.  The first
        :math:`2^{k-1}` rows are the base half-fraction in Yates order;
        the remaining :math:`2^{k-2}` rows are the augmented semifoldover
        runs.

    Raises
    ------
    ValueError
        If ``k < 3`` or ``fold_on`` is not in ``[1, k]``.

    Examples
    --------
    Four-factor design — saves 4 runs vs the full :math:`2^4 = 16` run design:

    >>> design = john_three_quarter_design(4)
    >>> design.shape
    (12, 4)
    >>> import numpy as np
    >>> bool(np.all(np.isin(design, [-1, 1])))
    True

    The first 8 rows are the :math:`2^{4-1}` half-fraction (generator
    :math:`X_4 = X_1 X_2 X_3`); rows 9-12 are the semifoldover augment on
    :math:`X_1`:

    >>> design[8:]  # augmented rows: X1 flipped from -1 to +1
    array([[ 1., -1., -1., -1.],
           [ 1., -1.,  1.,  1.],
           [ 1.,  1., -1.,  1.],
           [ 1.,  1.,  1., -1.]])

    Fold over a different factor (fold on :math:`X_2` instead):

    >>> john_three_quarter_design(4, fold_on=2).shape
    (12, 4)

    Five-factor design uses 24 runs instead of the full :math:`2^5 = 32`:

    >>> john_three_quarter_design(5).shape
    (24, 5)
    """
    if k < 3:
        raise ValueError(f"k must be at least 3, got {k}")
    if fold_on < 1 or fold_on > k:
        raise ValueError(
            f"fold_on must be between 1 and {k} (inclusive), got {fold_on}"
        )

    # --- Step 1: build the 2^(k-1) half-fraction ---
    # k-1 free factors in Yates order; last column = product of all free
    # columns, implementing the defining generator I = X_1 X_2 ... X_k.
    free = ff2n(k - 1)  # shape (2^(k-1), k-1), values in {-1.0, +1.0}
    gen_col = np.prod(free, axis=1, keepdims=True)
    half_fraction = np.hstack([free, gen_col])  # shape (2^(k-1), k)

    # --- Step 2-3: select rows where fold_on = -1, flip that column ---
    j = fold_on - 1  # convert to 0-based index
    mask = half_fraction[:, j] == -1
    augment = half_fraction[mask].copy()
    augment[:, j] = 1.0

    # --- Step 4: combine ---
    return np.vstack([half_fraction, augment])
