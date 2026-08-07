"""
Blocking of full factorial designs.

When the runs of a :math:`2^k` factorial design cannot all be carried
out under homogeneous conditions (e.g. they must be split across
several days, batches of raw material, or operators), the design can be
split into :math:`2^p` blocks by deliberately confounding one or more
high-order interactions with the block effect. Each chosen interaction
becomes completely indistinguishable from the block-to-block
difference, which is an acceptable trade-off when that interaction is
believed to be negligible.

References
----------
NIST/SEMATECH e-Handbook of Statistical Methods, Section 5.3.3.4.1 -
    "Blocking a replicated full factorial design",
    https://www.itl.nist.gov/div898/handbook/pri/section3/pri3341.htm
Box, G. E. P., Hunter, W. G., & Hunter, J. S. (2005). *Statistics for
    Experimenters* (2nd ed.). Wiley.
"""

from __future__ import annotations

import numpy as np

from pydoe.factorial.factorial import ff2n


__all__ = ["block_full_factorial"]


def block_full_factorial(
    k: int, generators: list[tuple[int, ...]]
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Split a :math:`2^k` full factorial design into :math:`2^p` blocks.

    Each element of ``generators`` is a tuple of 0-based factor indices
    whose interaction column defines a block contrast. For generator
    *i*, the sign of the product of the corresponding columns
    determines bit *i* of the block number, so ``len(generators)``
    generators produce :math:`2^{\\text{len(generators)}}` blocks of
    equal size. The chosen interactions (and any of their generalized
    interactions) are completely confounded with block effects.

    Parameters
    ----------
    k : int
        Number of factors, must be at least 2.
    generators : list of tuple of int
        Each tuple lists the 0-based factor indices (in
        ``range(k)``) whose interaction defines one block contrast.
        Every tuple must be non-empty with distinct indices in range.

    Returns
    -------
    design : ndarray of shape (2**k, k)
        The full :math:`2^k` factorial design with coded levels -1
        and 1, in standard order.
    blocks : ndarray of shape (2**k,)
        Integer block label (0 to ``2**len(generators) - 1``) for
        each row of ``design``.

    Raises
    ------
    ValueError
        If ``k < 2``, ``generators`` is empty, or any generator is
        empty, contains an out-of-range or repeated index, or
        :math:`2^{\\text{len(generators)}}` exceeds :math:`2^k`.

    Examples
    --------
    Confound the three-factor interaction ``ABC`` with two blocks:

    >>> design, blocks = block_full_factorial(3, [(0, 1, 2)])
    >>> design
    array([[-1., -1., -1.],
           [-1., -1.,  1.],
           [-1.,  1., -1.],
           [-1.,  1.,  1.],
           [ 1., -1., -1.],
           [ 1., -1.,  1.],
           [ 1.,  1., -1.],
           [ 1.,  1.,  1.]])
    >>> blocks
    array([1, 0, 0, 1, 0, 1, 1, 0])

    Within each block, the sign of the ``ABC`` product is constant:

    >>> bool(np.all(np.sign(design.prod(axis=1))[blocks == 0] == 1))
    True
    """
    if k < 2:
        raise ValueError(f"k must be at least 2, got {k}")
    if not generators:
        raise ValueError("generators must contain at least one tuple")

    p = len(generators)
    if 2**p > 2**k:
        raise ValueError(
            f"2**len(generators) = {2**p} exceeds the number of runs "
            f"2**k = {2**k}"
        )

    for gen in generators:
        if not gen:
            raise ValueError("each generator must be a non-empty tuple")
        if len(set(gen)) != len(gen):
            raise ValueError(f"generator {gen} contains repeated indices")
        if any(idx < 0 or idx >= k for idx in gen):
            raise ValueError(
                f"generator {gen} contains an index outside range(0, {k})"
            )

    design = ff2n(k)
    n_runs = design.shape[0]
    blocks = np.zeros(n_runs, dtype=int)
    for i, gen in enumerate(generators):
        contrast = np.prod(design[:, gen], axis=1)
        bit = (contrast < 0).astype(int)
        blocks += bit * (2**i)

    return design, blocks
