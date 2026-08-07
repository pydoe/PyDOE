"""
This module implements the Hammersley point set, a finite low-discrepancy
sequence used in numerical integration, sampling, and global optimization
tasks.

Unlike the Halton sequence, which is generated incrementally and can be
extended indefinitely, the Hammersley point set is defined for a fixed
number of points ``num_points``. The first coordinate of the i-th point is
``i / num_points``, and the remaining coordinates are generated using the
van der Corput sequence with successive prime bases, exactly as in the
Halton sequence.

References
----------
Hammersley, J. M. (1960). "Monte Carlo methods for solving multivariate
    problems." *Annals of the New York Academy of Sciences*, 86(1), 844-874.
    https://doi.org/10.1111/j.1749-6632.1960.tb42846.x
"""

import numpy as np

from .halton import next_primes, van_der_corput


__all__ = ["hammersley_sequence"]


def hammersley_sequence(num_points: int, dimension: int) -> np.ndarray:
    """
    Generate a Hammersley point set in a given dimension.

    The Hammersley point set is a low-discrepancy, deterministic point set
    commonly used in numerical integration, sampling, and global
    optimization. Its first coordinate is the equispaced sequence
    ``i / num_points``, and the remaining ``dimension - 1`` coordinates are
    generated using the van der Corput sequence with successive prime
    bases.

    Parameters
    ----------
    num_points : int
        Number of points to generate in the point set, must be positive.
    dimension : int
        Number of dimensions (features) of the point set, must be
        positive.

    Returns
    -------
    points : ndarray of shape (`num_points`, `dimension`)
        The generated Hammersley point set.

    Raises
    ------
    ValueError
        If ``num_points`` or ``dimension`` is less than 1.

    Examples
    --------
    >>> hammersley_sequence(4, 2)
    array([[0.  , 0.  ],
           [0.25, 0.5 ],
           [0.5 , 0.25],
           [0.75, 0.75]])
    """
    if num_points < 1:
        raise ValueError(f"num_points must be at least 1, got {num_points}")
    if dimension < 1:
        raise ValueError(f"dimension must be at least 1, got {dimension}")

    samples = np.empty((num_points, dimension), dtype=np.float64)
    samples[:, 0] = np.arange(num_points) / num_points

    if dimension > 1:
        bases = next_primes(dimension - 1)
        for dim in range(1, dimension):
            base = bases[dim - 1]
            for i in range(num_points):
                samples[i, dim] = van_der_corput(i, base)

    return samples
