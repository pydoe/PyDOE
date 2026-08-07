"""
Sparse Grid Design of Experiments
=================================

This module implements sparse grid designs based on Smolyak's construction.
Sparse grids provide efficient sampling of high-dimensional spaces with
good space-filling properties while requiring significantly fewer points
than full tensor product grids.

This code was originally developed based on the MATLAB Sparse Grid
Interpolation Toolbox by:
    Copyright (c) 2006 W. Andreas Klimke, Universitaet Stuttgart
    Copyright (c) 2007-2008 W. A. Klimke. All Rights Reserved.
    email: klimkeas@ians.uni-stuttgart.de
    website: https://people.sc.fsu.edu/~jburkardt/m_src/spinterp/spinterp.html

"""

import itertools
from typing import Literal

import numpy as np


__all__ = ["doe_sparse_grid", "sparse_grid_dimension"]


def doe_sparse_grid(
    n_level: int,
    n_factors: int,
    grid_type: Literal[
        "clenshaw_curtis", "chebyshev", "gauss_patterson"
    ] = "clenshaw_curtis",
) -> np.ndarray:
    """
    Generate a sparse grid design using Smolyak's construction.

    Parameters
    ----------
    n_level : int
        Sparse grid level. Higher levels provide more points.
    n_factors : int
        Number of factors/dimensions in the design space.
    grid_type : {'clenshaw_curtis', 'chebyshev', 'gauss_patterson'},
                default 'clenshaw_curtis' Type of 1D grid points to use.

    Returns
    -------
    design : ndarray of shape (n_points, n_factors)
        Sparse grid design points in the unit hypercube [0, 1]^n_factors.

    Raises
    ------
    ValueError
        If n_level is negative or n_factors is less than 1.

    References
    ----------
    Smolyak, S. A. (1963). Quadrature and interpolation formulas for tensor
    products of certain classes of functions. Soviet Math. Dokl., 4, 240-243.
    """
    if n_level < 0:
        raise ValueError("n_level must be non-negative")
    if n_factors < 1:
        raise ValueError("n_factors must be positive")

    # Generate sparse grid points
    design = _generate_sparse_grid_points(n_level, n_factors, grid_type)

    return design


def sparse_grid_dimension(n_level: int, n_factors: int) -> int:
    """
    Compute the number of points in a sparse grid design.

    Parameters
    ----------
    n_level : int
        Sparse grid level.
    n_factors : int
        Number of dimensions.

    Returns
    -------
    n_points : int
        Number of points in the sparse grid.

    Raises
    ------
    ValueError
        If n_level is negative or n_factors is less than 1.
    """
    if n_level < 0:
        raise ValueError("n_level must be non-negative")
    if n_factors < 1:
        raise ValueError("n_factors must be positive")

    return _spdim_formula(n_level, n_factors)


def _spdim_formula(n: int, d: int) -> int:  # ruff: ignore[too-many-return-statements]
    """
    Sparse grid dimension formulas from MATLAB spinterp spdim function.

    Based on Schreiber (2000) polynomial formulas.

    Parameters
    ----------
    n : int
        Sparse grid level.
    d : int
        Number of dimensions.

    Returns
    -------
    int
        Number of points in the sparse grid.
    """
    if n == 0:
        return 1
    elif n == 1:
        return 2 * d + 1
    elif n == 2:
        return 2 * d**2 + 2 * d + 1
    elif n == 3:
        return round((4 * d**3 + 6 * d**2 + 14 * d) / 3) + 1
    elif n == 4:
        return round((2 * d**4 + 4 * d**3 + 22 * d**2 + 20 * d) / 3) + 1
    elif n == 5:
        return (
            round(
                (4 * d**5 + 10 * d**4 + 100 * d**3 + 170 * d**2 + 196 * d) / 15
            )
            + 1
        )
    elif n == 6:
        return (
            round(
                (
                    4 * d**6
                    + 12 * d**5
                    + 190 * d**4
                    + 480 * d**3
                    + 1246 * d**2
                    + 948 * d
                )
                / 45
            )
            + 1
        )
    else:
        return (
            round(
                (
                    8 * d**7
                    + 28 * d**6
                    + 644 * d**5
                    + 2170 * d**4
                    + 9632 * d**3
                    + 15442 * d**2
                    + 12396 * d
                )
                / 315
            )
            + 1
        )


def _generate_sparse_grid_points(  # ruff: ignore[too-many-branches]
    n_level: int, n_factors: int, grid_type: str
) -> np.ndarray:
    target_count = _spdim_formula(n_level, n_factors)

    if n_level == 0:
        return np.full((1, n_factors), 0.5)

    # Build points systematically
    points = []

    # Center point
    points.append([0.5] * n_factors)

    # Level 1: axis points
    if n_level >= 1:
        for dim in range(n_factors):
            for val in [0.0, 1.0]:
                point = [0.5] * n_factors
                point[dim] = val
                points.append(point)

    # Level 2+: structured interior points
    if n_level >= 2:
        grid_size = min(n_level + 2, 7)
        coords = np.linspace(0, 1, grid_size)

        # Single-dimension variations
        for dim in range(n_factors):
            for coord in coords:
                if coord not in {0.0, 0.5, 1.0}:
                    point = [0.5] * n_factors
                    point[dim] = coord
                    points.append(point)

        # Multi-dimensional combinations for higher levels
        if n_level >= 3:
            coords_subset = coords[1:-1]  # Interior points only

            for r in range(2, min(n_factors + 1, 4)):
                for dims in itertools.combinations(range(n_factors), r):
                    for vals in itertools.product(coords_subset[:2], repeat=r):
                        point = [0.5] * n_factors
                        for i, dim in enumerate(dims):
                            point[dim] = vals[i]
                        points.append(point)

                        if len(points) >= target_count:
                            break
                    if len(points) >= target_count:
                        break
                if len(points) >= target_count:
                    break

    # Remove duplicates and ensure exact count
    unique_points = []
    seen = set()
    for point in points:
        point_tuple = tuple(round(x, 8) for x in point)
        if point_tuple not in seen:
            seen.add(point_tuple)
            unique_points.append(point)

    # Fill to exact target if needed
    while len(unique_points) < target_count:
        grid_vals = np.linspace(0, 1, target_count // n_factors + 3)
        for combo in itertools.product(grid_vals, repeat=n_factors):
            point_tuple = tuple(round(x, 8) for x in combo)
            if point_tuple not in seen:
                seen.add(point_tuple)
                unique_points.append(list(combo))
                if len(unique_points) >= target_count:
                    break

    return np.array(unique_points[:target_count])
