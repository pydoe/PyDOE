from __future__ import annotations

from collections.abc import Callable
from warnings import warn

import numpy as np


def random_k_means(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    num_points: int,
    dimension: int,
    num_steps: int | None = None,
    initial_points: np.ndarray = None,
    callback: Callable | None = None,
    random_state: int | None = None,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """
    MacQueen's K-Means algorithm.

    Parameters
    ----------
    num_points : int
        Number of cluster centers to generate.
    dimension : int
        Dimensionality of the space.
    num_steps : int, optional
        Number of iterations. Defaults to 100 * num_points.
    initial_points : array_like, optional
        Initial cluster centers. If None, random points in [0, 1]^dimension
        are used.
    callback : callable, optional
        Function called at each step with current cluster centers.
    random_state : int, optional
        (Deprecated) Random seed for reproducibility. Use `seed` parameter
        instead.
    seed : int or np.random.Generator, optional
        Random seed or np.random.Generator for reproducibility.

    Returns
    -------
    cluster_centers : np.ndarray
        Array of shape (num_points, dimension) containing the cluster centers.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """
    if random_state is not None:
        warn(
            "The 'random_state' parameter is deprecated. "
            "Use the 'seed' parameter instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if seed is None:
            seed = random_state

    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    if num_steps is None:
        num_steps = 100 * num_points

    # Initialize cluster centers
    if initial_points is None:
        cluster_centers = rng.random((num_points, dimension))
    else:
        cluster_centers = np.array(initial_points)
        if cluster_centers.shape != (num_points, dimension):
            raise ValueError(
                "initial_points must have shape (num_points, dimension)"
            )
        if not np.all((0.0 <= cluster_centers) & (cluster_centers <= 1.0)):
            raise ValueError("initial_points must be in [0, 1]^dimension")

    # Initialize counts for incremental mean
    counts = np.ones(num_points)

    for _ in range(num_steps):
        if callback is not None:
            callback(cluster_centers)

        # Sample a random point in the unit hypercube
        x = rng.random(dimension)

        # Compute Euclidean distances to cluster centers
        distances = np.linalg.norm(cluster_centers - x, axis=1)

        # Find nearest cluster center
        idx = np.argmin(distances)

        # Update cluster center incrementally (MacQueen's update)
        cluster_centers[idx] = (counts[idx] * cluster_centers[idx] + x) / (
            counts[idx] + 1
        )
        counts[idx] += 1

    return cluster_centers
