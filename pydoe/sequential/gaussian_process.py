"""
Minimal Gaussian process regression with an RBF kernel.

This module implements a lightweight Gaussian process (GP) regressor
used as a surrogate model for sequential / adaptive design. Only
``numpy`` and ``scipy`` are required.

References
----------
Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for
    Machine Learning*. MIT Press.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.spatial.distance import cdist


__all__ = ["GaussianProcessRegressor"]


class GaussianProcessRegressor:
    """
    Gaussian process regressor with a squared-exponential kernel.

    The kernel is the radial basis function (RBF):

    .. math::

        k(x, x') = \\exp\\left(-\\frac{\\lVert x - x' \\rVert^2}
        {2 \\ell^2}\\right)

    where :math:`\\ell` is the ``length_scale``.

    Attributes
    ----------
    length_scale : float
        Length scale :math:`\\ell` of the RBF kernel.
    noise : float
        Variance added to the diagonal of the training kernel matrix
        for numerical stability and to model observation noise.

    Parameters
    ----------
    length_scale : float, optional
        Length scale of the RBF kernel, must be strictly positive.
        Default is 1.0.
    noise : float, optional
        Non-negative noise variance added to the kernel diagonal.
        Default is 1e-8.

    Raises
    ------
    ValueError
        If ``length_scale`` is not strictly positive or ``noise`` is
        negative.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0.0], [0.5], [1.0]])
    >>> y = np.array([0.0, 1.0, 0.0])
    >>> gp = GaussianProcessRegressor(length_scale=0.5).fit(X, y)
    >>> mean, std = gp.predict(np.array([[0.5]]), return_std=True)
    >>> bool(abs(mean[0] - 1.0) < 0.1)
    True
    >>> bool(std[0] >= 0.0)
    True
    """

    def __init__(self, length_scale: float = 1.0, noise: float = 1e-8) -> None:
        if length_scale <= 0:
            raise ValueError(
                f"length_scale must be strictly positive, got {length_scale}"
            )
        if noise < 0:
            raise ValueError(f"noise must be non-negative, got {noise}")

        self.length_scale = length_scale
        self.noise = noise
        self._X_train: np.ndarray | None = None
        self._L: np.ndarray | None = None
        self._alpha: np.ndarray | None = None
        self._y_mean: float | None = None

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Compute the RBF Gram matrix between two sets of points.

        Parameters
        ----------
        X1 : ndarray of shape (n1, d)
            First set of points.
        X2 : ndarray of shape (n2, d)
            Second set of points.

        Returns
        -------
        ndarray of shape (n1, n2)
            RBF kernel matrix.
        """
        sq_dists = cdist(X1, X2, "sqeuclidean")
        return np.exp(-sq_dists / (2 * self.length_scale**2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
        """
        Fit the Gaussian process to training data.

        Parameters
        ----------
        X : ndarray of shape (n, d)
            Training input points.
        y : ndarray of shape (n,)
            Training target values.

        Returns
        -------
        GaussianProcessRegressor
            The fitted estimator (for method chaining).

        Raises
        ------
        ValueError
            If ``X`` and ``y`` have mismatched lengths or if ``X`` is
            empty.

        Examples
        --------
        >>> import numpy as np
        >>> X = np.array([[0.0], [1.0]])
        >>> y = np.array([0.0, 1.0])
        >>> gp = GaussianProcessRegressor().fit(X, y)
        >>> gp is not None
        True
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples, got "
                f"{X.shape[0]} and {y.shape[0]}"
            )
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")

        self._y_mean = float(np.mean(y))
        y_centered = y - self._y_mean

        K = self._kernel(X, X) + self.noise * np.eye(X.shape[0])
        L = cholesky(K, lower=True)
        alpha = cho_solve((L, True), y_centered)

        self._X_train = X
        self._L = L
        self._alpha = alpha
        return self

    def predict(
        self, X: np.ndarray, *, return_std: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Predict the posterior mean (and optionally std) at new points.

        Parameters
        ----------
        X : ndarray of shape (n, d)
            Query points.
        return_std : bool, optional
            If True, also return the posterior standard deviation at
            each query point. Default is False.

        Returns
        -------
        mean : ndarray of shape (n,)
            Posterior mean predictions.
        std : ndarray of shape (n,)
            Posterior standard deviations, only returned if
            ``return_std`` is True.

        Raises
        ------
        ValueError
            If the model has not been fit yet.

        Examples
        --------
        >>> import numpy as np
        >>> X = np.array([[0.0], [1.0]])
        >>> y = np.array([0.0, 1.0])
        >>> gp = GaussianProcessRegressor().fit(X, y)
        >>> mean = gp.predict(np.array([[0.0], [1.0]]))
        >>> mean.shape
        (2,)
        """
        if self._X_train is None:
            raise ValueError("model has not been fit")

        X = np.asarray(X, dtype=float)
        K_star = self._kernel(self._X_train, X)
        mean = K_star.T @ self._alpha + self._y_mean

        if not return_std:
            return mean

        v = solve_triangular(self._L, K_star, lower=True)
        var = 1.0 - np.sum(v**2, axis=0)
        var = np.clip(var, 0.0, None)
        std = np.sqrt(var)
        return mean, std
