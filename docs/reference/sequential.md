Sequential (adaptive) designs use the results of previous runs to choose
where to sample next. They are particularly useful for optimizing
expensive black-box functions, where a Gaussian process surrogate model
is fit to all observations so far and an *acquisition function* is used
to balance exploration and exploitation when selecting the next point.

This section includes the following sequential design tools:

- [Gaussian Process Regressor](#gaussian-process-regressor)
- [Acquisition Functions](#acquisition-functions)
- [Sequential Design](#sequential-design)

!!! hint
    All available tools can be accessed after a simple import statement:

    ```pycon
    >>> from pydoe import (
    ...     GaussianProcessRegressor,
    ...     expected_improvement,
    ...     probability_of_improvement,
    ...     upper_confidence_bound,
    ...     sequential_design,
    ... )
    ```

## Gaussian Process Regressor (`GaussianProcessRegressor`) {#gaussian-process-regressor}

`GaussianProcessRegressor` is a minimal Gaussian process regression model
with a squared-exponential (RBF) kernel, used internally by
`sequential_design` as the surrogate model. It can also be used directly.

```pycon
>>> gp = GaussianProcessRegressor(length_scale=1.0, noise=1e-8)
>>> gp.fit(X, y)
>>> mean, std = gp.predict(X_new, return_std=True)
```

- `length_scale`: positive float controlling the smoothness of the RBF
  kernel (default: 1.0).
- `noise`: non-negative float added to the diagonal of the kernel matrix
  for numerical stability (default: 1e-8).
- `fit(X, y)`: fit the model to training inputs `X` of shape `(n, d)` and
  targets `y` of shape `(n,)`. Returns `self`.
- `predict(X, *, return_std=False)`: predict the mean (and optionally the
  standard deviation) at query points `X` of shape `(m, d)`.

## Acquisition Functions {#acquisition-functions}

Acquisition functions score candidate points by how promising they are to
sample next, given the GP's predicted `mean` and `std` at each candidate
and the best objective value `best_f` observed so far. For all three
functions, **higher scores indicate more promising points**, regardless
of `maximize`.

```pycon
>>> expected_improvement(mean, std, best_f, *, maximize=True)
>>> probability_of_improvement(mean, std, best_f, *, maximize=True)
>>> upper_confidence_bound(mean, std, *, kappa=1.96, maximize=True)
```

### Examples

```pycon
>>> import numpy as np
>>> from pydoe import (
...     expected_improvement,
...     probability_of_improvement,
...     upper_confidence_bound,
... )
>>> mean = np.array([0.5, 1.5])
>>> std = np.array([0.1, 0.2])
>>> expected_improvement(mean, std, best_f=1.0)
array([5.34616553e-09, 5.00400827e-01])
>>> probability_of_improvement(mean, std, best_f=1.0)
array([2.86651572e-07, 9.93790335e-01])
>>> upper_confidence_bound(mean, std)
array([0.696, 1.892])
```

## Sequential Design (`sequential_design`) {#sequential-design}

`sequential_design` runs a Bayesian-optimization-style adaptive design
loop for an expensive black-box `objective`: it generates an initial
space-filling Latin hypercube of `n_initial` points, fits a
`GaussianProcessRegressor` to all observations, and then for `n_iter`
iterations selects the candidate point (from `n_candidates` random
candidates) maximizing the chosen acquisition function, evaluates
`objective` there, and refits the model.

```pycon
>>> sequential_design(objective, bounds, n_initial, n_iter, *,
...     acquisition="ei", maximize=True, n_candidates=1000,
...     length_scale=1.0, seed=None)
```

- `objective`: callable taking a 1D array of shape `(d,)` and returning a
  scalar float.
- `bounds`: array of shape `(d, 2)`, each row `[low, high]`.
- `n_initial`: number of initial Latin hypercube samples (must be at
  least 1).
- `n_iter`: number of adaptive iterations (must be at least 0).
- `acquisition`: one of `"ei"`, `"pi"`, `"ucb"` (default: `"ei"`).
- `maximize`: whether `objective` is being maximized (default: `True`).
- `n_candidates`: number of random candidate points evaluated by the
  acquisition function at each iteration (default: 1000).
- `length_scale`: passed to the internal `GaussianProcessRegressor`
  (default: 1.0).
- `seed`: an integer or `np.random.Generator` for reproducibility
  (default: `None`).

`sequential_design` returns a tuple `(X, y)` where `X` has shape
`(n_initial + n_iter, d)` and `y` has shape `(n_initial + n_iter,)`.

### Examples

```pycon
>>> import numpy as np
>>> from pydoe import sequential_design
>>> def neg_quadratic(x):
...     return -float((x[0] - 0.3) ** 2)
>>> bounds = np.array([[0.0, 1.0]])
>>> X, y = sequential_design(neg_quadratic, bounds, n_initial=4, n_iter=5, seed=0)
>>> X.shape
(9, 1)
>>> y.shape
(9,)
>>> bool(y.max() > y[:4].max())
True
```

## References

- [Jones, D. R., Schonlau, M., & Welch, W. J. (1998). "Efficient global optimization of expensive black-box functions." *Journal of Global Optimization*, 13(4), 455-492.](https://doi.org/10.1023/A:1008306431147)
- Mockus, J. (1989). *Bayesian Approach to Global Optimization*. Kluwer Academic Publishers.
- [Kushner, H. J. (1964). "A new method of locating the maximum point of an arbitrary multipeak curve in the presence of noise." *Journal of Basic Engineering*, 86(1), 97-106.](https://doi.org/10.1115/1.3653121)
- Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. (2010). "Gaussian process optimization in the bandit setting: No regret and experimental design." *ICML*.
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
