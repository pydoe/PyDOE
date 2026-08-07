Low-discrepancy sequences, also known as *quasi-random sequences*, are
used for sampling points in a multi-dimensional space in a way that fills
the space more uniformly than uncorrelated random points. These sequences
are particularly valuable in computer experiments, numerical integration,
global optimization, and design of experiments.

This section includes the following quasi-random designs:

- [Sukharev Grid](#sukharev_grid)
- [Sobol' Sequence](#sobol_sequence)
- [Halton Sequence](#halton_sequence)
- [Hammersley Point Set](#hammersley_sequence)
- [Rank-1 Lattice Design](#rank1_lattice)
- [Korobov Sequence](#korobov_sequence)
- [Faure Sequence](#faure_sequence)
- [Niederreiter Sequence](#niederreiter_sequence)
- [Cranley-Patterson Randomization](#cranley_patterson)

!!! hint
    All sequence functions are available with:

    ```python
    >>> from pydoe import (sukharev_grid, sobol_sequence,
    ...     halton_sequence, hammersley_sequence, rank1_lattice,
    ...     korobov_sequence, faure_sequence, niederreiter_sequence,
    ...     cranley_patterson_shift)
    ```

## Sukharev Grid (`sukharev_grid`) {#sukharev_grid}

The **Sukharev grid** is a deterministic low-discrepancy design that places
points at the centers of equally sized subcells in the unit hypercube.
Unlike random sampling, no points are located on the boundaries, which
minimizes the covering radius with respect to the max-norm.

**Syntax**:

```python
sukharev_grid(num_points, dimension)
```

- `num_points`: total number of points to generate. Must be an integer power of `dimension`.
- `dimension`: dimensionality of the space.

**Example**:

```pycon
>>> sukharev_grid(4, 2)
array([[0.25, 0.25],
       [0.25, 0.75],
       [0.75, 0.25],
       [0.75, 0.75]])
```

!!! note
    The Sukharev grid is especially useful when deterministic space-filling coverage of the design space is desired.

### See Also

- [Low-discrepancy sequences](https://en.wikipedia.org/wiki/Low-discrepancy_sequence)
- [Quasi-Monte Carlo methods](https://en.wikipedia.org/wiki/Quasi-Monte_Carlo_method)

## Sobol' Sequence (`sobol_sequence`) {#sobol_sequence}

Sobol' sequences are highly uniform low-discrepancy sequences commonly
used in numerical methods and uncertainty quantification.

**Syntax**:

```python
sobol_sequence(n, d, *, scramble=False, seed=None, bounds=None,
               skip=0, use_pow_of_2=True)
```

- `n`: number of points to generate.
- `d`: number of dimensions (must be ≤ 21201).
- `scramble`: whether to apply Owen scrambling (default: `False`).
- `seed`: integer seed for reproducibility (only used when `scramble=True`).
- `bounds`: array of shape `(d, 2)` with per-dimension `(min, max)` pairs.
- `skip`: number of initial Sobol' points to skip via fast-forward (default: 0).
- `use_pow_of_2`: if `True` (default), rounds `n` up to the nearest power of 2
  and uses `random_base2` for ideal balance properties. If `False`, generates
  exactly `n` samples using `random` — useful when an exact count is required
  (e.g. inside `saltelli_sampling`).

**Example**:

```python
>>> sobol_sequence(4, 2)
array([[0.    , 0.    ],
       [0.5   , 0.5   ],
       [0.75  , 0.25  ],
       [0.25  , 0.75  ]])
```

!!! note
    For best balance and coverage, `n` should be a power of 2 and
    `use_pow_of_2=True` (the default). Non-power-of-2 values are silently
    rounded up when `use_pow_of_2=True`; set `use_pow_of_2=False` only when
    the caller needs an exact sample count.

## Halton Sequence (`halton_sequence`) {#halton_sequence}

The Halton sequence generates low-discrepancy samples using mutually
prime number bases for each dimension.

**Syntax**:

```python
>>> halton_sequence(num_points, dimension, primes=None)
```

- `num_points`: number of samples.
- `dimension`: number of dimensions.
- `primes`: optional list of prime bases; defaults to the first `dimension` primes.

**Example**:

```python
>>> halton_sequence(5, 2)
array([[0.        , 0.        ],
       [0.5       , 0.33333333],
       [0.25      , 0.66666667],
       [0.75      , 0.11111111],
       [0.125     , 0.44444444]])
```

## Hammersley Point Set (`hammersley_sequence`) {#hammersley_sequence}

The **Hammersley point set** is a finite, fixed-size low-discrepancy
point set. Its first coordinate places one point at $i / N$ for
$i = 0, \ldots, N-1$, while the remaining coordinates are generated using
the van der Corput sequence with successive prime bases, exactly as in
the Halton sequence. Because it is defined for a fixed sample size $N$,
it achieves better uniformity than the Halton sequence for that size.

**Syntax**:

```python
>>> hammersley_sequence(num_points, dimension)
```

- `num_points`: number of points to generate, must be at least 1.
- `dimension`: number of dimensions, must be at least 1.

**Example**:

```python
>>> hammersley_sequence(4, 2)
array([[0.  , 0.  ],
       [0.25, 0.5 ],
       [0.5 , 0.25],
       [0.75, 0.75]])
```

!!! note
    Unlike `halton_sequence`, `hammersley_sequence` cannot be extended
    incrementally — the entire point set depends on `num_points`. Use it
    when the total sample size is known in advance.

## Rank-1 Lattice Design (`rank1_lattice`) {#rank1_lattice}

A **Rank-1 Lattice** is a deterministic method to construct points that
fill the space uniformly using modular arithmetic.

**Syntax**:

```python
rank1_lattice(num_points, dimension, generator=None)
```

- `num_points`: number of points.
- `dimension`: dimensionality of space.
- `generator`: optional list of length `dimension` used as a multiplier.

**Example**:

```python
>>> rank1_lattice(5, 2)
array([[0, 0],
       [2, 2],
       [4, 4],
       [1, 1],
       [3, 3]])
```

## Korobov Sequence (`korobov_sequence`) {#korobov_sequence}

The **Korobov sequence** is a special case of rank-1 lattices using a
single integer base to construct all dimensions.

**Syntax**:

```python
>>> korobov_sequence(num_points, dimension, a=None)
```

- `num_points`: number of points.
- `dimension`: number of dimensions.
- `generator_param`: optional generator integer (default: None).

**Example**:

```python
>>> korobov_sequence(5, 3, generator_param=3)
array([[0, 0, 0],
       [1, 3, 4],
       [2, 1, 3],
       [3, 4, 2],
       [4, 2, 1]])
```

## Faure Sequence (`faure_sequence`) {#faure_sequence}

The **Faure sequence** is a low-discrepancy sequence that uses a single
prime base $b \geq \max(\text{dimension}, 2)$ for every dimension. Each
dimension's digit expansion is obtained by applying a Pascal
(binomial-coefficient) permutation matrix, raised to a dimension-dependent
power, to the base-$b$ digits of the sequence index. This yields better
uniformity than the Halton sequence in higher dimensions.

**Syntax**:

```python
faure_sequence(num_points, dimension, skip=0)
```

- `num_points`: number of points to generate, must be at least 1.
- `dimension`: number of dimensions, must be at least 1.
- `skip`: number of initial sequence indices to skip (default: 0).

**Example**:

```python
>>> faure_sequence(5, 2)
array([[0.   , 0.   ],
       [0.5  , 0.5  ],
       [0.25 , 0.75 ],
       [0.75 , 0.25 ],
       [0.125, 0.625]])
```

!!! note
    For `dimension == 1`, `faure_sequence` reduces to the van der Corput
    sequence in base 2.

## Niederreiter Sequence (`niederreiter_sequence`) {#niederreiter_sequence}

The **Niederreiter sequence** is a digital $(t, m, s)$-net constructed in
base 2. Each dimension's generating matrix is built from the
linear-feedback-shift-register (LFSR) sequence of a primitive polynomial
over $\mathrm{GF}(2)$, achieving optimal theoretical discrepancy bounds for
high-dimensional integration.

**Syntax**:

```python
niederreiter_sequence(num_points, dimension, n_bits=30)
```

- `num_points`: number of points to generate, must be at least 1.
- `dimension`: number of dimensions, must be between 1 and 20 (inclusive).
- `n_bits`: number of bits of precision used to build the generating
  matrices (default: 30, must be at least 2).

**Example**:

```python
>>> niederreiter_sequence(4, 2)
array([[0.        , 0.        ],
       [0.42857143, 0.18110236],
       [0.85714286, 0.36220472],
       [0.71428571, 0.4488189 ]])
```

!!! note
    `dimension` is limited to 20 by the table of pre-verified primitive
    polynomials over $\mathrm{GF}(2)$ used to build the generating matrices.

## Cranley-Patterson Randomization (`cranley_patterson_shift`) {#cranley_patterson}

The **Cranley-Patterson method** applies a random shift to a
quasi-random sequence and wraps the result within the unit hypercube.

**Syntax**:

```python
>>> cranley_patterson_shift(samples, seed=None)
```

- `samples`: input samples to randomize.
- `seed`: optional random seed for reproducibility.

**Example**:

```python
>>> from pydoe import halton_sequence, cranley_patterson_shift
>>> x = halton_sequence(4, 2)
>>> cranley_patterson_shift(x, seed=42)
array([[0.77395605, 0.43887844],
       [0.27395605, 0.77221177],
       [0.02395605, 0.10554511],
       [0.52395605, 0.54998955]])
```

!!! note
    Cranley-Patterson randomization improves statistical independence between runs and is particularly helpful when replicating experiments or integrating results.

### See Also

- [Sobol sequence](https://en.wikipedia.org/wiki/Sobol_sequence)
- [Halton sequence](https://en.wikipedia.org/wiki/Halton_sequence)
- [Low-discrepancy sequences](https://en.wikipedia.org/wiki/Low-discrepancy_sequence)

## References

- [Sukharev, A. G. (1971). "Optimal strategies of the search for an extremum." *USSR Computational Mathematics and Mathematical Physics*, 11(4), 119-137.](https://doi.org/10.1016/0041-5553(71)90008-5)
- [Cranley, R., and Patterson, T. N. L. (1976). "Randomization of Number Theoretic Methods for Multiple Integration." *SIAM Journal on Numerical Analysis*, 13(6), 904-914.](https://doi.org/10.1137/0713071)
- [Halton, J. H. (1964). "Algorithm 247: Radical-inverse quasi-random point sequence." *Communications of the ACM*, 7(12), 701.](https://doi.org/10.1145/355588.365104)
- [Sobol', I. M. (1967). "Distribution of points in a cube and approximate evaluation of integrals." *Zh. Vych. Mat. Mat. Fiz.*, 7: 784-802 (in Russian); *U.S.S.R. Comput. Maths. Math. Phys.*, 7: 86-112.](https://doi.org/10.1016/0041-5553(71)90008-5)
- [Hammersley, J. M. (1960). "Monte Carlo methods for solving multivariate problems." *Annals of the New York Academy of Sciences*, 86(1), 844-874.](https://doi.org/10.1111/j.1749-6632.1960.tb42846.x)
- Faure, H. (1982). "Discrépance de suites associées à un système de numération (en dimension s)." *Acta Arithmetica*, 41(4), 337-351.
- Niederreiter, H. (1988). "Low-discrepancy and low-dispersion sequences." *Journal of Number Theory*, 30(1), 51-70.
- [Bratley, P., & Fox, B. L. (1988). "Algorithm 659: Implementing Sobol's quasirandom sequence generator." *ACM Transactions on Mathematical Software*, 14(1), 88-100.](https://doi.org/10.1145/42288.214372)
