In this section, the following specialized designs are described:

- [Definitive Screening Design](#definitive-screening-design-definitive_screening_design)
- [Supersaturated Design](#supersaturated-design-supersaturated_design)

!!! note
    All available designs can be accessed after a simple import statement:
    ```pycon
    >>> import numpy as np
    >>> from pydoe import definitive_screening_design, supersaturated_design
    ```

## Definitive Screening Design (`definitive_screening_design`) {#definitive-screening-design-definitive_screening_design}

A **definitive screening design** (Jones & Nachtsheim, 2011) is a
three-level design for continuous factors that estimates all main
effects independently of two-factor interactions and can detect
curvature. The Jones & Nachtsheim (2013) extension adds two-level
categorical factors.

```pycon
>>> definitive_screening_design(
...     k,
...     n_cat=0,
...     *,
...     method="dsd",
...     min_13=False,
...     n_fake_factors=0,
... )
```

- `k` is the number of continuous factors. It may be zero when
  `n_cat` is positive.
- `n_cat` is the number of two-level categorical factors.
- `method="dsd"` de-aliases two-factor interactions involving
  categorical factors. `method="orth"` instead produces an orthogonal
  main-effects plan.
- `min_13=True` augments a small design with enough temporary factors
  to provide at least 13 runs.
- `n_fake_factors` manually adds temporary continuous factors to
  increase the number of runs. Their columns are not returned.

Continuous-factor columns precede categorical-factor columns. Continuous
levels are coded as `-1`, `0`, and `1`, while categorical levels use
PyDOE's conventional two-level coding, `-1` and `1`. The function always
returns a floating-point NumPy array.

The historical one-argument result and row order are unchanged:

```pycon
>>> definitive_screening_design(4)
array([[ 0.,  0.,  0.,  0.],
       [ 0.,  1.,  1.,  1.],
       [-1.,  0.,  1., -1.],
       [-1., -1.,  0.,  1.],
       [-1.,  1., -1.,  0.],
       [ 0., -1., -1., -1.],
       [ 1.,  0., -1.,  1.],
       [ 1.,  1.,  0., -1.],
       [ 1., -1.,  1.,  0.]])
```

!!! note
    When a Paley conference matrix exists at the requested size, the
    design is $D = [0; C; -C]$. Other factor counts use the next suitable
    conference matrix and retain only the requested columns. Efficient
    special constructions are used around 10, 16, and 26 factors.

Mixed continuous and categorical designs are generated directly:

```pycon
>>> design = definitive_screening_design(3, n_cat=2)
>>> design.shape
(14, 5)
>>> np.unique(design[:, :3])
array([-1.,  0.,  1.])
>>> np.unique(design[:, 3:])
array([-1.,  1.])
```

The orthogonal method adds the center runs needed to make all main-effect
columns orthogonal:

```pycon
>>> orthogonal = definitive_screening_design(3, n_cat=2, method="orth")
>>> orthogonal.shape
(16, 5)
>>> gram = orthogonal.T @ orthogonal
>>> np.allclose(gram - np.diag(np.diag(gram)), 0)
True
```

Small designs can be enlarged without exposing the temporary columns:

```pycon
>>> definitive_screening_design(4).shape
(9, 4)
>>> definitive_screening_design(4, min_13=True).shape
(13, 4)
>>> definitive_screening_design(4, n_fake_factors=2).shape
(13, 4)
```

## Supersaturated Design (`supersaturated_design`) {#supersaturated-design-supersaturated_design}

A **supersaturated design** has more two-level factors than runs
($k > n$). It cannot estimate all main effects simultaneously, but is
useful for screening when only a small fraction of factors are
expected to be active. `supersaturated_design` performs a random
search to minimize $E(s^2)$, the average squared off-diagonal element
of $X^T X$.

```pycon
>>> supersaturated_design(n_factors, n_runs, iterations=1000, seed=None)  # (1)!
```

1. `n_factors` — number of two-level factors $k$ (must exceed
   `n_runs`). `n_runs` — number of runs $n$ (≥ 2). `iterations` —
   number of random candidates to evaluate. `seed` — for
   reproducibility.

```pycon
>>> supersaturated_design(6, 4, iterations=200, seed=0)
array([[-1.,  1.,  1., -1., -1.,  1.],
       [ 1.,  1., -1.,  1.,  1.,  1.],
       [ 1.,  1.,  1., -1.,  1., -1.],
       [ 1.,  1.,  1.,  1., -1., -1.]])
```

!!! note
    Smaller $E(s^2)$ values indicate lower average correlation between
    factor columns, allowing cleaner estimation of the active effects
    under effect sparsity.

## More Information

For further reading, see:

- Jones, B., & Nachtsheim, C. J. (2011). A class of three-level designs
  for definitive screening in the presence of second-order effects.
  *Journal of Quality Technology*, 43(1), 1-15.
- Xiao, L., Lin, D. K. J., & Bai, F. (2012). Constructing definitive
  screening designs using conference matrices. *Journal of Quality
  Technology*, 44(1), 2-8.
- Jones, B., & Nachtsheim, C. J. (2013). Definitive screening designs
  with added two-level categorical factors. *Journal of Quality
  Technology*, 45(2), 121-129.
- Lin, D. K. J. (1993). A new class of supersaturated designs.
  *Technometrics*, 35(1), 28-31.
- [NIST Handbook Section 5.3.3.4 — Supersaturated Designs](https://www.itl.nist.gov/div898/handbook/pri/section3/pri334.htm)
