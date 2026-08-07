In this section, the following mixture designs are described:

- [Simplex-Lattice Design](#simplex-lattice-design-simplex_lattice_design)
- [Simplex-Centroid Design](#simplex-centroid-design-simplex_centroid_design)
- [Axial (Screening) Design](#axial-screening-design-mixture_axial_design)
- [Extreme-Vertices Design](#extreme-vertices-design-extreme_vertices_design)
- [Mixture-Process Variable Design](#mixture-process-variable-design-mixture_process_design)

!!! note
    All available designs can be accessed after a simple import statement:
    ```pycon
    >>> from pydoe import (
    ...     simplex_lattice_design,
    ...     simplex_centroid_design,
    ...     mixture_axial_design,
    ...     extreme_vertices_design,
    ...     mixture_process_design,
    ... )
    ```

## Background

Mixture experiments differ from standard factorial experiments because the
factors are **component proportions** that must sum to 1:

$$x_1 + x_2 + \cdots + x_q = 1, \quad x_i \ge 0.$$

The experimental region is not a hypercube but the *q*-component **simplex**
— a triangle for *q* = 3, a tetrahedron for *q* = 4, and so on. Ordinary
factorial or response-surface designs cannot be used directly because they
ignore the mixture constraint.

Typical applications include formulation studies in food science, chemical
engineering, pharmaceuticals, and materials science, where the response
(yield, viscosity, strength, …) depends on the *relative amounts* of
ingredients rather than their absolute quantities.

## Simplex-Lattice Design (`simplex_lattice_design`) {#simplex-lattice-design-simplex_lattice_design}

A **{q, m} simplex-lattice design** (Scheffé 1958) places design points at
all lattice points of the simplex at resolution 1/*m*.  Each component takes
values in the set $\{0, 1/m, 2/m, \ldots, 1\}$, with all *q* proportions
summing to 1.  The total number of design points is

$$N = \binom{q + m - 1}{m}.$$

```pycon
>>> simplex_lattice_design(q, m)  # (1)!
```

1. `q` — number of components (≥ 2). `m` — lattice degree (≥ 1).

**Two-component lattice of degree 3** — four evenly-spaced blends along
the binary simplex (a line segment):

```pycon
>>> simplex_lattice_design(2, 3)
array([[1.        , 0.        ],
       [0.66666667, 0.33333333],
       [0.33333333, 0.66666667],
       [0.        , 1.        ]])
```

**Three-component lattice of degree 2** — 6 points: the 3 pure-component
vertices plus the 3 binary edge midpoints:

```pycon
>>> simplex_lattice_design(3, 2)
array([[1. , 0. , 0. ],
       [0.5, 0.5, 0. ],
       [0.5, 0. , 0.5],
       [0. , 1. , 0. ],
       [0. , 0.5, 0.5],
       [0. , 0. , 1. ]])
```

The run count grows combinatorially with *m*:

| *q* \\ *m* | 1 | 2 | 3 |  4 |
|:---:|:---:|:---:|:---:|:---:|
| 2 | 2 | 3 | 4 | 5 |
| 3 | 3 | 6 | 10 | 15 |
| 4 | 4 | 10 | 20 | 35 |
| 5 | 5 | 15 | 35 | 70 |

!!! note
    *m* = 1 returns the *q* pure-component vertices — the same as the rows
    of the *q* × *q* identity matrix.  *m* = 2 adds binary blends (edge
    midpoints).  Higher *m* fills the simplex more densely and supports
    fitting polynomial models of degree *m*.

## Simplex-Centroid Design (`simplex_centroid_design`) {#simplex-centroid-design-simplex_centroid_design}

A **simplex-centroid design** (Scheffé 1963) consists of the centroid of
every non-empty subset of the *q* components.  For a subset of size *s*, the
centroid sets each component in the subset to 1/*s* and all others to 0.
The design contains $2^q - 1$ points, ordered by increasing subset size:

```pycon
>>> simplex_centroid_design(q)  # (1)!
```

1. `q` — number of components (≥ 2).

**Three-component simplex-centroid** — 7 points (3 vertices, 3 edge
midpoints, 1 overall centroid):

```pycon
>>> simplex_centroid_design(3)
array([[1.        , 0.        , 0.        ],
       [0.        , 1.        , 0.        ],
       [0.        , 0.        , 1.        ],
       [0.5       , 0.5       , 0.        ],
       [0.5       , 0.        , 0.5       ],
       [0.        , 0.5       , 0.5       ],
       [0.33333333, 0.33333333, 0.33333333]])
```

The design always contains:

- The *q* **pure-component vertices** (first *q* rows): each row is a
  standard basis vector.
- The $\binom{q}{2}$ **binary edge midpoints** (next rows): equal blends of
  every pair of components.
- … (higher-order subset centroids) …
- The single **overall centroid** (last row): $(1/q, \ldots, 1/q)$.

| *q* | Points ($2^q - 1$) |
|:---:|:---:|
| 2 | 3 |
| 3 | 7 |
| 4 | 15 |
| 5 | 31 |

!!! note
    The simplex-centroid design is a special case of the simplex-lattice
    design only for *q* = 2 and *q* = 3.  For *q* ≥ 4 the two designs
    differ: the centroid design emphasizes subset centroids, while the
    lattice design provides a regular grid.  For fitting full Scheffé
    canonical polynomial models, the centroid design is often preferred
    because it supports estimation of all interaction blending coefficients.

## Axial (Screening) Design (`mixture_axial_design`) {#axial-screening-design-mixture_axial_design}

An **axial design** places one point at the overall centroid plus, for
each component, one point a fraction `delta` of the way from the
centroid towards that component's pure vertex. It is a simple way to
screen which components most affect the response.

```pycon
>>> mixture_axial_design(q, delta=0.5)  # (1)!
```

1. `q` — number of components (≥ 2). `delta` — fraction of the distance
   from the centroid to each vertex, $0 < \delta \le 1$ (default 0.5).

```pycon
>>> mixture_axial_design(3)
array([[0.33333333, 0.33333333, 0.33333333],
       [0.66666667, 0.16666667, 0.16666667],
       [0.16666667, 0.66666667, 0.16666667],
       [0.16666667, 0.16666667, 0.66666667]])
```

!!! note
    Every row sums to 1. The first row is always the centroid; row $i+1$
    perturbs only component $i$ relative to the centroid.

## Extreme-Vertices Design (`extreme_vertices_design`) {#extreme-vertices-design-extreme_vertices_design}

When each component is restricted to its own range
$l_i \le x_i \le u_i$, the feasible mixture region becomes a polytope
nested inside the simplex. An **extreme-vertices design** places a
design point at every vertex of this polytope.

```pycon
>>> extreme_vertices_design(lower, upper)  # (1)!
```

1. `lower`, `upper` — arrays of per-component lower and upper bounds,
   with $\sum_i l_i \le 1 \le \sum_i u_i$.

```pycon
>>> extreme_vertices_design([0.1, 0.1, 0.1], [0.6, 0.6, 0.6])
array([[0.1, 0.3, 0.6],
       [0.1, 0.6, 0.3],
       [0.3, 0.1, 0.6],
       [0.3, 0.6, 0.1],
       [0.6, 0.1, 0.3],
       [0.6, 0.3, 0.1]])
```

!!! note
    This implementation enumerates vertices of regions defined by simple
    per-component bounds only; it does not support additional general
    linear constraints between components.

## Mixture-Process Variable Design (`mixture_process_design`) {#mixture-process-variable-design-mixture_process_design}

Many mixture experiments also vary independent **process variables**
(temperature, mixing time, etc.) that are not subject to the
sum-to-one constraint. `mixture_process_design` crosses a mixture
design with a process-variable design, running every combination of
the two.

```pycon
>>> mixture_process_design(mixture, process)  # (1)!
```

1. `mixture` — an `(n1, q)` mixture design matrix (rows summing to 1).
   `process` — an `(n2, p)` process-variable design matrix, e.g. from
   `ff2n`.

```pycon
>>> mixture = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
>>> process = np.array([[-1.0], [1.0]])
>>> mixture_process_design(mixture, process)
array([[ 1. ,  0. , -1. ],
       [ 1. ,  0. ,  1. ],
       [ 0.5,  0.5, -1. ],
       [ 0.5,  0.5,  1. ],
       [ 0. ,  1. , -1. ],
       [ 0. ,  1. ,  1. ]])
```

!!! note
    The result has `n1 * n2` rows and `q + p` columns: the first `q`
    columns are mixture proportions, the last `p` columns are
    process-variable settings.

## More Information

For further reading on mixture experiment theory and analysis, see:

- [NIST Handbook Section 5.6 — Mixture Designs](https://www.itl.nist.gov/div898/handbook/pri/section6/pri6.htm)
- Cornell, J. A. (2002). *Experiments with Mixtures* (3rd ed.). Wiley.
- Scheffé, H. (1958). Experiments with mixtures. *J. Royal Stat. Soc. B*, 20(2), 344-360.
