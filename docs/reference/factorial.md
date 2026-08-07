In this section, the following kinds of factorial designs will be described:

- [General Full-Factorial](#general-full-factorial-fullfact)
- [2-Level Full-Factorial](#2-level-full-factorial-ff2n)
- [2-Level Fractional-Factorial](#2-level-fractional-factorial-fracfact)
- [Plackett-Burman](#plackett-burman)
- [Generalized Subset Design](#generalized-subset-design)
- [John's 3/4 Fractional Factorial](#johns-34-fractional-factorial-john_three_quarter_design)
- [Latin Square Designs](#latin-square-designs-latin_square)
- [Graeco-Latin Square Designs](#graeco-latin-square-designs-graeco_latin_square)
- [Hyper-Graeco-Latin Square Designs](#hyper-graeco-latin-square-designs-hyper_graeco_latin_square)
- [Blocking a Full Factorial Design](#blocking-a-full-factorial-design-block_full_factorial)

!!! note
    All available designs can be accessed after a simple import statement:
    ```pycon
    >>> from pydoe import (
    ...     fullfact,
    ...     ff2n,
    ...     fracfact,
    ...     pbdesign,
    ...     gsd,
    ...     john_three_quarter_design,
    ...     latin_square,
    ...     graeco_latin_square,
    ...     hyper_graeco_latin_square,
    ...     block_full_factorial,
    ... )
    ```

## General Full-Factorial (`fullfact`)

This kind of design offers full flexibility as to the number of discrete levels for each factor in the design. Its usage is simple:

```pycon
>>> fullfact(levels)  # (1)!
>>> fullfact([2, 3])
array([[ 0.,  0.],
       [ 1.,  0.],
       [ 0.,  1.],
       [ 1.,  1.],
       [ 0.,  2.],
       [ 1.,  2.]])
```

1. `levels` is array of integers.

As can be seen in the output, the design matrix has as many columns as
items in the input array.

## 2-Level Full Factorial (`ff2n`)

This function is a convenience wrapper to `fullfact` that forces all the
factors to have two levels each, you simple tell it how many factors to
create a design for.

```pycon
>>> ff2n(3)
array([[-1., -1., -1.],
       [ 1., -1., -1.],
       [-1.,  1., -1.],
       [ 1.,  1., -1.],
       [-1., -1.,  1.],
       [ 1., -1.,  1.],
       [-1.,  1.,  1.],
       [ 1.,  1.,  1.]])
```

## 2-Level Fractional-Factorial (`fracfact`)

This function requires a little more knowledge of how the *confounding*
will be allowed (this means that some factor effects get muddled with
other interaction effects, so it's harder to distinguish between them).

Let's assume that we just can't afford (for whatever reason) the number
of runs in a *full-factorial* design. We can systematically decide on a
*fraction of the full-factorial* by allowing some of the factor *main
effects* to be confounded with other factor *interaction effects*. This
is done by defining an *alias* structure that defines, symbolically,
these interactions. These alias structures are written like $C = AB$ or
$I = ABC$, or $AB = CD$, etc. These define how one column is related to
the others.

For example, the alias $C = AB$ or $I = ABC$ indicate that there are
three factors ($A$, $B$, and $C$) and that the main effect of factor
$C$ is confounded with the interaction effect of the product $AB$, and by
extension, $A$ is confounded with $BC$ and $B$ is confounded with $AC$. A full-
factorial design with these three factors results in a design matrix with
8 runs, but we will assume that we can only afford 4 of those runs. To
create this *fractional* design, we need a matrix with three columns, one
for $A$, $B$, and $C$, only now where the levels in the $C$ column is created by
the product of the $A$ and $B$ columns.

The input to `fracfact` is a generator string of symbolic characters
(lowercase or uppercase, but not both) separated by spaces, like::

```pycon
>>> gen = "a b ab"

```

This design would result in a 3-column matrix, where the third column is
implicitly defined as `"c = ab"`. This means that the factor in the third
column is confounded with the interaction of the factors in the first two
columns. The design ends up looking like this;

```pycon
>>> fracfact("a b ab")
array([[-1., -1.,  1.],
       [ 1., -1., -1.],
       [-1.,  1., -1.],
       [ 1.,  1.,  1.]])
```

Fractional factorial designs are usually specified using the notation
$2^{(k-p)}$, where $k$ is the number of columns and $p$ is the number
of effects that are confounded. In terms of *resolution* level, higher is
"better". The above design would be considered a $2^{(3-1)}$
fractional factorial design, a 1/2-fraction design, or a *Resolution III*
design (since the smallest alias $I=ABC$ has three terms on the right-hand
side). Another common design is a Resolution III, $2^{(7-4)}$
fractional factorial and would be created using the following string
generator.

```pycon
>>> fracfact("a b ab c ac bc abc")
array([[-1., -1.,  1., -1.,  1.,  1., -1.],
       [ 1., -1., -1., -1., -1.,  1.,  1.],
       [-1.,  1., -1., -1.,  1., -1.,  1.],
       [ 1.,  1.,  1., -1., -1., -1., -1.],
       [-1., -1.,  1.,  1., -1., -1.,  1.],
       [ 1., -1., -1.,  1.,  1., -1., -1.],
       [-1.,  1., -1.,  1., -1.,  1., -1.],
       [ 1.,  1.,  1.,  1.,  1.,  1.,  1.]])
```

More sophisticated generator strings can be created using the "`+`" and
"`-`" operators. The "`-`" operator swaps the levels of that column like
this:

```pycon
>>> fracfact("a b -ab")
array([[-1., -1., -1.],
       [ 1., -1.,  1.],
       [-1.,  1.,  1.],
       [ 1.,  1., -1.]])
```

In order to reduce confounding, we can utilize the `fold` function:

```pycon
>>> m = fracfact("a b ab")
>>> fold(m)
array([[-1., -1.,  1.],
       [ 1., -1., -1.],
       [-1.,  1., -1.],
       [ 1.,  1.,  1.],
       [ 1.,  1., -1.],
       [-1.,  1.,  1.],
       [ 1., -1.,  1.],
       [-1., -1., -1.]])
```

Applying the fold to all columns in the design breaks the alias chains
between every *main factor and two-factor interactions*. This means that
we can then estimate *all the main effects clear of any two-factor
interactions*. Typically, when all columns are folded, this "upgrades"
the resolution of the design.

By default, `fold` applies the level swapping to all columns, but we can
fold specific columns (first column = 0), if desired, by supplying an array
to the keyword `columns`:

```pycon
>>> fold(m, columns=[2])
array([[-1., -1.,  1.],
       [ 1., -1., -1.],
       [-1.,  1., -1.],
       [ 1.,  1.,  1.],
       [-1., -1., -1.],
       [ 1., -1.,  1.],
       [-1.,  1.,  1.],
       [ 1.,  1., -1.]])
```

Another way to reduce confounding it to scan several (or all) available
fractional designs and pick the one that has less confounding. The function
`fracfact_opt` performs just that. For a $2^{k-p}$ fractional factorial the
function scans all generators that create at most $2^{k-p}$ experiments, and pick
the one that has confounding on interactions of order as high as possible:

```pycon
>>> design, alias_map, alias_cost = fracfact_opt(6, 2)
>>> design
"a b c d bcd acd"
>>> print("\n".join(alias_map))
a = bef = cdf = abcde
b = aef = cde = abcdf
c = adf = bde = abcef
d = acf = bce = abdef
e = abf = bcd = acdef
f = abe = acd = bcdef
af = be = cd = abcdef
ab = ef = acde = bcdf
ac = df = abde = bcef
ad = cf = abce = bdef
ae = bf = abcd = cdef
bc = de = abdf = acef
bd = ce = abcf = adef
abc = ade = bdf = cef
abd = ace = bcf = def
abef = acdf = bcde
```

You can generate the human-readable `alias_map` of any design with the function
`fracfact_aliasing`:

```pycon
>>> print("\n".join(fracfact_aliasing(fracfact("a b ab"))[0]))
a = bc
b = ac
c = ab
abc
```

!!! note
    Care should be taken to decide the appropriate alias structure for
    your design and the effects that folding has on it.


### 2-Level Fractional-Factorial specified by resolution (`fracfact_by_res`)

This function constructs a minimal design at given resolution. It does so
by constructing a generator string with a minimal number of base factors
and passes it to `fracfact`. This approach favors convenience over
fine-grained control over which factors that are confounded.

To construct a 6-factor, resolution III-design, `fractfact_by_res`
is used like this;

```pycon
>>> fracfact_by_res(6, 3)
array([[-1., -1., -1.,  1.,  1.,  1.],
       [ 1., -1., -1., -1., -1.,  1.],
       [-1.,  1., -1., -1.,  1., -1.],
       [ 1.,  1., -1.,  1., -1., -1.],
       [-1., -1.,  1.,  1., -1., -1.],
       [ 1., -1.,  1., -1.,  1., -1.],
       [-1.,  1.,  1., -1., -1.,  1.],
       [ 1.,  1.,  1.,  1.,  1.,  1.]])
```

Available Factorial Designs (with Resolution)

<div class="doe-table-wrapper">

<table class="doe-table">
  <thead>
    <tr>
      <th>Number of Runs</th>
      <th colspan="14">Number of Factors</th>
    </tr>
    <tr>
      <th></th>
      <th>2</th><th>3</th><th>4</th><th>5</th>
      <th>6</th><th>7</th><th>8</th><th>9</th>
      <th>10</th><th>11</th><th>12</th><th>13</th>
      <th>14</th><th>15</th>
    </tr>
  <tbody>

    <tr>
      <th>4</th>
      <td class="full">$2^2$</td><td class="res3">$2^{3-1}$</td>
      <td class="na"></td><td class="na"></td><td class="na"></td><td class="na"></td>
      <td class="na"></td><td class="na"></td><td class="na"></td><td class="na"></td>
      <td class="na"></td><td class="na"></td><td class="na"></td><td class="na"></td>
    </tr>

    <tr>
      <th>8</th>
      <td class="na"></td><td class="full">$2^3$</td>
      <td class="res4">$2^{4-1}$</td><td class="res3">$2^{5-2}$</td>
      <td class="res3">$2^{6-3}$</td><td class="res3">$2^{7-4}$</td>
      <td class="na"></td><td class="na"></td><td class="na"></td><td class="na"></td>
      <td class="na"></td><td class="na"></td><td class="na"></td><td class="na"></td>
    </tr>

    <tr>
      <th>16</th>
      <td class="na"></td><td class="na"></td>
      <td class="full">$2^4$</td><td class="res5">$2^{5-1}$</td>
      <td class="res4">$2^{6-2}$</td><td class="res4">$2^{7-3}$</td>
      <td class="res4">$2^{8-4}$</td><td class="res3">$2^{9-5}$</td>
      <td class="res3">$2^{10-6}$</td><td class="res3">$2^{11-7}$</td>
      <td class="res3">$2^{12-8}$</td><td class="res3">$2^{13-9}$</td>
      <td class="res3">$2^{14-10}$</td><td class="res3">$2^{15-11}$</td>
    </tr>

    <tr>
      <th>32</th>
      <td class="na"></td><td class="na"></td><td class="na"></td>
      <td class="full">$2^5$</td><td class="res5">$2^{6-1}$</td>
      <td class="res4">$2^{7-2}$</td><td class="res4">$2^{8-3}$</td>
      <td class="res4">$2^{9-4}$</td><td class="res4">$2^{10-5}$</td>
      <td class="res4">$2^{11-6}$</td><td class="res4">$2^{12-7}$</td>
      <td class="res4">$2^{13-8}$</td><td class="res4">$2^{14-9}$</td>
      <td class="res4">$2^{15-10}$</td>
    </tr>

    <tr>
      <th>64</th>
      <td class="na"></td><td class="na"></td><td class="na"></td><td class="na"></td>
      <td class="full">$2^6$</td><td class="res5">$2^{7-1}$</td>
      <td class="res5">$2^{8-2}$</td><td class="res4">$2^{9-3}$</td>
      <td class="res4">$2^{10-4}$</td><td class="res4">$2^{11-5}$</td>
      <td class="res4">$2^{12-6}$</td><td class="res4">$2^{13-7}$</td>
      <td class="res4">$2^{14-8}$</td><td class="res4">$2^{15-9}$</td>
    </tr>

    <tr>
      <th>128</th>
      <td class="na"></td><td class="na"></td>
      <td class="na"></td><td class="na"></td>
      <td class="na"></td>
      <td class="full">$2^7$</td><td class="res5">$2^{8-1}$</td>
      <td class="res5">$2^{9-2}$</td><td class="res5">$2^{10-3}$</td>
      <td class="res5">$2^{11-4}$</td><td class="res4">$2^{12-5}$</td>
      <td class="res4">$2^{13-6}$</td><td class="res4">$2^{14-7}$</td>
      <td class="res4">$2^{15-8}$</td>
    </tr>

  </tbody>
</table>

<div class="doe-legend">
  <div><span class="legend full"></span> Full Factorial Design</div>
  <div><span class="legend res5"></span> Resolution V (or Higher) Design</div>
  <div><span class="legend res4"></span> Resolution IV Design</div>
  <div><span class="legend res3"></span> Resolution III Design</div>
  <div><span class="legend na"></span> Not Available</div>
</div>

</div>

## Plackett-Burman (`pbdesign`) {#plackett-burman}

Another way to generate fractional-factorial designs is through the use
of **Plackett-Burman** designs. These designs are unique in that the
number of trial conditions (rows) expands by multiples of four (e.g. 4,
8, 12, etc.). The max number of columns allowed before a design increases
the number of rows is always one less than the next higher multiple of four.

For example, I can use up to 3 factors in a design with 4 rows:

```pycon
>>> pbdesign(3)
array([[-1., -1.,  1.],
       [ 1., -1., -1.],
       [-1.,  1., -1.],
       [ 1.,  1.,  1.]])
```

But if I want to do 4 factors, the design needs to increase the number
of rows up to the next multiple of four (8 in this case):

```pycon
>>> pbdesign(4)
array([[-1., -1.,  1., -1.],
       [ 1., -1., -1., -1.],
       [-1.,  1., -1., -1.],
       [ 1.,  1.,  1., -1.],
       [-1., -1.,  1.,  1.],
       [ 1., -1., -1.,  1.],
       [-1.,  1., -1.,  1.],
       [ 1.,  1.,  1.,  1.]])
```

Thus, an 8-run Plackett-Burman design can handle up to (8 - 1) = 7 factors.

As a side note, It just so happens that the Plackett-Burman and $2^{(7-4)}$
fractional factorial design are identical:

```pycon
>>> np.all(pbdesign(7) == fracfact("a b ab c ac bc abc"))
True
```

## Generalized Subset Design (`gsd`) {#generalized-subset-design}

GSD is a generalization of traditional fractional factorial designs to problems
where factors can have more than two levels.

In many application problems, factors can have categorical or quantitative factors
on more than two levels. Previous reduced designs have not been able to deal with
such types of problems. Full multi-level factorial designs can handle such problems
but are however not economical regarding the number of experiments.

The GSD provide balanced designs in multi-level experiments with the number of
experiments reduced by a user-specified reduction factor. Complementary reduced
designs are also provided analogous to fold-over in traditional fractional factorial
designs.

An example with three factors using three, four and six levels respectively reduced
with a factor 4:

```pycon
>>> gsd([3, 4, 6], 4)
array([[0, 0, 0],
       [0, 0, 4],
       [0, 1, 1],
       [0, 1, 5],
       [0, 2, 2],
       [0, 3, 3],
       [1, 0, 1],
       [1, 0, 5],
       [1, 1, 2],
       [1, 2, 3],
       [1, 3, 0],
       [1, 3, 4],
       [2, 0, 2],
       [2, 1, 3],
       [2, 2, 0],
       [2, 2, 4],
       [2, 3, 1],
       [2, 3, 5]])
```

## John's 3/4 Fractional Factorial (`john_three_quarter_design`) {#johns-34-fractional-factorial-john_three_quarter_design}

John's three-quarter design is a *semifoldover* design that uses exactly
$\tfrac{3}{4} \times 2^k$ runs — more runs than a half-fraction but fewer
than the full $2^k$ factorial. It was introduced by P. W. M. John (1971)
and is particularly useful when a full mirror-image foldover doubles the
cost but a half-fraction leaves too many interactions aliased.

The construction proceeds in four steps:

1. **Half-fraction** — build the $2^{k-1}$ design generated by $X_k = X_1 X_2 \cdots X_{k-1}$.
2. **Select** the $2^{k-2}$ rows where the chosen factor (`fold_on`) is at $-1$.
3. **Flip** the sign of `fold_on` in those rows ($-1 \to +1$).
4. **Append** the flipped rows to the base half-fraction.

The resulting design has $2^{k-1} + 2^{k-2} = 3 \times 2^{k-2}$ runs and
de-aliases all two-factor interactions that involve the folded factor.

```pycon
>>> john_three_quarter_design(k)  # (1)!
>>> john_three_quarter_design(k, fold_on=j)  # (2)!
```

1. `k` is the number of two-level factors (≥ 3). Folds over the first factor by default.
2. `fold_on` is the 1-based column index of the factor to fold over.

A four-factor design uses 12 runs instead of the full $2^4 = 16$:

```pycon
>>> john_three_quarter_design(4)
array([[-1., -1., -1., -1.],
       [ 1., -1., -1.,  1.],
       [-1.,  1., -1.,  1.],
       [ 1.,  1., -1., -1.],
       [-1., -1.,  1.,  1.],
       [ 1., -1.,  1., -1.],
       [-1.,  1.,  1., -1.],
       [ 1.,  1.,  1.,  1.],
       [ 1., -1., -1., -1.],
       [ 1.,  1., -1.,  1.],
       [ 1., -1.,  1.,  1.],
       [ 1.,  1.,  1., -1.]])
```

The first 8 rows are the $2^{4-1}$ half-fraction (generator $X_4 = X_1 X_2 X_3$);
rows 9–12 are the semifoldover augment on $X_1$.

Folding on a different factor breaks different alias pairs — choose the
factor whose two-factor interactions are most important to estimate:

```pycon
>>> john_three_quarter_design(4, fold_on=2).shape
(12, 4)
>>> john_three_quarter_design(5).shape
(24, 5)
```

!!! note
    The generator column is always $X_k = X_1 X_2 \cdots X_{k-1}$, which
    corresponds to the highest-resolution defining word $I = X_1 X_2 \cdots X_k$.
    The `fold_on` parameter (default ``1``) selects which factor's two-factor
    interactions are de-aliased by the semifoldover.  See NIST Handbook
    Section 5.5.7 for the full alias analysis.

## Latin Square Designs (`latin_square`) {#latin-square-designs-latin_square}

A **Latin square** of order $n$ is an $n \times n$ array filled with $n$
different symbols, each occurring exactly once in each row and exactly
once in each column. Latin square designs are used to remove the effect
of two nuisance factors (rows and columns) while studying a single
treatment factor with $n$ levels, using only $n^2$ runs instead of the
$n^3$ runs a full factorial would require.

```pycon
>>> latin_square(n)  # (1)!
```

1. `n` — order of the square (number of levels), must be at least 2.

`latin_square` builds the square using the cyclic construction
$L[i, j] = (i + j) \bmod n$, which guarantees every row and column is a
permutation of $0, 1, \ldots, n-1$:

```pycon
>>> latin_square(4)
array([[0, 1, 2, 3],
       [1, 2, 3, 0],
       [2, 3, 0, 1],
       [3, 0, 1, 2]])
```

!!! note
    Row $i$ corresponds to a level of the first nuisance (blocking)
    factor, column $j$ to a level of the second nuisance factor, and the
    entry $L[i, j]$ gives the treatment level to apply in that cell.

## Graeco-Latin Square Designs (`graeco_latin_square`) {#graeco-latin-square-designs-graeco_latin_square}

A **Graeco-Latin square** superimposes two orthogonal Latin squares,
allowing a *third* nuisance factor to be removed while still studying a
single treatment factor in only $n^2$ runs. Two Latin squares are
*orthogonal* if, when superimposed, every ordered pair of symbols occurs
exactly once.

```pycon
>>> latin, graeco = graeco_latin_square(n)  # (1)!
```

1. `n` — order of the squares; must be a **prime number** greater than 2.

```pycon
>>> latin, graeco = graeco_latin_square(3)
>>> latin
array([[0, 1, 2],
       [1, 2, 0],
       [2, 0, 1]])
>>> graeco
array([[0, 2, 1],
       [1, 0, 2],
       [2, 1, 0]])
```

!!! note
    Both squares are constructed with $L_a[i, j] = (i + a j) \bmod n$ for
    $a \in \{1, 2\}$. For prime $n$ this guarantees orthogonality, but it
    also means `graeco_latin_square` only supports prime $n > 2$ (e.g.
    3, 5, 7, 11, ...).

## Hyper-Graeco-Latin Square Designs (`hyper_graeco_latin_square`) {#hyper-graeco-latin-square-designs-hyper_graeco_latin_square}

A **hyper-Graeco-Latin square** extends the Graeco-Latin square idea to
four or more mutually orthogonal Latin squares, removing that many
nuisance factors while studying a single treatment factor in $n^2$ runs.

```pycon
>>> squares = hyper_graeco_latin_square(n, k)  # (1)!
```

1. `n` — order of the squares (a **prime number** greater than 2); `k` —
   number of mutually orthogonal Latin squares, with $2 \le k \le n - 1$.

```pycon
>>> squares = hyper_graeco_latin_square(5, 3)
>>> squares.shape
(3, 5, 5)
>>> squares[0]
array([[0, 1, 2, 3, 4],
       [1, 2, 3, 4, 0],
       [2, 3, 4, 0, 1],
       [3, 4, 0, 1, 2],
       [4, 0, 1, 2, 3]])
```

!!! note
    Each square is built with $L_a[i, j] = (i + a j) \bmod n$ for
    $a = 1, \ldots, k$. For prime $n$, a complete set of $n - 1$ mutually
    orthogonal Latin squares exists, so `k` must satisfy
    $2 \le k \le n - 1$.

## Blocking a Full Factorial Design (`block_full_factorial`) {#blocking-a-full-factorial-design-block_full_factorial}

When the runs of a $2^k$ factorial design cannot all be carried out
under homogeneous conditions, the design can be split into $2^p$
blocks by confounding one or more high-order interactions with the
block effect.

```pycon
>>> design, blocks = block_full_factorial(k, generators)  # (1)!
```

1. `k` — number of factors (≥ 2). `generators` — a list of tuples of
   0-based factor indices; each tuple's interaction column defines one
   block contrast, giving $2^{\text{len(generators)}}$ blocks.

Confound the three-factor interaction `ABC` with two blocks:

```pycon
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
```

!!! note
    The chosen interactions (and their generalized interactions) become
    completely confounded with block effects, so they should be
    interactions believed to be negligible.

## More Information

If the user needs more information about appropriate designs, please
consult the following articles on Wikipedia:

- [`Factorial designs`](http://en.wikipedia.org/wiki/Factorial_experiment)
- [`Plackett-Burman designs`](http://en.wikipedia.org/wiki/Plackett-Burman_design)

There is also a wealth of information on the [`NIST`](http://www.itl.nist.gov/div898/handbook/pri/pri.htm) website about the
various design matrices that can be created as well as detailed information
about designing/setting-up/running experiments in general.
