"""
This code was originally published by the following individuals for use with
Scilab:
    Copyright (C) 2012 - 2013 - Michael Baudin
    Copyright (C) 2012 - Maria Christopoulou
    Copyright (C) 2010 - 2011 - INRIA - Michael Baudin
    Copyright (C) 2009 - Yann Collette
    Copyright (C) 2009 - CEA - Jean-Marc Martinez

    website: forge.scilab.org/index.php/p/scidoe/sourcetree/master/macros

Much thanks goes to these individuals. It has been converted to Python by
Abraham Lee.
"""

from __future__ import annotations

import collections
import itertools
import math
import re
import string

import numpy as np


__all__ = [
    "alias_vector_indices",
    "ff2n",
    "fracfact",
    "fracfact_aliasing",
    "fracfact_by_res",
    "fracfact_opt",
    "fullfact",
]


def fullfact(levels: np.ndarray) -> np.ndarray:
    """
    Create a general full-factorial design

    Parameters
    ----------
    levels : array-like
        An array of integers that indicate the number of levels of each input
        design factor.

    Returns
    -------
    mat : 2d-array
        The design matrix with coded levels 0 to k-1 for a k-level factor

    Examples
    --------
    ::

        >>> fullfact([2, 4, 3])
        array([[0., 0., 0.],
               [1., 0., 0.],
               [0., 1., 0.],
               [1., 1., 0.],
               [0., 2., 0.],
               [1., 2., 0.],
               [0., 3., 0.],
               [1., 3., 0.],
               [0., 0., 1.],
               [1., 0., 1.],
               [0., 1., 1.],
               [1., 1., 1.],
               [0., 2., 1.],
               [1., 2., 1.],
               [0., 3., 1.],
               [1., 3., 1.],
               [0., 0., 2.],
               [1., 0., 2.],
               [0., 1., 2.],
               [1., 1., 2.],
               [0., 2., 2.],
               [1., 2., 2.],
               [0., 3., 2.],
               [1., 3., 2.]])

    """
    n = len(levels)  # number of factors
    nb_lines = np.prod(levels)  # number of trial conditions
    H = np.zeros((nb_lines, n))

    level_repeat = 1
    range_repeat = np.prod(levels)
    for i in range(n):
        range_repeat //= levels[i]
        lvl = []
        for j in range(levels[i]):
            lvl += [j] * level_repeat
        rng = lvl * range_repeat
        level_repeat *= levels[i]
        H[:, i] = rng

    return H


def ff2n(n_factors: int) -> np.ndarray:
    """
    Create a 2-Level full-factorial design

    Parameters
    ----------
    n_factors : int
        The number of factors in the design.

    Returns
    -------
    mat : 2d-array
        The design matrix with coded levels -1 and 1

    Examples
    -------
    ::

        >>> ff2n(3)
        array([[-1., -1., -1.],
               [-1., -1.,  1.],
               [-1.,  1., -1.],
               [-1.,  1.,  1.],
               [ 1., -1., -1.],
               [ 1., -1.,  1.],
               [ 1.,  1., -1.],
               [ 1.,  1.,  1.]])

    Returns
    -------
    mat : 2d-array
        The full factorial design matrix
    """
    return np.array(list(itertools.product([-1.0, 1.0], repeat=n_factors)))


def validate_generator(n_factors: int, generator: str) -> str:
    """Validates the generator and thows an error if it is not valid.

    Returns
    -------
    str
        The validated generator string.

    Raises
    ------
    ValueError
        If the generator is invalid.
    """

    if len(generator.split(" ")) != n_factors:
        raise ValueError("Generator does not match the number of factors.")
    # clean it and transform it into a list
    generators = [item for item in re.split(r"\-|\s|\+", generator) if item]
    lengthes = [len(i) for i in generators]

    # Indices of single letters (main factors)
    idx_main = [i for i, item in enumerate(lengthes) if item == 1]

    if len(idx_main) == 0:
        raise ValueError("At least one unconfounded main factor is needed.")

    # Check that single letters (main factors) are unique
    if len(idx_main) != len({generators[i] for i in idx_main}):
        raise ValueError("Main factors are confounded with each other.")

    # Check that single letters (main factors) follow the alphabet
    if (
        "".join(sorted([generators[i] for i in idx_main]))
        != string.ascii_lowercase[: len(idx_main)]
    ):
        raise ValueError(
            "Use the letters "
            f"`{' '.join(string.ascii_lowercase[: len(idx_main)])}` "
            "for the main factors."
        )

    # Indices of letter combinations.
    idx_combi = [i for i, item in enumerate(generators) if item != 1]

    # check that main factors come before combinations
    if min(idx_combi) > max(idx_main):
        raise ValueError("Main factors have to come before combinations.")

    # Check that letter combinations are unique
    if len(idx_combi) != len({generators[i] for i in idx_combi}):
        raise ValueError("Generators are not unique.")

    # Check that only letters are used in the combinations
    # that are also single letters (main factors)
    if not all(
        set(item).issubset({generators[i] for i in idx_main})
        for item in [generators[i] for i in idx_combi]
    ):
        raise ValueError("Generators are not valid.")

    return generator


def fracfact(gen: str) -> np.ndarray:
    """
    Create a 2-level fractional-factorial design with a generator string.

    Parameters
    ----------
    gen : str
        A string, consisting of lowercase, uppercase letters or operators "-"
        and "+", indicating the factors of the experiment

    Returns
    -------
    H : 2d-array
        A m-by-n matrix, the fractional factorial design. m is 2^k, where k
        is the number of letters in ``gen``, and n is the total number of
        entries in ``gen``.

    Notes
    -----
    In ``gen`` we define the main factors of the experiment and the factors
    whose levels are the products of the main factors. For example, if

        gen = "a b ab"

    then "a" and "b" are the main factors, while the 3rd factor is the product
    of the first two. If we input uppercase letters in ``gen``, we get the same
    result. We can also use the operators "+" and "-" in ``gen``.

    For example, if

        gen = "a b -ab"

    then the 3rd factor is the opposite of the product of "a" and "b".

    The output matrix includes the two level full factorial design, built by
    the main factors of ``gen``, and the products of the main factors. The
    columns of ``H`` follow the sequence of ``gen``.

    For example, if

        gen = "a b ab c"

    then columns H[:, 0], H[:, 1], and H[:, 3] include the two level full
    factorial design and H[:, 2] includes the products of the main factors.

    Examples
    --------
    ::

        >>> fracfact("a b ab")
        array([[-1., -1.,  1.],
               [-1.,  1., -1.],
               [ 1., -1., -1.],
               [ 1.,  1.,  1.]])

        >>> fracfact("A B AB")
        array([[-1., -1.,  1.],
               [-1.,  1., -1.],
               [ 1., -1., -1.],
               [ 1.,  1.,  1.]])

        >>> fracfact("a b -ab c +abc")
        array([[-1., -1., -1., -1., -1.],
               [-1., -1., -1.,  1.,  1.],
               [-1.,  1.,  1., -1.,  1.],
               [-1.,  1.,  1.,  1., -1.],
               [ 1., -1.,  1., -1.,  1.],
               [ 1., -1.,  1.,  1., -1.],
               [ 1.,  1., -1., -1., -1.],
               [ 1.,  1., -1.,  1.,  1.]])

    """
    gen = validate_generator(
        n_factors=gen.count(" ") + 1, generator=gen.lower()
    )

    generators = [item for item in re.split(r"\-|\s|\+", gen) if item]
    lengthes = [len(i) for i in generators]

    # Indices of single letters (main factors)
    idx_main = [i for i, item in enumerate(lengthes) if item == 1]

    # Indices of letter combinations.
    idx_combi = [i for i, item in enumerate(generators) if item != 1]

    # Check if there are "-" operators in gen
    idx_negative = [
        i for i, item in enumerate(gen.split(" ")) if item[0] == "-"
    ]  # remove empty strings

    # Fill in design with two level factorial design
    H1 = ff2n(len(idx_main))
    H = np.zeros((H1.shape[0], len(lengthes)))
    H[:, idx_main] = H1

    # Recognize combinations and fill in the rest of matrix H2 with the proper
    # products
    for k in idx_combi:
        # For lowercase letters
        xx = np.array([ord(c) for c in generators[k]]) - 97

        H[:, k] = np.prod(H1[:, xx], axis=1)

    # Update design if gen includes "-" operator
    if len(idx_negative) > 0:
        H[:, idx_negative] *= -1

    # Return the fractional factorial design
    return H


# A design matrix has 2**k rows, so this cap also keeps `fracfact` from
# being handed a request it could not allocate.
_MAX_BASE_FACTORS = 26

# Effort one run size may spend before the search gives up, counted in set
# operations rather than in search nodes because the cost of a node grows
# with the design. Giving up falls through to the next run size, which costs
# runs but never resolution: designs that exist are found long before this,
# and only proving that one does *not* exist is expensive. A call gives up
# entirely after `_MAX_EXHAUSTED_SEARCHES` such run sizes, which keeps the
# worst case bounded for requests far outside the published catalogues.
_SEARCH_EFFORT_LIMIT = 1_000_000
_MAX_EXHAUSTED_SEARCHES = 3

# Maximum number of factors a regular resolution V design in 2**k runs can
# carry, i.e. the maximum length of a binary linear code with k parity
# checks and minimum distance 5. There is no closed form; entries up to
# k = 7 were confirmed by exhaustive search over GF(2)**k and the remainder
# come from the code tables (Grassl; Draper & Mitchell 1968 for k = 8).
_MAX_FACTORS_RESOLUTION_V = {
    1: 1,
    2: 2,
    3: 3,
    4: 5,
    5: 6,
    6: 8,
    7: 11,
    8: 17,
    9: 23,
    10: 33,
    11: 47,
    12: 65,
}


def fracfact_by_res(n: int, res: int) -> np.ndarray:
    r"""
    Create a 2-level fractional factorial design with `n` factors
    and resolution `res`.

    The design uses the smallest number of runs :math:`2^k` in which `n`
    factors can be accommodated at resolution `res` or better.

    Parameters
    ----------
    n : int
        The number of factors in the design.
    res : int
        Desired design resolution (typically 3, 4, or 5).

    Returns
    -------
    H : np.ndarray
        A m-by-`n` matrix, the fractional factorial design. m is the
        minimal amount of rows possible for creating a fractional
        factorial design matrix at resolution `res`

    Raises
    ------
    ValueError
        If `n` is below 2, if `res` is below 3, if the design would need
        more than 26 base factors, or if no design could be constructed.

    Notes
    -----
    The resolution of a design is defined as the length of the shortest
    word in the defining relation. The resolution describes the level of
    confounding between factors and interaction effects, where higher
    resolution indicates lower degree of confounding.

    For example, consider the 2^4-1-design defined by

        gen = "a b c ab"

    The factor "d" is defined by "ab" with defining relation I="abd", where
    I is the unit vector. In this simple example the shortest word is "abd"
    meaning that this is a resolution III-design.

    In practice resolution III-, IV- and V-designs are most commonly applied.

    * III: Main effects may be confounded with two-factor interactions.
    * IV: Main effects are unconfounded by two-factor interactions, but
          two-factor interactions may be confounded with each other.
    * V: Main effects unconfounded with up to four-factor interactions,
         two-factor interactions unconfounded with up to three-factor
         interactions. Three-factor interactions may be confounded with
         each other.

    A design of resolution `res` or *better* is returned; requesting
    resolution III for 4 factors, for instance, cannot be done in fewer
    than 8 runs, and the 8-run design happens to be of resolution IV.

    **Construction.** A regular design with `n` factors in :math:`2^k` runs
    is a choice of `n` distinct non-zero column vectors in
    :math:`\mathrm{GF}(2)^k`; a word of the defining relation is a subset of
    those columns summing to zero, so

    .. math::

        \text{resolution} \ge R \iff
        \text{no } R - 1 \text{ or fewer columns sum to zero.}

    Since the columns must span :math:`\mathrm{GF}(2)^k`, the `k` standard
    basis vectors are taken as the base factors ``a``, ``b``, ... and the
    remaining columns are searched for in order of increasing interaction
    order. This is what guarantees the resolution of the *whole* defining
    relation rather than of each generator taken on its own: a design whose
    generators are individually long enough can still have a short word among
    their products.

    The number of factors a run size can carry is
    :math:`2^k - 1` at resolution III (the saturated design) and
    :math:`2^{k-1}` at resolution IV; from resolution V upward there is no
    closed form and tabulated maxima are used.

    The run count is minimal at resolutions III and IV over the whole range
    the 26-base-factor cap allows, and at resolutions V and VI for at least
    20 factors, which covers every published catalogue. Beyond that - a
    resolution of VII or more, or several dozen factors at resolution V -
    the search may give up before it can prove that a run size is too small
    and move on to the next one. The returned design always meets the
    requested resolution; it may just use one more doubling than strictly
    necessary. Requests large enough that several run sizes in a row are
    inconclusive are rejected rather than ground out.

    References
    ----------
    Fries, A., & Hunter, W. G. (1980). Minimum aberration 2^(k-p) designs.
        *Technometrics*, 22(4), 601-608.
    Chen, H., & Cheng, C.-S. (2006). Doubling and projection: a method of
        constructing two-level designs of resolution IV.
        *The Annals of Statistics*, 34(1), 546-558.
    NIST/SEMATECH e-Handbook of Statistical Methods, Section 5.3.3.4.7 -
        "Summary tables of useful fractional factorial designs",
        https://www.itl.nist.gov/div898/handbook/pri/section3/pri3347.htm
    Montgomery, D. C. (2012). *Design and Analysis of Experiments* (8th ed.),
        Table 8.14. Wiley.

    Examples
    --------
    ::

        >>> fracfact_by_res(6, 3)
        array([[-1., -1., -1.,  1.,  1.,  1.],
               [-1., -1.,  1.,  1., -1., -1.],
               [-1.,  1., -1., -1.,  1., -1.],
               [-1.,  1.,  1., -1., -1.,  1.],
               [ 1., -1., -1., -1., -1.,  1.],
               [ 1., -1.,  1., -1.,  1., -1.],
               [ 1.,  1., -1.,  1., -1., -1.],
               [ 1.,  1.,  1.,  1.,  1.,  1.]])

    Seven factors fit in 8 runs at resolution III - the saturated
    :math:`2^{7-4}_{III}` design::

        >>> fracfact_by_res(7, 3).shape
        (8, 7)

    Four factors need those same 8 runs to reach resolution IV::

        >>> fracfact_by_res(4, 4).shape
        (8, 4)

    """
    if n < 2:
        raise ValueError("n must be at least 2")
    if res < 3:
        raise ValueError("resolution must be >= 3")

    # n distinct non-zero columns are needed whatever the resolution, and
    # GF(2)**k offers 2**k - 1 of them, so k >= n.bit_length().
    k_min = int(n).bit_length()
    if k_min > _MAX_BASE_FACTORS:
        raise ValueError("design requires more than 26 base factors")

    # A full factorial (k = n) always reaches the requested resolution, so
    # the search terminates for every n the base-factor cap allows.
    exhausted = 0
    for k in range(k_min, min(max(n, k_min), _MAX_BASE_FACTORS) + 1):
        if _max_factors_for_resolution(k, res) < n:
            continue
        columns, spent = _resolution_columns(k, n, res)
        if columns is not None:
            return fracfact(_generator_from_columns(columns, k))
        if spent > _SEARCH_EFFORT_LIMIT:
            exhausted += 1
            if exhausted == _MAX_EXHAUSTED_SEARCHES:
                break

    raise ValueError(
        f"no design with {n} factors at resolution {res} could be "
        "constructed; use a lower resolution or fewer factors"
    )


def _max_factors_for_resolution(k: int, res: int) -> int:
    """
    Maximum number of factors achievable with k base factors
    at the given resolution.

    Parameters
    ----------
    k : int
        Number of base factors.
    res : int
        Desired design resolution.

    Returns
    -------
    int
        The maximum number of factors achievable with k base factors at the
        given resolution. Exact for resolutions III and IV and for the
        tabulated resolution V run sizes; an upper bound otherwise.
    """
    if k < 1:
        return 0
    if res <= 3:
        # Saturated design: every non-zero vector of GF(2)**k is a column.
        return (1 << k) - 1
    if res == 4:
        # The odd-weight vectors of GF(2)**k, i.e. the fold-over of the
        # saturated resolution III design (Chen & Cheng 2006).
        return 1 << (k - 1)
    if res == 5 and k in _MAX_FACTORS_RESOLUTION_V:
        return _MAX_FACTORS_RESOLUTION_V[k]
    if res % 2 == 0:
        # A design of even resolution is a design of resolution res - 1 with
        # one extra parity column.
        return _max_factors_for_resolution(k - 1, res - 1) + 1

    # Sphere-packing (Hamming) bound: with res = 2t + 1 the sums of t or
    # fewer columns are distinct, so their number cannot exceed 2**k.
    t = (res - 1) // 2
    limit = 1 << k
    n = k
    while n < limit - 1 and sum(math.comb(n + 1, j) for j in range(t + 1)) <= (
        limit
    ):
        n += 1
    return n


def _candidate_columns(k: int) -> list[int]:
    """
    Interaction columns of GF(2)**k ordered by increasing interaction order.

    Parameters
    ----------
    k : int
        Number of base factors.

    Returns
    -------
    list of int
        Bit masks of every non-zero vector of weight two or more, ordered
        first by weight and then lexicographically, so that the generators
        chosen from them are the low-order interactions first.
    """
    return [
        sum(1 << i for i in combination)
        for order in range(2, k + 1)
        for combination in itertools.combinations(range(k), order)
    ]


def _resolution_columns(
    k: int, n: int, res: int, effort_limit: int = _SEARCH_EFFORT_LIMIT
) -> tuple[list[int] | None, int]:
    """
    Search for `n` columns of GF(2)**k with no short dependency.

    Parameters
    ----------
    k : int
        Number of base factors, so the design has ``2**k`` runs.
    n : int
        Number of factors the design must carry.
    res : int
        Requested resolution.
    effort_limit : int, optional
        Number of set operations the search may spend before giving up.

    Returns
    -------
    columns : list of int or None
        Bit masks of the chosen columns, basis vectors first, or None when
        no such design exists in ``2**k`` runs (or the effort ran out
        before it could be ruled out). Callers pass ``k <= n``; when
        ``n < k`` the surplus base factors are simply left out, so the
        design spans ``2**n`` runs rather than ``2**k``.
    effort : int
        Number of operations actually spent, so that a caller trying
        several run sizes can keep the whole call bounded.
    """
    basis = [1 << i for i in range(k)]
    if n <= k:
        # A full factorial in the first n base factors: no defining word at
        # all, so the resolution is unbounded.
        return basis[:n], 0

    # A candidate is rejected when it equals the sum of `depth` or fewer
    # chosen columns, which is exactly what would create a word shorter
    # than `res`. Only n - 1 columns are ever available to form such a sum.
    depth = min(res - 2, n - 1)
    # sums[j] holds the sums of exactly j chosen columns.
    sums: list[set[int]] = [{0}] + [set() for _ in range(depth)]

    def extend(column: int) -> list[tuple[int, set[int]]]:
        record = []
        for j in range(depth, 0, -1):
            new = {column ^ s for s in sums[j - 1]} - sums[j]
            if new:
                sums[j] |= new
                record.append((j, new))
        return record

    def undo(record: list[tuple[int, set[int]]]) -> None:
        for j, new in record:
            sums[j] -= new

    chosen = list(basis)
    for column in basis:
        extend(column)

    candidates = _candidate_columns(k)
    trail: list[tuple[int, list[tuple[int, set[int]]]]] = []
    index = 0
    effort = 0

    while True:
        if len(chosen) == n:
            return chosen, effort
        if effort > effort_limit:
            return None, effort

        placed = False
        last = len(candidates) - (n - len(chosen))
        while index <= last:
            column = candidates[index]
            effort += 1
            if any(column in sums[j] for j in range(1, depth + 1)):
                index += 1
                continue
            # Extending costs one pass over the largest layer of sums.
            effort += len(sums[depth - 1])
            trail.append((index, extend(column)))
            chosen.append(column)
            index += 1
            placed = True
            break

        if placed:
            continue

        if not trail:
            return None, effort
        # Resume the level below just past the column that was undone.
        index, record = trail.pop()
        undo(record)
        chosen.pop()
        index += 1


def _generator_from_columns(columns: list[int], k: int) -> str:
    """
    Turn GF(2)**k column masks into a generator string for `fracfact`.

    Parameters
    ----------
    columns : list of int
        Bit masks of the design columns.
    k : int
        Number of base factors.

    Returns
    -------
    str
        A generator string such as ``"a b c ab ac bc abc"``.
    """
    names = string.ascii_lowercase
    return " ".join(
        "".join(names[i] for i in range(k) if column >> i & 1)
        for column in columns
    )


################################################################################


def fracfact_opt(
    n_factors: int, n_erased: int, max_attempts: int = 0
) -> tuple[str, list[str], np.ndarray]:
    """
    Find the optimal generator string for a 2-level fractional-factorial design
    with the specified number of factors and erased factors.

    Parameters
    ----------
    n_factors : int
        The number of factors in the full factorial design
    n_erased : int
        The number of factors to "remove" to create the fractional design
    max_attempts : int
        The design is searched by exhaustive search, with the most "promising"
        combinations attempted first. For large designs it might be unfeasible
        to attempt all combinations.
        Posite values give the number of models to attemps. Zero or negative
        values indicate all combinations should be attempted.

    Returns
    -------
    gen : str
        A generator string in the format expected by fracfact() with the 2^k-p
        design, where k=n_factors and p=n_erased. The design disallows aliasing
        of main factors, and minimizes aliasing of low-order interactions.
    alias_map : list of str
        The map of aliases that the design inflicts.
        More details in fracfact_aliasing().
    alias_vector : 1d numpy.array
        The vector with the cost of the design in term of aliasings.
        More details in fracfact_aliasing().

    Raises
    ------
    ValueError
        If the number of factors is invalid or too many factors are erased.

    Notes
    -----
    A :math:`2^{k-p}` design is built from :math:`m = k - p` main factors
    spanning :math:`2^m` runs. Every erased factor is assigned to one of the
    interaction columns of those main factors, and there are exactly

    .. math::

        \\sum_{j=2}^{m} \\binom{m}{j} = 2^m - m - 1

    such columns. The design is therefore feasible if and only if
    :math:`p \\le 2^m - m - 1`. The boundary case
    :math:`p = 2^m - m - 1` consumes every interaction column and yields the
    *saturated* resolution III design with :math:`k = 2^m - 1` factors in
    :math:`2^m` runs, e.g. the :math:`2^{7-4}_{III}` design in 8 runs or the
    :math:`2^{15-11}_{III}` design in 16 runs.

    Examples
    --------
    ::

        >>> gen, _alias_map, _alias_vector = fracfact_opt(7, 4)
        >>> gen
        'a b c abc bc ac ab'
        >>> fracfact(gen).shape
        (8, 7)

    """

    def n_comb(n: int, k: int) -> int:
        if k <= 0 or n <= 0 or k > n:
            return 0
        return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

    if n_factors > 20:
        raise ValueError("Design too big, use 20 factors or less")

    if n_factors < 2:
        raise ValueError("Design too small")

    if n_erased < 0:
        raise ValueError("Number of erased factors must be non-negative")

    n_main_factors = n_factors - n_erased
    # Number of interaction columns available to carry the erased factors:
    # sum_{j=2}^{m} C(m, j) = 2^m - m - 1 for m main factors.
    n_aliases = sum(
        n_comb(n_main_factors, n) for n in range(2, n_main_factors + 1)
    )

    if n_erased > n_aliases:
        raise ValueError(
            "Too many erased factors to create aliasing: "
            f"{n_main_factors} main factors provide only {n_aliases} "
            f"interaction columns, but {n_erased} are needed"
        )

    all_names = string.ascii_lowercase
    # factors = range(n_factors)
    main_factors = range(n_main_factors)
    main_design = " ".join([all_names[f] for f in main_factors])
    aliases = itertools.chain.from_iterable(
        itertools.combinations(main_factors, n)
        for n in range(2, n_main_factors + 1)
    )

    aliases = sorted(list(aliases), key=lambda a: (len(a), a), reverse=True)
    best_design = None
    best_map = []
    best_vector = np.repeat(n_factors, n_factors)
    design_shape = (2**n_main_factors, n_factors)
    all_combinations = itertools.combinations(aliases, n_erased)
    all_combinations = (
        all_combinations
        if max_attempts <= 0
        else itertools.islice(all_combinations, 0, max_attempts)
    )

    for aliasing in all_combinations:
        aliasing_design = ["".join([all_names[f] for f in a]) for a in aliasing]
        # ``aliasing_design`` is empty when n_erased == 0; joining the parts
        # this way keeps the generator free of a trailing separator.
        complete_design = " ".join([main_design, *aliasing_design])
        design = fracfact(complete_design)
        if design.shape != design_shape:
            raise ValueError(
                f"Design shape {design.shape} does not "
                f"match expected {design_shape}"
            )
        alias_map, alias_vector = fracfact_aliasing(design)
        if list(alias_vector) < list(best_vector):
            best_design = complete_design
            best_map = alias_map
            best_vector = alias_vector

    return best_design, best_map, best_vector


def fracfact_aliasing(design: np.ndarray) -> tuple[list[str], np.ndarray]:
    """
    Find the aliasings in a design, given the contrasts.

    Parameters
    ----------
    design : numpy 2d array
        A design like those returned by fracfact()

    Returns
    -------
    alias_map : list of str
        The map of aliases that the design inflicts. Each string in the list is
        a set of factors and interactions that are aliased among themselves, in
        the format a = bcd = def etc. If there is no aliasing (n_erased=0) the
        map simply lists all factors and interactions.
    alias_vector : 1d numpy.array
        The vector with the cost of the design in term of aliasings. Each cell
        in the array counts the number of aliasing between factors/interactions
        of size i and of size j, as given by alias_vector_indices().

        The alias cost vector can be turned into a more explicit upper
        triangular cost matrix with the idiom:
        alias_matrix = np.zeros((n_factors, n_factors,))
        alias_matrix[alias_vector_indices(n_factors)] = alias_vector

        The entry in alias_matrix[i,j] (i<=j) shows how many aliasings where
        created among i-th order interactions and j-th order interactions.

    Raises
    ------
    ValueError
        If the design is too large (more than 20 factors), or if it contains
        entries other than -1 and +1.

    Notes
    -----
    Every one of the :math:`2^{k} - 1` factors and interactions is reduced to
    the bit pattern of the rows in which its contrast is negative. Because the
    columns only hold :math:`\\pm 1`, the elementwise product of a set of
    columns is the bitwise *exclusive or* of their patterns, so the whole
    alias structure is found with integer arithmetic instead of one NumPy
    reduction per subset.
    """
    _n_rounds, n_factors = design.shape

    if n_factors > 20:
        raise ValueError("Design too big, use 20 factors or less")

    if not np.all(np.abs(design) == 1):
        raise ValueError("Design must only contain the levels -1 and +1")

    all_names = string.ascii_lowercase
    factors = range(n_factors)

    # Bit i of column_bits[j] is set when design[i, j] is at its low level.
    column_bits = [
        sum(1 << int(row) for row in np.flatnonzero(design[:, j] < 0))
        for j in factors
    ]

    all_combinations = itertools.chain.from_iterable(
        itertools.combinations(factors, n) for n in range(1, n_factors + 1)
    )
    aliases = {}

    for combination in all_combinations:
        contrast = 0
        for factor in combination:
            contrast ^= column_bits[factor]
        aliases.setdefault(contrast, []).append(combination)

    aliases_list = []
    for alias in aliases.values():
        aliases_list.append(sorted(alias, key=lambda a: (len(a), a)))
    aliases_list = sorted(
        aliases_list, key=lambda lst: ([len(a) for a in lst], lst)
    )

    aliases_readable = []
    alias_matrix = np.zeros((n_factors, n_factors))

    for alias in aliases_list:
        alias_readable = " = ".join([
            "".join([all_names[f] for f in a]) for a in alias
        ])
        aliases_readable.append(alias_readable)
        # Count the aliased pairs by interaction order rather than iterating
        # over every pair: an alias chain holding `count[s]` terms of order
        # `s` contributes count[s] * count[t] pairs of orders (s, t), and
        # count[s] * (count[s] - 1) / 2 pairs of equal order s.
        counts = collections.Counter(len(a) for a in alias)
        orders = sorted(counts)
        for i, low in enumerate(orders):
            alias_matrix[low - 1, low - 1] += (
                counts[low] * (counts[low] - 1) // 2
            )
            for high in orders[i + 1 :]:
                alias_matrix[low - 1, high - 1] += counts[low] * counts[high]

    alias_vector = alias_matrix[alias_vector_indices(n_factors)]

    return aliases_readable, alias_vector


def alias_vector_indices(n_factors: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the indexes to convert the alias_vector into a square matrix and
    vice-versa.

    Parameters
    ----------
    n_factors : int
        The number of factors in the full factorial design

    Returns
    -------
    rows : 1d numpy array
    cols : 1d numpy.array
        Rows and columns of the indices of the upper triangular square matrix
        with n_factor rows/columns. This function returns a different indice
        order than numpy.triu_indices, as it puts the indices representing the
        most serious aliasings first, to help in the optimization procedure.

    Raises
    ------
    ValueError
        If the design is too large (more than 20 factors).
    """
    if n_factors > 20:
        raise ValueError("Design too big, use 20 factors or less")

    indices = list(itertools.combinations_with_replacement(range(n_factors), 2))
    indices = sorted(indices, key=max)

    rows = np.asarray([i[0] for i in indices])
    cols = np.asarray([i[1] for i in indices])

    return rows, cols
