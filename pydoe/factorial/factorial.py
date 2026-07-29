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

import itertools
import math
import re
import string

import numpy as np
from scipy.special import binom


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


def fracfact_by_res(n: int, res: int) -> np.ndarray:
    """
    Create a 2-level fractional factorial design with `n` factors
    and resolution `res`.

    Given a number of factors `n` and a desired resolution `res`, this function
    determines a reasonable number of base factors (runs = 2^k) using standard
    DOE tables where possible, and a calculated fallback otherwise. Generators
    are then constructed to reach `n` total factors while respecting the
    resolution constraints.

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
        If the inputs are invalid or the requested design cannot be constructed.

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
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    if res < 3:
        raise ValueError("resolution must be >= 3")

    # trying the standard DOE tables first
    k = _get_base_factors_from_table(n, res)
    if k is None:
        k = _calculate_min_base_factors(n, res)

    if k > 26:
        raise ValueError("design requires more than 26 base factors")

    #  validation
    max_n = _max_factors_for_resolution(k, res)
    if n > max_n:
        raise ValueError(
            f"{n} factors not possible at resolution {res} with {2**k} runs "
            f"(max supported: {max_n})"
        )

    # Base factor names: a, b, c, ...
    base = list(string.ascii_lowercase[:k])

    # Generator terms must be of order >= (res - 1)
    combos = (
        "".join(c)
        for r in range(res - 1, len(base) + 1)
        for c in itertools.combinations(base, r)
    )
    generated = list(itertools.islice(combos, n - k))

    gen_str = " ".join(base + generated)
    return fracfact(gen_str)


def _get_base_factors_from_table(n: int, res: int) -> int | None:
    """
    Lookup k (base factors) from commonly used fractional factorial tables.
    Returns None if the (n, res) pair is not in the table.

    Parameters
    ----------
    n : int
        Total number of factors.
    res : int
        Desired design resolution.

    Returns
    -------
    int or None
        The number of base factors k if found, otherwise None.
    """
    table = {
        # Resolution III
        (3, 3): 2,
        (4, 3): 3,
        (5, 3): 3,
        (6, 3): 3,
        (7, 3): 4,
        (8, 3): 4,
        (9, 3): 4,
        (10, 3): 4,
        (11, 3): 4,
        (12, 3): 4,
        (13, 3): 4,
        (14, 3): 4,
        (15, 3): 4,
        # Resolution IV
        (3, 4): 3,
        (4, 4): 4,
        (5, 4): 5,
        (6, 4): 5,
        (7, 4): 5,
        (8, 4): 5,
        (9, 4): 6,
        (10, 4): 6,
        (11, 4): 6,
        (12, 4): 6,
        (13, 4): 6,
        (14, 4): 6,
        (15, 4): 6,
        # Resolution V
        (4, 5): 5,
        (5, 5): 5,
        (6, 5): 6,
        (7, 5): 6,
        (8, 5): 7,
        (9, 5): 7,
        (10, 5): 7,
        (11, 5): 7,
        (12, 5): 8,
        (13, 5): 8,
        (14, 5): 8,
        (15, 5): 8,
    }
    return table.get((n, res))


def _calculate_min_base_factors(n: int, res: int) -> int:
    """
    Compute the smallest k such that a 2^k design can support n factors
    at the requested resolution.

    Parameters
    ----------
    n : int
        Total number of factors.
    res : int
        Desired design resolution.

    Returns
    -------
    int
        The smallest k such that a 2^k design can support n factors at the
        requested resolution.
    """
    if res == 3:
        return max(2, math.ceil(math.log2(n + 1)))

    if res in {4, 5}:
        k = max(res - 1, math.ceil(math.log2(n + 1)))
        while _max_factors_for_resolution(k, res) < n and k <= 26:
            k += 1
        return k

    # Conservative fallback for higher resolutions
    k = max(res - 1, math.ceil(math.log2(n + 1)))
    while _max_factors_for_resolution(k, res) < n and k <= 26:
        k += 1
    return k


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
        given resolution.
    """
    if k < res - 1:
        return k

    total = k
    for r in range(res - 1, k + 1):
        total += int(binom(k, r))
    return total


def _n_fac_at_res(n: int, res: int) -> int:
    """
    Calculate number of possible factors for fractional factorial
    design with `n` base factors at resolution `res`.

    Parameters
    ----------
    n : int
        Number of base factors.
    res : int
        Desired design resolution.

    Returns
    -------
    int
        The number of possible factors for the given resolution.
    """
    return n + sum(int(binom(n, r)) for r in range(res - 1, n + 1))


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
    n_aliases = sum(
        n_comb(n_main_factors, n) for n in range(2, n_main_factors + 1)
    )

    if n_erased > n_comb(n_aliases, n_erased):
        raise ValueError("Too many erased factors to create aliasing")

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
        aliasing_design = " ".join([
            "".join([all_names[f] for f in a]) for a in aliasing
        ])
        complete_design = main_design + " " + aliasing_design
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
        If the design is too large (more than 20 factors).
    """
    _n_rounds, n_factors = design.shape

    if n_factors > 20:
        raise ValueError("Design too big, use 20 factors or less")

    all_names = string.ascii_lowercase
    factors = range(n_factors)
    all_combinations = itertools.chain.from_iterable(
        itertools.combinations(factors, n) for n in range(1, n_factors + 1)
    )
    aliases = {}

    for combination in all_combinations:
        contrast = np.prod(design[:, combination], axis=1)
        contrast.flags.writeable = False
        aliases[contrast.data.tobytes()] = aliases.get(
            contrast.data.tobytes(), []
        )
        aliases[contrast.data.tobytes()].append(combination)

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
        for sizes in itertools.combinations([len(a) for a in alias], 2):
            if not (sizes[0] >= 0 and sizes[1] >= 0):
                raise ValueError(f"Invalid alias sizes: {sizes}")
            if sizes[0] > sizes[1]:
                raise ValueError(f"Alias sizes not in order: {sizes}")
            alias_matrix[sizes[0] - 1, sizes[1] - 1] += 1

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
