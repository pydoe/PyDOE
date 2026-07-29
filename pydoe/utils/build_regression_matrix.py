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

from collections.abc import Iterator

import numpy as np


def grep(haystack: str, needle: str) -> Iterator[int]:
    start = 0
    while True:
        start = haystack.find(needle, start)
        if start == -1:
            return
        yield start
        start += len(needle)


def build_regression_matrix(  # ruff: ignore[too-many-branches]
    H: np.ndarray, model: str, build: np.ndarray[np.bool_] = None
) -> np.ndarray:
    """
    Build a regression matrix using a DOE matrix and a list of monomials.

    Parameters
    ----------
    H : 2d-array
    model : str
    build : bool-array

    Returns
    -------
    R : 2d-array

    """
    list_of_tokens = model.split(" ")
    if H.shape[1] == 1:
        size_index = len(str(H.shape[0]))
    else:
        size_index = len(str(H.shape[1]))

    if build is None:
        build = [True] * len(list_of_tokens)

    # Test if the vector has the wrong direction (lines instead of columns)
    if H.shape[0] == 1:
        H = H.T

    # FIXME: Unused variable 'monom_index'
    # Collect the list of monomials
    monom_index = []
    for i in range(len(list_of_tokens)):
        if build[i]:
            # BUG: Argument haystack should be str, not list of str
            monom_index += [
                grep(
                    list_of_tokens,
                    "x" + str(0) * (size_index - len(str(i))) + str(i),
                )
            ]

    monom_index = -np.sort(-monom_index)
    monom_index = np.unique(monom_index)

    if H.shape[1] == 1:
        # vector "mode": the number of vars is equal
        # to the number of lines of H
        nb_var = H.shape[0]
        vector_mode = True

        for i in range(nb_var):
            for j in range(list_of_tokens.shape[0]):
                list_of_tokens[j] = list_of_tokens[j].replace(
                    "x" + str(0) * (size_index - len(str(i))) + str(i),
                    "H(" + str(i) + ")",
                )
    else:
        # matrix "mode": the number of vars is equal
        # to the number of columns of H
        nb_var = H.shape[0]
        vector_mode = False

        for i in range(nb_var):
            for j in range(list_of_tokens.shape[0]):
                list_of_tokens[j] = list_of_tokens[j].replace(
                    "x" + str(0) * (size_index - len(str(i))) + str(i),
                    "H[i," + str(i) + ")",
                )

    # Now build the regression matrix
    if vector_mode:
        R = np.zeros((len(list_of_tokens), 1))
        for j in range(len(list_of_tokens)):
            R[j, 0] = eval(list_of_tokens[j])  # ruff: ignore[suspicious-eval-usage]
    else:
        R = np.zeros((H.shape[0], len(list_of_tokens)))
        for i in range(H.shape[0]):
            for j in range(len(list_of_tokens)):
                R[i, j] = eval(list_of_tokens[j])  # ruff: ignore[suspicious-eval-usage]

    return R
