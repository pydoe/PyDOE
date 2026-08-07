"""
Combined mixture and process-variable designs.

Many mixture experiments also involve independent *process variables*
(e.g. temperature, mixing time) that are not subject to the
sum-to-one constraint. A mixture-process design crosses a mixture
design with a process-variable design so that every combination of
mixture blend and process setting is run.

References
----------
Cornell, J. A. (2002). *Experiments with Mixtures: Designs, Models, and the
    Analysis of Mixture Data* (3rd ed.). Wiley.
NIST/SEMATECH e-Handbook of Statistical Methods, Section 5.6.4 -
    "Mixture and Process Variable Designs",
    https://www.itl.nist.gov/div898/handbook/pri/section6/pri64.htm
"""

from __future__ import annotations

import numpy as np


__all__ = ["mixture_process_design"]


def mixture_process_design(
    mixture: np.ndarray, process: np.ndarray
) -> np.ndarray:
    """
    Cross a mixture design with a process-variable design.

    Every row of ``mixture`` is paired with every row of ``process``,
    producing the full Cartesian product of the two designs.

    Parameters
    ----------
    mixture : array_like of shape (n1, q)
        Mixture design matrix, e.g. from
        [`simplex_lattice_design`][pydoe.simplex_lattice_design] or
        [`mixture_axial_design`][pydoe.mixture_axial_design]. Each row
        should sum to 1.
    process : array_like of shape (n2, p)
        Process-variable design matrix, e.g. from
        [`ff2n`][pydoe.ff2n], with one column per process variable.

    Returns
    -------
    ndarray of shape (n1 * n2, q + p)
        Combined design. The first ``q`` columns are the mixture
        proportions and the last ``p`` columns are the process-variable
        settings. Row ``i * n2 + j`` pairs mixture row ``i`` with
        process row ``j``.

    Examples
    --------
    >>> mixture = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    >>> process = np.array([[-1.0], [1.0]])
    >>> mixture_process_design(mixture, process)
    array([[ 1. ,  0. , -1. ],
           [ 1. ,  0. ,  1. ],
           [ 0.5,  0.5, -1. ],
           [ 0.5,  0.5,  1. ],
           [ 0. ,  1. , -1. ],
           [ 0. ,  1. ,  1. ]])
    """
    mixture = np.asarray(mixture, dtype=float)
    process = np.asarray(process, dtype=float)
    n1 = mixture.shape[0]
    n2 = process.shape[0]
    mix_rep = np.repeat(mixture, n2, axis=0)
    proc_rep = np.tile(process, (n1, 1))
    return np.hstack([mix_rep, proc_rep])
