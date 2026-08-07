"""
`PyDOE` original code was originally converted from code by the following
individuals for use with Scilab:

- Copyright (C) 2012-2013, Michael Baudin
- Copyright (C) 2012, Maria Christopoulou
- Copyright (C) 2010-2011, INRIA, Michael Baudin
- Copyright (C) 2009, Yann Collette
- Copyright (C) 2009, CEA, Jean-Marc Martinez

`PyDOE` was converted to Python by the following individual:

- Copyright (c) 2014, Abraham D. Lee

The following individuals forked `PyDOE` and worked on `PyDOE2`:

- Copyright (C) 2018, Rickard Sjögren and Daniel Svensson

The following individuals forked `PyDOE2` and worked on `PyDOE3`:

- Copyright (C) 2023 - Rémi Lafage
"""

from importlib.metadata import PackageNotFoundError, version

from .clustering import random_k_means
from .factorial import (
    alias_vector_indices,
    block_full_factorial,
    ff2n,
    fold,
    fracfact,
    fracfact_aliasing,
    fracfact_by_res,
    fracfact_opt,
    fullfact,
    graeco_latin_square,
    gsd,
    hyper_graeco_latin_square,
    john_three_quarter_design,
    latin_square,
    pbdesign,
)
from .mixture import (
    extreme_vertices_design,
    mixture_axial_design,
    mixture_process_design,
    simplex_centroid_design,
    simplex_lattice_design,
)
from .optimal import (
    a_efficiency,
    a_optimality,
    build_design_matrix,
    build_uniform_moment_matrix,
    c_optimality,
    criterion_value,
    d_efficiency,
    d_optimality,
    detmax,
    e_optimality,
    fedorov,
    g_optimality,
    generate_candidate_set,
    i_optimality,
    information_matrix,
    modified_fedorov,
    optimal_design,
    s_optimality,
    sequential_dykstra,
    simple_exchange_wynn_mitchell,
    t_optimality,
    v_optimality,
)
from .response_surface import (
    bbdesign,
    block_ccdesign,
    ccdesign,
    doehlert_shell_design,
    doehlert_simplex_design,
    repeat_center,
    small_composite_design,
    star,
    union,
)
from .sensitivity_analysis import morris_sampling, saltelli_sampling
from .sequential import (
    GaussianProcessRegressor,
    expected_improvement,
    probability_of_improvement,
    sequential_design,
    upper_confidence_bound,
)
from .space_filling.quasi_random import (
    cranley_patterson_shift,
    faure_sequence,
    halton_sequence,
    hammersley_sequence,
    korobov_sequence,
    niederreiter_sequence,
    rank1_lattice,
    sobol_sequence,
    sukharev_grid,
)
from .space_filling.stochastic import (
    lhs,
    maximin_design,
    maxpro_design,
    minimax_design,
    nearly_orthogonal_lhs,
    nested_lhs,
    oa_lhd,
    random_uniform,
    sliced_lhs,
)
from .sparse_grid import doe_sparse_grid, sparse_grid_dimension
from .specialized import definitive_screening_design, supersaturated_design
from .taguchi import (
    TaguchiObjective,
    compute_snr,
    get_orthogonal_array,
    list_orthogonal_arrays,
    taguchi_design,
)
from .utils import iman_conover, scale_samples, var_regression_matrix


try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "GaussianProcessRegressor",
    "TaguchiObjective",
    "a_efficiency",
    "a_optimality",
    "alias_vector_indices",
    "bbdesign",
    "block_ccdesign",
    "block_full_factorial",
    "build_design_matrix",
    "build_uniform_moment_matrix",
    "c_optimality",
    "ccdesign",
    "compute_snr",
    "cranley_patterson_shift",
    "criterion_value",
    "d_efficiency",
    "d_optimality",
    "definitive_screening_design",
    "detmax",
    "doe_sparse_grid",
    "doehlert_shell_design",
    "doehlert_simplex_design",
    "e_optimality",
    "expected_improvement",
    "extreme_vertices_design",
    "faure_sequence",
    "fedorov",
    "ff2n",
    "fold",
    "fracfact",
    "fracfact_aliasing",
    "fracfact_by_res",
    "fracfact_opt",
    "fullfact",
    "g_optimality",
    "generate_candidate_set",
    "get_orthogonal_array",
    "graeco_latin_square",
    "gsd",
    "halton_sequence",
    "hammersley_sequence",
    "hyper_graeco_latin_square",
    "i_optimality",
    "iman_conover",
    "information_matrix",
    "john_three_quarter_design",
    "korobov_sequence",
    "latin_square",
    "lhs",
    "list_orthogonal_arrays",
    "maximin_design",
    "maxpro_design",
    "minimax_design",
    "mixture_axial_design",
    "mixture_process_design",
    "modified_fedorov",
    "morris_sampling",
    "nearly_orthogonal_lhs",
    "nested_lhs",
    "niederreiter_sequence",
    "oa_lhd",
    "optimal_design",
    "pbdesign",
    "probability_of_improvement",
    "random_k_means",
    "random_uniform",
    "rank1_lattice",
    "repeat_center",
    "s_optimality",
    "saltelli_sampling",
    "scale_samples",
    "sequential_design",
    "sequential_dykstra",
    "simple_exchange_wynn_mitchell",
    "simplex_centroid_design",
    "simplex_lattice_design",
    "sliced_lhs",
    "small_composite_design",
    "sobol_sequence",
    "sparse_grid_dimension",
    "star",
    "sukharev_grid",
    "supersaturated_design",
    "t_optimality",
    "taguchi_design",
    "union",
    "upper_confidence_bound",
    "v_optimality",
    "var_regression_matrix",
]
