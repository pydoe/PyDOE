# PyDOE

![PyDOE](./assets/images/light-theme/pydoe-banner.svg#only-light)
![PyDOE](./assets/images/dark-theme/pydoe-banner.svg#only-dark)

*An Experimental Design Package for Python*

The `PyDOE` package is designed to help the **scientist, engineer,
statistician,** etc., to construct appropriate 
**experimental designs**.


!!! tip "Quick Start"
    All available designs can be accessed after a simple import statement:

    ```python
    >>> from pydoe import *
    ```

## Overview

The package provides extensive support for design-of-experiments (DOE) methods and is capable of creating designs for any number of factors.

It provides:

- **Factorial Designs**
  - General Full-Factorial (``fullfact``)
  - 2-level Full-Factorial (``ff2n``)
  - 2-level Fractional Factorial (``fracfact``, ``fracfact_aliasing``, ``fracfact_by_res``, ``fracfact_opt``, ``alias_vector_indices``)
  - Plackett-Burman (``pbdesign``)
  - Generalized Subset Designs (``gsd``)
  - Fold-over Designs (``fold``)
  - John's 3/4 Fractional Factorial (``john_three_quarter_design``)
  - Latin Square Designs (``latin_square``)
  - Graeco-Latin Square Designs (``graeco_latin_square``)
  - Hyper-Graeco-Latin Square Designs (``hyper_graeco_latin_square``)
  - Blocking of Full Factorial Designs (``block_full_factorial``)

- **Mixture Designs**
  - Simplex-Lattice Design (``simplex_lattice_design``)
  - Simplex-Centroid Design (``simplex_centroid_design``)
  - Axial (Screening) Design (``mixture_axial_design``)
  - Extreme-Vertices Design (``extreme_vertices_design``)
  - Mixture-Process Variable Design (``mixture_process_design``)

- **Response-Surface Designs**
  - Box-Behnken (``bbdesign``)
  - Central-Composite (``ccdesign``)
  - Doehlert Design (``doehlert_shell_design``, ``doehlert_simplex_design``)
  - Star Designs (``star``)
  - Union Designs (``union``)
  - Repeated Center Points (``repeat_center``)
  - Blocked Central Composite Design (``block_ccdesign``)
  - Small Composite Design (``small_composite_design``)

- **Space-Filling Designs**
  - Latin-Hypercube (``lhs``)
  - Orthogonal Array-based Latin Hypercube (``oa_lhd``)
  - Sliced Latin Hypercube (``sliced_lhs``)
  - Nested Latin Hypercube (``nested_lhs``)
  - Maximin Distance Design (``maximin_design``)
  - Minimax Distance Design (``minimax_design``)
  - Maximum Projection Design (``maxpro_design``)
  - Nearly Orthogonal Latin Hypercube (``nearly_orthogonal_lhs``)
  - Random Uniform (``random_uniform``)

- **Low-Discrepancy Sequences**
  - Sukharev Grid (``sukharev_grid``)
  - Sobol’ Sequence (``sobol_sequence``)
  - Halton Sequence (``halton_sequence``)
  - Hammersley Point Set (``hammersley_sequence``)
  - Rank-1 Lattice Design (``rank1_lattice``)
  - Korobov Sequence (``korobov_sequence``)
  - Faure Sequence (``faure_sequence``)
  - Niederreiter Sequence (``niederreiter_sequence``)
  - Cranley-Patterson Randomization (``cranley_patterson_shift``)

- **Clustering Designs**
  - Random K-Means (``random_k_means``)

- **Sensitivity Analysis Designs**
  - Morris Method (``morris_sampling``)
  - Saltelli Sampling (``saltelli_sampling``)
  - Iman-Conover Method (``iman_conover``)

- **Taguchi Designs**
  - Orthogonal arrays and robust design utilities (``taguchi_design``, ``compute_snr``, ``get_orthogonal_array``, ``list_orthogonal_arrays``, ``TaguchiObjective``)

- **Optimal Designs**
  - Advanced optimal design algorithms (``optimal_design``)
  - Optimality criteria (``a_optimality``, ``c_optimality``, ``d_optimality``, ``e_optimality``, ``g_optimality``, ``i_optimality``, ``s_optimality``, ``t_optimality``, ``v_optimality``)
  - Efficiency measures (``a_efficiency``, ``d_efficiency``)
  - Search algorithms (``sequential_dykstra``, ``simple_exchange_wynn_mitchell``, ``fedorov``, ``modified_fedorov``, ``detmax``)
  - Design utilities (``criterion_value``, ``information_matrix``, ``build_design_matrix``, ``build_uniform_moment_matrix``, ``generate_candidate_set``)

- **Sparse Grid Designs**
  - Sparse Grid Design (``doe_sparse_grid``)
  - Sparse Grid Dimension (``sparse_grid_dimension``)

- **Specialized Designs**
  - Definitive Screening Design (``definitive_screening_design``)
  - Supersaturated Design (``supersaturated_design``)

- **Sequential / Adaptive Designs**
  - Sequential Design Driver (``sequential_design``)
  - Gaussian Process Surrogate (``GaussianProcessRegressor``)
  - Acquisition Functions (``expected_improvement``, ``probability_of_improvement``, ``upper_confidence_bound``)
