# Changelog

All notable changes to PyDOE are documented here.

---

## [**Latest**](https://github.com/pydoe/pydoe/compare/v1.3.0...master)

### :material-wrench: Fixed
- `fracfact_opt` no longer rejects saturated designs. The feasibility check compared the number of erased factors against $\binom{n_{\text{aliases}}}{p}$ instead of against the number of available interaction columns $2^m - m - 1$, so every resolution III design with $k = 2^m - 1$ factors in $2^m$ runs — such as $2^{7-4}_{III}$ and $2^{15-11}_{III}$ — raised *"Too many erased factors to create aliasing"*. `fracfact_opt(n, 0)` also raised `IndexError` instead of returning the full factorial ([#152](https://github.com/pydoe/pydoe/issues/152))

### :material-refresh: Changed
- Generalize `definitive_screening_design` with arbitrary continuous-factor counts, two-level categorical factors, orthogonal and interaction-de-aliased methods, minimum-run augmentation, and fake factors while preserving historical one-argument results ([#145](https://github.com/pydoe/pydoe/pull/145)) — [@danieleongari](https://github.com/danieleongari)

### :material-speedometer: Performance
- `fracfact_aliasing` reduces each contrast to the bit pattern of its negative rows, so the elementwise product of a subset of columns becomes a bitwise exclusive or. The alias structure of a 15-factor design is now found in well under a second instead of about a minute ([#152](https://github.com/pydoe/pydoe/issues/152))

---

## [**v1.3.0**](https://github.com/pydoe/pydoe/releases/tag/v1.3.0) <small>2026-06-10</small>

### :material-plus-circle: Added
- Maximum projection design (`maxpro_design`) — optimizes Latin hypercube projections through coordinate-exchange search — [@saudzahirr](https://github.com/saudzahirr)
- Nearly orthogonal Latin hypercube (`nearly_orthogonal_lhs`) — minimizes correlation while retaining Latin hypercube stratification — [@saudzahirr](https://github.com/saudzahirr)
- Maximin and minimax distance designs (`maximin_design`, `minimax_design`) — spread points by optimizing pairwise or covering distances — [@saudzahirr](https://github.com/saudzahirr)
- Faure and Niederreiter low-discrepancy sequences (`faure_sequence`, `niederreiter_sequence`) — deterministic space-filling sequences for quasi-Monte Carlo sampling — [@saudzahirr](https://github.com/saudzahirr)
- Sequential Bayesian-optimization designs (`sequential_design`) with Gaussian-process regression and expected-improvement, probability-improvement, and upper-confidence-bound acquisition functions — [@saudzahirr](https://github.com/saudzahirr)

---

## [**v1.2.0**](https://github.com/pydoe/pydoe/releases/tag/v1.2.0) <small>2026-06-10</small>

### :material-plus-circle: Added
- Latin square (`latin_square`), Graeco-Latin square (`graeco_latin_square`), and hyper-Graeco-Latin square (`hyper_graeco_latin_square`) designs — remove the effect of two, three, or more nuisance factors using mutually orthogonal Latin squares — [@saudzahirr](https://github.com/saudzahirr)
- Hammersley point set (`hammersley_sequence`) — finite, fixed-size low-discrepancy point set with one equispaced coordinate and the remaining coordinates from the Halton sequence — [@saudzahirr](https://github.com/saudzahirr)
- Iman-Conover method (`iman_conover`) — induces a target rank correlation structure among sampled variables while preserving their marginal distributions — [@saudzahirr](https://github.com/saudzahirr)
- Mixture axial (screening) design (`mixture_axial_design`) — places one point at the simplex centroid and one axial point per component for screening blending effects — [@saudzahirr](https://github.com/saudzahirr)
- Extreme-vertices design (`extreme_vertices_design`) — enumerates the vertices of a constrained mixture region defined by per-component lower/upper bounds — [@saudzahirr](https://github.com/saudzahirr)
- Mixture-process variable design (`mixture_process_design`) — crosses a mixture design with an independent process-variable design — [@saudzahirr](https://github.com/saudzahirr)
- Blocking of full factorial designs (`block_full_factorial`) — splits a $2^k$ factorial into $2^p$ blocks by confounding chosen interactions with block effects — [@saudzahirr](https://github.com/saudzahirr)
- Orthogonally-blocked central composite design (`block_ccdesign`) — splits a CCD into a factorial block and an axial block with `alpha` chosen for orthogonal blocking — [@saudzahirr](https://github.com/saudzahirr)
- Hartley's small composite design (`small_composite_design`) — augments a resolution-III fractional factorial with star points, requiring fewer runs than a standard CCD — [@saudzahirr](https://github.com/saudzahirr)
- Definitive screening design (`definitive_screening_design`) — three-level Jones-Nachtsheim design requiring only $2k+1$ runs, built from a Paley conference matrix and its fold-over — [@saudzahirr](https://github.com/saudzahirr)
- Supersaturated design (`supersaturated_design`) — random-search construction of $k > n$ two-level designs minimizing $E(s^2)$ for screening under effect sparsity — [@saudzahirr](https://github.com/saudzahirr)
- Nested Latin hypercube (`nested_lhs`) — constructs nested designs for multi-fidelity experiments — [@saudzahirr](https://github.com/saudzahirr)
- Orthogonal array-based Latin hypercube design (`oa_lhd`) — Tang's (1993) construction turning a symmetric orthogonal array into a Latin hypercube with improved two-dimensional uniformity — [@saudzahirr](https://github.com/saudzahirr)
- Sliced Latin hypercube design (`sliced_lhs`) — partitions an $N=mt$-point Latin hypercube into $t$ slices of $m$ points, each a Latin hypercube in its own right — [@saudzahirr](https://github.com/saudzahirr)

---

## [**v1.1.0**](https://github.com/pydoe/pydoe/releases/tag/v1.1.0) <small>2026-06-09</small>

### :material-plus-circle: Added
- Simplex-lattice design (`simplex_lattice_design`) — mixture experiment design placing lattice points on the *q*-component simplex at resolution 1/*m*; supports fitting Scheffé polynomial models of degree *m* — [@saudzahirr](https://github.com/saudzahirr)
- Simplex-centroid design (`simplex_centroid_design`) — mixture experiment design with the centroids of all $2^q - 1$ non-empty subsets; supports estimation of all Scheffé interaction blending coefficients — [@saudzahirr](https://github.com/saudzahirr)
- John's 3/4 fractional factorial design (`john_three_quarter_design`) — semifoldover design using exactly 3/4 of the runs of the next full $2^k$ design; de-aliases all two-factor interactions involving the chosen factor — [@saudzahirr](https://github.com/saudzahirr)

### :material-wrench: Fixed
- Eliminate 604 test warnings by correctly catching `scipy.linalg.LinAlgWarning` in `regularized_inv`; suppress scipy Sobol advisory in the Saltelli path — [@saudzahirr](https://github.com/saudzahirr)

### :material-file-document: Documentation
- Fix contributing docs to reference `zensical` instead of `mkdocs serve` — [@saudzahirr](https://github.com/saudzahirr)
- Add `Mixture Designs` reference page — [@saudzahirr](https://github.com/saudzahirr)

---

## [**v1.0.2**](https://github.com/pydoe/pydoe/releases/tag/v1.0.2) <small>2026-05-17</small>

### :material-plus-circle: Added
- Add theory documentation pages ([#114](https://github.com/pydoe/pydoe/pull/114)) — [@saudzahirr](https://github.com/saudzahirr)

### :material-wrench: Fixed
- Fix typo in `factorial.md` ([#111](https://github.com/pydoe/pydoe/pull/111)) — [@saudzahirr](https://github.com/saudzahirr)

---

## [**v1.0.1**](https://github.com/pydoe/pydoe/releases/tag/v1.0.1) <small>2026-05-05</small>

Patch release following the 1.0.0 stable release. — [@saudzahirr](https://github.com/saudzahirr)

---

## [**v1.0.0**](https://github.com/pydoe/pydoe/releases/tag/v1.0.0) <small>2026-05-05</small>

:tada: **First stable release.** No breaking changes from 0.9.9.

---

## [**v0.9.9**](https://github.com/pydoe/pydoe/releases/tag/v0.9.9) <small>2026-04-07</small>

### :material-file-document: Documentation
- Migrate documentation tooling from MkDocs to Zensical ([#71](https://github.com/pydoe/pydoe/pull/71)) — [@saudzahirr](https://github.com/saudzahirr)
- Update copyright information ([#92](https://github.com/pydoe/pydoe/pull/92)) — [@saudzahirr](https://github.com/saudzahirr)

---

## [**v0.9.8**](https://github.com/pydoe/pydoe/releases/tag/v0.9.8) <small>2026-03-31</small>

### :material-wrench: Fixed
- Fix unexpected number of runs generated for fractional factorial designs ([#89](https://github.com/pydoe/pydoe/pull/89)) — [@Noor-Mustafa123](https://github.com/Noor-Mustafa123)

### :material-file-document: Documentation
- Format and clean up documentation ([#78](https://github.com/pydoe/pydoe/pull/78)) — [@saudzahirr](https://github.com/saudzahirr)

---

## [**v0.9.7**](https://github.com/pydoe/pydoe/releases/tag/v0.9.7) <small>2026-02-21</small>

### :material-plus-circle: Added
- Compatibility shim for legacy `pyDOE` import paths ([#70](https://github.com/pydoe/pydoe/pull/70)) — [@laraibg786](https://github.com/laraibg786)

---

## [**v0.9.6**](https://github.com/pydoe/pydoe/releases/tag/v0.9.6) <small>2026-02-16</small>

### :material-refresh: Changed
- Restructure codebase into modular package layout ([#63](https://github.com/pydoe/pydoe/pull/63)) — [@laraibg786](https://github.com/laraibg786)
- Rename modules to `snake_case` to comply with PEP 8 ([#62](https://github.com/pydoe/pydoe/pull/62)) — [@saudzahirr](https://github.com/saudzahirr)

### :material-wrench: Fixed
- Use `np.isclose` for logical float comparison in `pbdesign` ([#65](https://github.com/pydoe/pydoe/pull/65)) — [@saudzahirr](https://github.com/saudzahirr)

---

## [**v0.9.5**](https://github.com/pydoe/pydoe/releases/tag/v0.9.5) <small>2026-02-11</small>

### :material-refresh: Changed
- Refresh documentation badges and `project.urls` ([#59](https://github.com/pydoe/pydoe/pull/59)) — [@saudzahirr](https://github.com/saudzahirr)

---

## [**v0.9.4**](https://github.com/pydoe/pydoe/releases/tag/v0.9.4) <small>2026-02-10</small>

:rocket: **First public PyPI release.**

### :material-plus-circle: Added
- Factorial designs resolution table in documentation ([#54](https://github.com/pydoe/pydoe/pull/54)) — [@saudzahirr](https://github.com/saudzahirr)
- Type annotations throughout the codebase ([#49](https://github.com/pydoe/pydoe/pull/49)) — [@saudzahirr](https://github.com/saudzahirr)
- Ruff as default linter and formatter with Codecov integration ([#47](https://github.com/pydoe/pydoe/pull/47)) — [@saudzahirr](https://github.com/saudzahirr)
- PyPI release workflow ([#57](https://github.com/pydoe/pydoe/pull/57)) — [@laraibg786](https://github.com/laraibg786)
- `uv` as package manager with updated `pyproject.toml` ([#40](https://github.com/pydoe/pydoe/pull/40)) — [@laraibg786](https://github.com/laraibg786)
- Dynamic versioning without a `_version` file ([#38](https://github.com/pydoe/pydoe/pull/38)) — [@laraibg786](https://github.com/laraibg786)

### :material-wrench: Fixed
- Buggy resolution values in fractional factorial designs ([#52](https://github.com/pydoe/pydoe/pull/52)) — [@Noor-Mustafa123](https://github.com/Noor-Mustafa123)
- `lhsmu` `TypeError` when using NumPy `Generator` API ([#44](https://github.com/pydoe/pydoe/pull/44)) — [@saudzahirr](https://github.com/saudzahirr)
- Negative determinant bug in `d_efficiency` ([#42](https://github.com/pydoe/pydoe/pull/42)) — [@saudzahirr](https://github.com/saudzahirr)
- Singular information matrix handling — [@saudzahirr](https://github.com/saudzahirr)
- Improve RAM usage in `_lhsclassic` ([#43](https://github.com/pydoe/pydoe/pull/43)) — [@saudzahirr](https://github.com/saudzahirr)

---

## [**v0.9.3**](https://github.com/pydoe/pydoe-archive/releases/tag/v0.9.3) <small>2026-01-29</small>

### :material-plus-circle: Added
- Type annotations throughout the codebase — [@saudzahirr](https://github.com/saudzahirr)
- Ruff as linter and formatter with Codecov integration — [@saudzahirr](https://github.com/saudzahirr)

### :material-file-document: Documentation
- Update and deploy documentation — [@saudzahirr](https://github.com/saudzahirr)

---

## [**v0.9.2**](https://github.com/pydoe/pydoe-archive/releases/tag/v0.9.2) <small>2026-01-07</small>

### :material-plus-circle: Added
- Migrate documentation site from Sphinx to MkDocs — [@saudzahirr](https://github.com/saudzahirr)
- Integrate `uv` as package manager — [@laraibg786](https://github.com/laraibg786)
- Dynamic versioning without a `_version` file — [@laraibg786](https://github.com/laraibg786)

### :material-wrench: Fixed
- Fix `lhsmu` `TypeError` when using NumPy `Generator` API — [@saudzahirr](https://github.com/saudzahirr)
- Fix negative determinant bug in `d_efficiency` — [@saudzahirr](https://github.com/saudzahirr)
- Improve RAM usage in `_lhsclassic` — [@saudzahirr](https://github.com/saudzahirr)

---

## [**v0.9.1**](https://github.com/pydoe/pydoe-archive/releases/tag/v0.9.1) <small>2025-11-27</small>

### :material-wrench: Fixed
- Fix singular information matrix handling — [@saudzahirr](https://github.com/saudzahirr)

---

## [**v0.9.0**](https://github.com/pydoe/pydoe-archive/releases/tag/v0.9.0) <small>2025-11-05</small>

### :material-plus-circle: Added
- Add `doe_sparse_grid` using Smolyak's sparse grid construction — [@saudzahirr](https://github.com/saudzahirr)

---

## [**v0.8.1**](https://github.com/pydoe/pydoe-archive/releases/tag/v0.8.1) <small>2025-10-27</small>

### :material-refresh: Changed
- Replace legacy NumPy `RandomState` with `Generator` API throughout — [@saudzahirr](https://github.com/saudzahirr)

---

## [**v0.8.0**](https://github.com/pydoe/pydoe-archive/releases/tag/v0.8.0) <small>2025-10-24</small>

### :material-plus-circle: Added
- Add `morris_sampling` for Morris one-at-a-time sensitivity screening — [@saudzahirr](https://github.com/saudzahirr)
- Add `saltelli_sampling` for variance-based Sobol' sensitivity analysis — [@saudzahirr](https://github.com/saudzahirr)
- Add `random_uniform` and `random_k_means` space-filling designs — [@saudzahirr](https://github.com/saudzahirr)
- Deprecate `random_state` parameter in `lhs` in favour of `seed` — [@saudzahirr](https://github.com/saudzahirr)

---

## [**v0.7.0**](https://github.com/pydoe/pydoe-archive/releases/tag/v0.7.0) <small>2025-09-09</small>

### :material-plus-circle: Added
- Add `halton_sequence`, `sobol_sequence`, `korobov_sequence`, `rank1_lattice`, and `sukharev_grid` — [@saudzahirr](https://github.com/saudzahirr)
- Add `cranley_patterson_shift` for randomising quasi-random sequences — [@saudzahirr](https://github.com/saudzahirr)

---

## [**v0.6.2**](https://github.com/pydoe/pydoe-archive/releases/tag/v0.6.2) <small>2025-09-02</small>

### :material-plus-circle: Added
- Add `optimal_design` with D, A, E, G, I, V, C, S, T optimality criteria — [@saudzahirr](https://github.com/saudzahirr)
- Add Fedorov, modified Fedorov, DETMAX, and Dykstra algorithms — [@saudzahirr](https://github.com/saudzahirr)

---

## [**v0.6.1**](https://github.com/pydoe/pydoe-archive/releases/tag/v0.6.1) <small>2025-07-08</small>

### :material-plus-circle: Added
- Add `doehlert_shell_design` and `doehlert_simplex_design` — [@saudzahirr](https://github.com/saudzahirr)

---

## [**v0.6.0**](https://github.com/pydoe/pydoe-archive/releases/tag/v0.6.0) <small>2025-07-03</small>

### :material-plus-circle: Added
- Add `taguchi_design` with orthogonal array lookup, SNR computation, and `TaguchiObjective` enum — [@saudzahirr](https://github.com/saudzahirr)

---

## [**v0.5.5**](https://github.com/pydoe/pydoe-archive/releases/tag/v0.5.5) <small>2025-05-16</small>

### :material-wrench: Fixed
- Update `gsd` to accept level count of 1 — [@saudzahirr](https://github.com/saudzahirr)

---

## [**v0.5.0**](https://github.com/pydoe/pydoe-archive/releases/tag/v0.5.0) <small>2023-10-25</small>

### :material-cog: Infrastructure
- Migrate build system to `pyproject.toml` with Hatchling — [@saudzahirr](https://github.com/saudzahirr)
- Add full test suite and CI automation — [@saudzahirr](https://github.com/saudzahirr)

### :material-wrench: Fixed
- Replace Black with Ruff; fix `fracfact_opt`; add doc-tests — [@saudzahirr](https://github.com/saudzahirr)
- Refactor `fracfact` with test updates and minor bug fixes — [@saudzahirr](https://github.com/saudzahirr)

---

## [**v0.4.0**](https://github.com/pydoe/pydoe-archive/releases/tag/v0.4.0) <small>2021-03-03</small>

### :material-plus-circle: Added
- Add `gsd` (Generalized Subset Design) — [@saudzahirr](https://github.com/saudzahirr)

---

## [**v0.3.9**](https://github.com/pydoe/pydoe-archive/releases/tag/v0.3.9) <small>2017-04-25</small>

### :material-wrench: Fixed
- Fix integer division for Python 3 compatibility — [@tisimst](https://github.com/tisimst)

---

## [**v0.3.8**](https://github.com/pydoe/pydoe-archive/releases/tag/v0.3.8) <small>2015-07-13</small>

### :material-wrench: Fixed
- Fix incorrect indexing variable in `_pdist` — [@tisimst](https://github.com/tisimst)

---

## [**v0.3.7**](https://github.com/pydoe/pydoe-archive/releases/tag/v0.3.7) <small>2014-12-09</small>

### :material-refresh: Changed
- Clarify BSD 3-clause license — [@tisimst](https://github.com/tisimst)

---

## [**v0.3.6**](https://github.com/pydoe/pydoe-archive/releases/tag/v0.3.6) <small>2014-09-22</small>

### :material-refresh: Changed
- Add code credits to original authors — [@tisimst](https://github.com/tisimst)

---

## [**v0.3.5**](https://github.com/pydoe/pydoe-archive/releases/tag/v0.3.5) <small>2014-02-18</small>

### :material-plus-circle: Added
- Add `var_regression_matrix` to public API — [@tisimst](https://github.com/tisimst)

---

## [**v0.3.4**](https://github.com/pydoe/pydoe-archive/releases/tag/v0.3.4) <small>2014-02-18</small>

### :material-wrench: Fixed
- Fix Python 3 import and compatibility issues — [@tisimst](https://github.com/tisimst)

---

## [**v0.3.3**](https://github.com/pydoe/pydoe-archive/releases/tag/v0.3.3) <small>2014-02-12</small>

### :material-wrench: Fixed
- Remove package import for better Python 3 compatibility — [@tisimst](https://github.com/tisimst)

---

## [**v0.3.2**](https://github.com/pydoe/pydoe-archive/releases/tag/v0.3.2) <small>2014-02-09</small>

### :material-wrench: Fixed
- Fix indexing problem; improve Python 3 compatibility — [@tisimst](https://github.com/tisimst)

---

## [**v0.3.1**](https://github.com/pydoe/pydoe-archive/releases/tag/v0.3.1) <small>2014-02-09</small>

### :material-wrench: Fixed
- Fix indexing issue — [@tisimst](https://github.com/tisimst)

---

## [**v0.3.0**](https://github.com/pydoe/pydoe-archive/releases/tag/v0.3.0) <small>2013-11-08</small>

### :material-refresh: Changed
- Simplify design function inputs to accept number of factors directly — [@tisimst](https://github.com/tisimst)
- Improve consistency across all design functions — [@tisimst](https://github.com/tisimst)

---

## [**v0.2.2**](https://github.com/pydoe/pydoe-archive/releases/tag/v0.2.2) <small>2013-10-23</small>

### :material-plus-circle: Added
- Add `criterion` keyword argument to `lhs` for more flexible sampling — [@tisimst](https://github.com/tisimst)

---

## [**v0.2.1**](https://github.com/pydoe/pydoe-archive/releases/tag/v0.2.1) <small>2013-09-13</small>

### :material-wrench: Fixed
- Fix syntax errors and update documentation links — [@tisimst](https://github.com/tisimst)

---

## [**v0.2.0**](https://github.com/pydoe/pydoe-archive/releases/tag/v0.2.0) <small>2013-09-13</small>

### :material-plus-circle: Added
- Add response surface designs (`bbdesign`, `ccdesign`) — [@tisimst](https://github.com/tisimst)
- Add Latin hypercube sampling (`lhs`) — [@tisimst](https://github.com/tisimst)

---

## [**0.1.alpha**](https://github.com/pydoe/pydoe-archive/releases/tag/0.1.alpha) <small>2013-09-05</small>

:seedling: **Initial release** with `fullfact`, `ff2n`, `fracfact`, `pbdesign`, and `gsd`. — [@tisimst](https://github.com/tisimst)
