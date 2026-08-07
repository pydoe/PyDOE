# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install all dependencies (including dev/lint/test/docs groups)
uv sync --all-groups

# Run full test suite (tests + doctests from all source modules)
uv run pytest

# Run test files in parallel (faster)
uv run pytest -n auto tests

# Run a single test file
uv run pytest tests/test_lhs.py

# Run a specific test by name
uv run pytest tests/test_lhs.py::TestLhs::test_lhs_vanilla

# Format code
uvx ruff format .

# Lint (check only)
uvx ruff check .

# Lint and auto-fix
uvx ruff check --fix .

# Build documentation
uv run python -m zensical build

# Print debug info for bug reports
uv run pydoe --debug
```

## Architecture

PyDOE is a Design of Experiments library. **Every public symbol is re-exported through `pydoe/__init__.py`** — that is the single public API surface. The package is organized into subpackages by DOE category:

| Subpackage | Contents |
|---|---|
| `pydoe/factorial/` | Full/fractional factorial (`fullfact`, `ff2n`, `fracfact*`), Plackett-Burman, GSD, fold-over, Latin/Graeco-Latin square, John's 3/4 fractional factorial, blocking |
| `pydoe/mixture/` | Simplex-lattice/centroid, axial (screening), extreme-vertices, mixture+process-variable designs |
| `pydoe/response_surface/` | Box-Behnken, central composite (CCD), Doehlert, star, union, repeat_center, small composite, blocking |
| `pydoe/space_filling/quasi_random/` | Low-discrepancy sequences: Sobol', Halton, Hammersley, Korobov, rank-1 lattice, Sukharev, Faure, Niederreiter, Cranley-Patterson |
| `pydoe/space_filling/stochastic/` | Latin hypercube (`lhs`), OA-LHD, sliced/nested LHS, maximin/minimax distance designs, MaxPro, nearly-orthogonal LHS, random uniform |
| `pydoe/sequential/` | Adaptive/Bayesian-optimization designs: `sequential_design`, `GaussianProcessRegressor`, acquisition functions (EI, PI, UCB) |
| `pydoe/optimal/` | Optimal design algorithms and all optimality criteria |
| `pydoe/taguchi/` | Taguchi orthogonal arrays (data files in `orthogonal_arrays/`), SNR, `TaguchiObjective` |
| `pydoe/sensitivity_analysis/` | Morris method, Saltelli sampling |
| `pydoe/clustering/` | Random K-means sampling |
| `pydoe/sparse_grid/` | Sparse grid designs |
| `pydoe/specialized/` | Definitive screening designs, supersaturated designs |
| `pydoe/utils/` | `scale_samples` (unit-hypercube → arbitrary bounds), `var_regression_matrix`, `iman_conover` |
| `pydoe/debug.py` | CLI entrypoint (`pydoe --debug`) for printing environment info |

## Optimal Design Module (`pydoe/optimal/`)

The optimal design module is the most mathematically complex part. Key files:

- **`model.py`** — Builds the design matrix `X` from candidate points for polynomial models (linear, quadratic, higher-order). The information matrix is `M = X^T X / n`.
- **`criterion.py`** — All optimality criteria (D, A, I, C, E, G, V, S, T). All criteria that require `M^{-1}` go through `regularized_inv()`.
- **`utils.py`** — `criterion_value()` dispatcher; `_best_single_add/_drop` helpers used by all algorithms; `information_matrix()`.
- **`algorithms.py`** — Five iterative algorithms: Sequential Dykstra (greedy), Wynn-Mitchell (simple exchange), Fedorov (all-pairs exchange), Modified Fedorov, DETMAX (exchange with excursions).
- **`efficiency.py`** — D-efficiency and A-efficiency post-hoc measures.
- **`optimal.py`** — Top-level `optimal_design()` that dispatches to the chosen algorithm.

### Numerical conditioning in optimal design

During iterative search the algorithms evaluate criterion values for many intermediate candidate designs, some of which produce an ill-conditioned information matrix (e.g. too few points to span the model space). `regularized_inv()` in `criterion.py` handles this by:

1. Attempting a plain `scipy.linalg.inv(M)` inside a `warnings.catch_warnings()` context that promotes `scipy.linalg.LinAlgWarning` to an exception.
2. Catching both `LinAlgWarning` (ill-conditioned) and `np.linalg.LinAlgError` (exactly singular).
3. Falling back to Tikhonov regularization: `M_reg = M + λI` where `λ = 1e-8`, which shifts all eigenvalues up by `λ` and guarantees positive-definiteness.

**Important**: `scipy.linalg.inv` emits `LinAlgWarning` for ill-conditioned matrices but only raises `LinAlgError` for exactly singular ones. The original `except np.linalg.LinAlgError` pattern was therefore dead code for the ill-conditioned case.

## Sequential / Adaptive Design Module (`pydoe/sequential/`)

`sequential_design()` (in `adaptive.py`) drives a Bayesian-optimization
loop: it builds an initial space-filling design via
`space_filling.stochastic.lhs` + `utils.scale_samples`, fits a
`GaussianProcessRegressor` (RBF kernel, in `gaussian_process.py`) to the
observations, and at each iteration scores random candidates with one of
the acquisition functions in `acquisition.py` (`expected_improvement`,
`probability_of_improvement`, `upper_confidence_bound`) to pick the next
evaluation point. All GP fitting/prediction is done on inputs normalized
to `[0, 1]^d` so `length_scale` stays meaningful regardless of `bounds`.
Acquisition functions always return "higher = more promising" scores,
even when `maximize=False`.

## Sobol' Sequence and Saltelli Sampling

`sobol_sequence()` has a `use_pow_of_2` flag (default `True`):
- `True`: rounds `n` up to the nearest power of 2 and calls `random_base2(m)` — ideal balance and coverage.
- `False`: generates exactly `n` samples via `random(n)`. Used internally by `saltelli_sampling`, which needs an exact count of `N + skip_values` samples.

`saltelli_sampling` passes `use_pow_of_2=False` and suppresses the resulting scipy `UserWarning` internally, since `saltelli_sampling` already validates and warns the user if `N` is not a power of 2.

## Testing

Pytest is configured in `pyproject.toml` with `addopts = "--doctest-modules"` and `testpaths = ["tests", "pydoe"]`. This means **every docstring example in every source file is a live test**. Keep all docstring examples accurate and runnable.

Test files in `tests/` have relaxed lint rules: no type annotations required (`ANN001`/`ANN201`/`ANN202`), `assert` allowed (`S101`), uppercase variable names allowed (`N802`/`N803`/`N806`).

The deprecated `random_state` parameter of `lhs()` is still supported but emits `DeprecationWarning`. Tests for the deprecated path use `self.assertWarns(DeprecationWarning)` as the outer context.

## Linting

Ruff enforces strict rules. Notable ones:
- `ANN` — type annotations required on all public functions/methods.
- `FBT` — boolean positional arguments must be keyword-only.
- `DOC` — docstring must list all parameters, returns, and warnings actually raised.
- `N803`/`N806` — **ignored**: uppercase matrix variable names like `X`, `M`, `N` are accepted throughout (scientific convention).
- `PLR2004` — **ignored**: magic number comparisons allowed.
- `S311` — **ignored**: non-cryptographic random usage allowed.

Line length is 80. Do not use `# noqa` suppression comments. When a suppression
is genuinely necessary, use Ruff's human-readable form, such as
`# ruff: ignore[too-many-arguments]` or
`# ruff: ignore[too-many-branches]`.

## Documentation

Docs live in `docs/` and are built with **zensical** (not mkdocs directly, despite `mkdocs.yml` being present — zensical reads it). Run `uv run python -m zensical build` to build. Reference docs for each module are in `docs/reference/`; theory pages are in `docs/theory/`. API docs are written by hand (not auto-generated from docstrings), so keep both in sync when changing public interfaces.
