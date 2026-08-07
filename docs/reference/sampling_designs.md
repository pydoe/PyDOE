Sampling designs are structured experimental design methods used to efficiently
explore parameter spaces and quantify relationships between input variables and
model outputs. These methods are particularly valuable for sensitivity analysis,
understanding model behavior, identifying influential parameters, and reducing
computational burden by focusing on the most important variables.

This section includes the following sampling design methods:

- [Morris Method](#morris_method)
- [Saltelli Sampling](#saltelli_sampling)
- [Iman-Conover Method](#iman_conover)

!!! hint
    All sampling design functions are available with:

    ```python
    >>> from pydoe import morris_sampling, saltelli_sampling, iman_conover
    ```

## Morris Method (`morris_sampling`) {#morris_method}

The Morris method, also known as the Morris screening method, is a computationally
efficient global sensitivity analysis technique that estimates the importance of
input variables by analyzing one-at-a-time (OAT) trajectories through a discretized
parameter space. It is especially useful for identifying influential parameters in
high-dimensional models with relatively low computational cost.

The method constructs trajectories through the parameter space where each trajectory
consists of $D+1$ points (where $D$ is the number of input variables),
and consecutive points differ in exactly one coordinate. This allows for efficient
estimation of sensitivity measures with fewer model evaluations compared to
full-factorial or Monte Carlo approaches.

Morris samples can be created using the following simple syntax:

```python
>>> morris_samples = morris_sampling(num_vars=3, N=10, num_levels=4, seed=128)
```

This creates 10 Morris trajectories for 3 variables with samples in the `[0, 1]` range,
using a 4-level grid. The resulting sample matrix has shape $(N \times (D+1), D) = (30, 3)$.

Available keyword arguments:

- `num_vars` (int): Number of input variables (dimensionality of the problem)
- `N` (int): Number of trajectories to generate
- `num_levels` (int): Number of levels in the grid (must be even, default=4)
- `seed` (int, optional): Random seed for reproducibility

### Usage Examples

Basic usage with 2 variables:

```python
>>> import numpy as np
>>> from pydoe import morris_sampling
>>> samples = morris_sampling(num_vars=2, N=5, num_levels=4, seed=128)
>>> samples.shape
(15, 2)
```

The Morris method with different grid levels:

```python
>>> # Using 6 levels instead of 4
>>> samples_6 = morris_sampling(num_vars=2, N=5, num_levels=6, seed=128)
>>> samples_6.shape
(15, 2)
```

High-dimensional example:

```python
>>> # 10-dimensional problem
>>> samples_10d = morris_sampling(num_vars=10, N=20, num_levels=4)
>>> samples_10d.shape
(220, 10)  # 20 trajectories * 11 points per trajectory
```

### Morris Method Theory

The Morris method is based on computing elementary effects for each input variable.
For a model $f(\mathbf{x})$ where $\mathbf{x} = (x_1, x_2, \ldots, x_D)$,
the elementary effect of the $i$-th variable is defined as:

$$
EE_i = \frac{f(x_1, \ldots, x_i + \Delta, \ldots, x_D) - f(x_1, \ldots, x_i, \ldots, x_D)}{\Delta}
$$

where $\Delta$ is the step size determined by the grid level.

The Morris method estimates two sensitivity measures:

1. **Mean of elementary effects** ($\mu_i$): Indicates the overall influence of variable $i$
2. **Standard deviation of elementary effects** ($\sigma_i$): Indicates non-linear effects and interactions

### Advantages of the Morris Method

- Computationally efficient: requires only $N \times (D+1)$ model evaluations
- Suitable for high-dimensional problems
- Provides screening of important variables
- Good for preliminary sensitivity analysis
- Returns samples in `[0, 1]` hypercube that can be easily transformed to any bounds

## Saltelli Sampling (`saltelli_sampling`) {#saltelli_sampling}

Saltelli sampling is a specialized sampling scheme designed for computing Sobol'
sensitivity indices, which are variance-based global sensitivity measures. This
method is based on Sobol' sequences and provides an efficient way to estimate
first-order, total-order, and optionally second-order Sobol' indices.

Unlike the Morris method which provides qualitative screening, Saltelli sampling
enables quantitative variance-based sensitivity analysis. It uses quasi-random
low-discrepancy Sobol' sequences which provide better convergence properties
compared to random sampling methods.

Saltelli samples can be created using the following simple syntax:

```python
>>> saltelli_samples = saltelli_sampling(num_vars=3, N=1024,
                                         calc_second_order=True, seed=128)
```

This creates Saltelli samples for 3 variables with 1024 base samples, including
second-order interaction terms. The resulting sample matrix has shape 
$(N \times (2D+2), D) = (8192, 3)$. All samples are in the `[0, 1]` range.

Available keyword arguments:

- `num_vars` (int): Number of input variables (dimensions)
- `N` (int): Base sample size (ideally a power of 2 for optimal Sobol' sequence properties)
- `calc_second_order` (bool): Include second-order interaction terms (default=True)
- `skip_values` (int, optional): Number of Sobol' points to skip (set automatically if None)
- `scramble` (bool): Whether to use scrambling for Sobol' sequence (default=False)
- `seed` (int, optional): Random seed (only used if scramble=True)

### Usage Examples

Basic usage with first and total-order indices:

```python
>>> import numpy as np
>>> from pydoe import saltelli_sampling
>>> samples = saltelli_sampling(num_vars=3, N=1024, seed=128)
>>> samples.shape
(8192, 3)  # (1)!
```

1. N * (2 * D + 2) = 1024 * 8

First-order indices only (excluding second-order interactions):

```python
>>> samples_first = saltelli_sampling(num_vars=3, N=1024, 
...                                   calc_second_order=False, seed=128)
>>> samples_first.shape
(5120, 3)  # (1)!
```

1. N * (D + 2) = 1024 * 5

Using scrambled Sobol' sequences:

```python
>>> samples_scrambled = saltelli_sampling(num_vars=3, N=1024,
...                                       scramble=True, seed=128)
```

High-dimensional example:

```python
>>> # 8-dimensional problem
>>> samples_8d = saltelli_sampling(num_vars=8, N=2048)
>>> samples_8d.shape
(36864, 8)  # (1)!
```

1. 2048 * (2*8 + 2) = 2048 * 18

### Saltelli Sampling Theory

The Saltelli sampling scheme is designed to efficiently compute Sobol' indices, which
decompose the total variance of a model output into contributions from individual
variables and their interactions: $$
\text{Var}(Y) = \sum_{i} V_i + \sum_{i<j} V_{ij} + \ldots + V_{1,2,\ldots,D}
$$
where $V_i$ represents the first-order effect of variable $i$, and
$V_{ij}$ represents the second-order interaction between variables $i$ and $j$.

### Sobol' Indices

1. **First-order index**: $S_i = \frac{V_i}{\text{Var}(Y)}$
2. **Total-order index**: $S_{T_i} = \frac{E_{\mathbf{x}_{\sim i}}[\text{Var}_{x_i}(Y|\mathbf{x}_{\sim i})]}{\text{Var}(Y)}$
3. **Second-order index**: $S_{ij} = \frac{V_{ij}}{\text{Var}(Y)}$

### Sample Matrix Structure

For $D$ variables and $N$ base samples, the Saltelli method generates:

- **With second-order terms**: $N \times (2D + 2)$ samples
- **First-order only**: $N \times (D + 2)$ samples

The matrix consists of:

1. $N$ samples from matrix **A**
2. $N \times D$ samples from matrix **A** with individual columns replaced by **B**
3. (Optional) $N \times D$ samples from matrix **B** with individual columns replaced by **A**
4. $N$ samples from matrix **B**

### Advantages of Saltelli Sampling

- Provides quantitative sensitivity measures (Sobol' indices)
- Uses low-discrepancy sequences for better convergence
- Enables computation of interaction effects
- Well-established theoretical foundation
- Suitable for variance-based global sensitivity analysis

### Computational Requirements

- **First-order + Total-order**: $N \times (D + 2)$ model evaluations
- **Including second-order**: $N \times (2D + 2)$ model evaluations

### Important Notes

- $N$ should ideally be a power of 2 for optimal Sobol' sequence properties
- Larger $N$ values provide more accurate sensitivity estimates
- The method requires more model evaluations than Morris but provides quantitative results
- Suitable for models where variance-based sensitivity analysis is needed
- The implementation automatically skips initial Sobol' sequence points to improve quality
- All samples are returned in the `[0,1]` hypercube and should be transformed to desired bounds

## Iman-Conover Method (`iman_conover`) {#iman_conover}

The **Iman-Conover method** induces a target rank correlation structure
among columns of a sample matrix without changing the marginal
distribution of any column. It is widely used in Monte Carlo simulation
and sensitivity analysis to model correlated uncertain parameters that
have been sampled independently from arbitrary marginal distributions.

Given a sample matrix `data` of shape `(n, k)` (each column drawn from
its own marginal distribution) and a target correlation matrix
`correlation_matrix` of shape `(k, k)`, the method reorders the rows of
each column so that the resulting matrix has approximately the requested
Spearman rank correlation structure:

```python
>>> iman_conover(data, correlation_matrix, seed=None)
```

- `data`: array of shape `(n, k)` — `n` samples of `k` variables.
- `correlation_matrix`: array of shape `(k, k)` — target correlation
  matrix, must be symmetric positive definite.
- `seed`: optional `int` or `numpy.random.Generator` for reproducibility.

### Usage Example

```python
>>> import numpy as np
>>> from pydoe import iman_conover
>>> rng = np.random.default_rng(0)
>>> data = rng.normal(size=(1000, 2))
>>> target = np.array([[1.0, 0.8], [0.8, 1.0]])
>>> reordered = iman_conover(data, target, seed=0)
>>> reordered.shape
(1000, 2)
```

!!! note
    The marginal distribution of each column of `data` is preserved
    exactly — only the *order* of the values within each column changes.
    Only the rank correlation between columns is altered to approximate
    `correlation_matrix`.

## References

- [Campolongo, F., Cariboni, J., & Saltelli, A. (2007). An effective screening design for sensitivity analysis of large models. *Environmental Modelling & Software*, 22(10), 1509-1518.](https://doi.org/10.1016/j.envsoft.2006.10.004)
- [Campolongo, F., Saltelli, A., & Cariboni, J. (2011). From screening to quantitative sensitivity analysis. A unified approach. *Computer Physics Communications*, 182, 978-988.](https://doi.org/10.1016/j.cpc.2010.12.039)
- [Iman, R. L., and Conover, W. J. (1982). A distribution-free approach to inducing rank correlation among input variables. *Communications in Statistics - Simulation and Computation*, 11(3), 311-334.](https://doi.org/10.1080/03610918208812265)
- [Morris, M. D. (1991). Factorial sampling plans for preliminary computational experiments. *Technometrics*, 33(2), 161-174.](https://doi.org/10.1080/00401706.1991.10484804)
- [Saltelli, A. (2002). Making best use of model evaluations to compute sensitivity indices. *Computer Physics Communications*, 145(2), 280-297.](https://doi.org/10.1016/S0010-4655(02)00280-1)
- [Sobol', I. M. (2001). Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates. *Mathematics and Computers in Simulation*, 55(1-3), 271-280.](https://doi.org/10.1016/S0378-4754(00)00270-6)