# Optimal Experimental Designs

The `pydoe.doe_optimal` module provides advanced algorithms for constructing
**optimal experimental designs** using a variety of optimality criteria and algorithms.
This is useful for maximizing the information gained from experiments while minimizing
the number of runs.

!!! hint
    All available optimal design tools can be accessed after a simple import statement:

    ```python
    >>> from pydoe import optimal_design, generate_candidate_set
    ```

## Overview

Optimal experimental design is based on maximizing or minimizing certain properties of the information matrix, typically denoted as M, which is defined as:
$$ M = X^T X $$
where $X$ is the model (design) matrix.

### Design Matrix (Model Matrix)

The design (model) matrix $X$ encodes the relationship between the model parameters and the input variables. For a polynomial model of degree $d$ with $k$ factors:
$$ \text{Linear:}\quad y = \beta_0 + \sum_{i=1}^k \beta_i x_i $$
$$ \text{Quadratic:}\quad y = \beta_0 + \sum_{i=1}^k \beta_i x_i + \sum_{i=1}^k \beta_{ii} x_i^2 + \sum_{i<j} \beta_{ij} x_i x_j $$

The design matrix X for n points and p parameters is constructed as:
$$
X = \begin{bmatrix}
1 & x_1^{(1)} & x_2^{(1)} & \cdots & (x_1^{(1)})^2 & x_1^{(1)} x_2^{(1)} & \cdots \\\\
1 & x_1^{(2)} & x_2^{(2)} & \cdots & (x_1^{(2)})^2 & x_1^{(2)} x_2^{(2)} & \cdots \\\\
\vdots & \vdots & \vdots & & \vdots & \vdots & \\\\
\end{bmatrix}
$$
where each row corresponds to a candidate point and each column to a model term.

### Efficiency Criteria

- **D-efficiency:**
  $$ \text{D-efficiency} = 100 \times (\det(M))^{1/p} $$
  $$ M = \frac{1}{n} X^T X $$
  where p is the number of parameters.

- **A-efficiency:**
  $$ \text{A-efficiency} = 100 \times \frac{p}{\operatorname{tr}(M^{-1})} $$
  $$ M = \frac{1}{n} X^T X $$
  where p is the number of parameters.

### Optimality Criteria

- **D-optimality**: Maximizes the determinant of the information matrix.
  $$ \text{D-optimality:}\quad \max \det(M) $$

- **A-optimality**: Minimizes the average variance of the parameter estimates (trace of the inverse information matrix).
  $$ \text{A-optimality:}\quad \min \operatorname{tr}(M^{-1}) $$

- **I-optimality**: Minimizes the average prediction variance over the candidate set.
  $$ \text{I-optimality:}\quad \min \frac{1}{|\mathcal{X}|} \sum_{x \in \mathcal{X}} x^T M^{-1} x $$

- **C-optimality**: Minimizes the variance of a linear combination of parameters.
  $$ \text{C-optimality:}\quad \min c^T M^{-1} c $$

- **E-optimality**: Maximizes the smallest eigenvalue of the information matrix.
  $$ \text{E-optimality:}\quad \max \lambda_{\min}(M) $$

- **G-optimality**: Minimizes the maximum prediction variance over the design space.
  $$ \text{G-optimality:}\quad \min \max_{x \in \mathcal{X}} x^T M^{-1} x $$

- **V-optimality**: Minimizes the average variance at specific points.
  $$ \text{V-optimality:}\quad \min \frac{1}{n} \sum_{i=1}^n x_i^T M^{-1} x_i $$

- **S-optimality**: Maximizes mutual orthogonality (various definitions, often based on maximizing the sum of squared off-diagonal elements of M).

- **T-optimality**: Model discrimination (maximizes the ability to distinguish between models).

### Algorithms

- **Sequential (Dykstra)**
- **Simple Exchange (Wynn-Mitchell)**
- **Fedorov**
- **Modified Fedorov**
- **DETMAX**

!!! note "Numerical conditioning during optimization"
    During iterative search, algorithms evaluate candidate designs that may
    produce an ill-conditioned information matrix (e.g. when too few distinct
    points span the model space). PyDOE handles this automatically: when the
    matrix is ill-conditioned, a Tikhonov regularization term $\lambda I$ is
    added before inverting, ensuring smooth criterion values throughout the
    search. The regularization strength is $\lambda = 10^{-8}$ by default and
    only applies to intermediate states — the final converged design is
    well-conditioned for the chosen model degree.

## Example Usage

Generate a D-optimal design for a quadratic model with 2 factors:

```python
>>> import numpy as np
>>> from pydoe import optimal_design, generate_candidate_set
>>> candidates = generate_candidate_set(n_factors=2, n_levels=5)
>>> design, info = optimal_design(
...     candidates=candidates,
...     n_points=10,
...     degree=2,
...     criterion="D",
...     method="detmax"
... )
>>> print(f"D-efficiency: {info['D_eff']:.2f}%")
```

After an optimal design is selected and experiments are performed, we can model our system by estimating the regression parameters using:
$$ \hat{\beta} = (X^{T} X)^{-1} X^{T} y $$
where X is the design matrix and y is the vector of observed responses.

## References

- Atkinson, A. C., & Donev, A. N. (1992). *Optimum Experimental Designs*. Oxford University Press.
- Fedorov, V. V. (1972). *Theory of Optimal Experiments*. Academic Press.
- Pukelsheim, F. (2006). *Optimal Design of Experiments*. SIAM.
- NIST: https://www.itl.nist.gov/div898/handbook/pri/section5/pri521.htm
- [Optimal experimental design](https://en.wikipedia.org/wiki/Optimal_experimental_design)

