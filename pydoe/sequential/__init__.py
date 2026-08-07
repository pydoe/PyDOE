from .acquisition import (
    expected_improvement,
    probability_of_improvement,
    upper_confidence_bound,
)
from .adaptive import sequential_design
from .gaussian_process import GaussianProcessRegressor


__all__ = [
    "GaussianProcessRegressor",
    "expected_improvement",
    "probability_of_improvement",
    "sequential_design",
    "upper_confidence_bound",
]
