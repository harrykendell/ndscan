from __future__ import annotations

from .base import (
    ALGORITHM_REGISTRY,
    AlgorithmParameter,
    ObjectiveSpec,
    Optimizer,
    OptimizeAcquisitionSpec,
    OptimizeAlgorithmSpec,
    OptimizeAxis,
    OptimizeSpec,
    build_algorithm_spec,
)
from .coordinate_search import (
    CoordinateSearchOptimizeAlgorithmSpec,
    CoordinateSearchOptimizer,
)
from .bayesian import BayesianOptimizerOptimizeAlgorithmSpec, BayesianOptimizer
from .nelder_mead import NelderMeadOptimizeAlgorithmSpec, NelderMeadOptimizer

__all__ = [
    "ALGORITHM_REGISTRY",
    "AlgorithmParameter",
    "BayesianOptimizerOptimizeAlgorithmSpec",
    "BayesianOptimizer",
    "CoordinateSearchOptimizeAlgorithmSpec",
    "CoordinateSearchOptimizer",
    "NelderMeadOptimizeAlgorithmSpec",
    "NelderMeadOptimizer",
    "ObjectiveSpec",
    "Optimizer",
    "OptimizeAcquisitionSpec",
    "OptimizeAlgorithmSpec",
    "OptimizeAxis",
    "OptimizeSpec",
    "build_algorithm_spec",
]

# To add an optimizer algorithm:
# 1. Create a new file for the algorithm
# 2a. Implement an OptimizeAlgorithmSpec dataclass for the algorithm's parameters.
# 2b. Implement an Optimizer subclass for the algorithm. Ensuring it follows the expected interface.
# 2c. Register the algorithm using register_algorithm() with appropriate metadata and parameters for its initialization.
# 3. Add the new classes to __all__ and imports above and in optimize.py.
