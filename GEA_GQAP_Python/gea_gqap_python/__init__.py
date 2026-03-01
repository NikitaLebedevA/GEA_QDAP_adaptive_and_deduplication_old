"""
Python implementation of the Generalized Evolutionary Algorithm (GEA) for
generalized quadratic assignment problems (GQAP).

The package provides:
    - A parser that converts the original MATLAB `.m` dataset definitions to
      Python-friendly data structures.
    - Modular implementations of crossover, mutation, and specialized mask
      mutation operators.
    - A direct translation of the `Algorithm_GA_Quadratic` routine including
      the optional scenario-specific operators.

Example
-------
>>> from gea_gqap_python.algorithm import run_ga
>>> from gea_gqap_python.model_loader import load_model
>>> model = load_model(\"c351595\")
>>> best = run_ga(model, iterations=5, population_size=30, random_seed=123)
"""

from .algorithm import run_ga, AlgorithmConfig, AlgorithmResult
from .model_loader import load_model, list_available_models

__all__ = [
    "run_ga",
    "AlgorithmConfig",
    "AlgorithmResult",
    "load_model",
    "list_available_models",
]

