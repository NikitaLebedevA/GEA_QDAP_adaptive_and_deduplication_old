from __future__ import annotations

from dataclasses import replace
from typing import Tuple

import numpy as np

from .models import Individual, Model


def create_xij(permutation: np.ndarray, model: Model) -> np.ndarray:
    x = np.zeros((model.I, model.J), dtype=int)
    x[permutation, np.arange(model.J)] = 1
    return x


def cost_function(x: np.ndarray, model: Model) -> Tuple[float, np.ndarray]:
    x_float = x.astype(float, copy=False)
    loads = (model.aij * x_float).sum(axis=1)
    cvar = model.bi - loads
    if np.any(cvar < 0):
        return float("inf"), cvar

    c1 = float(np.sum(model.cij * x_float))
    temp = np.einsum("ij,ik,kl->jl", x_float, model.DIS, x_float)
    c2 = float(np.sum(temp * model.F))
    return c1 + c2, cvar


def evaluate_permutation(permutation: np.ndarray, model: Model) -> Individual:
    x = create_xij(permutation, model)
    cost, cvar = cost_function(x, model)
    return Individual(permutation=permutation.copy(), xij=x, cost=cost, cvar=cvar)


def clone_individual(individual: Individual) -> Individual:
    return Individual(
        permutation=individual.permutation.copy(),
        xij=individual.xij.copy(),
        cost=individual.cost,
        cvar=individual.cvar.copy(),
    )

