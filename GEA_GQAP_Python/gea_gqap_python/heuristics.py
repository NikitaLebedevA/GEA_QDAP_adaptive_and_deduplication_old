from __future__ import annotations

import numpy as np

from .models import Individual, Model
from .utils import evaluate_permutation


def heuristic2(model: Model) -> Individual:
    I, J = model.I, model.J
    X = np.zeros((I, J), dtype=int)
    count = np.zeros(I, dtype=float)

    CT = np.zeros((I, J), dtype=float)
    for i in range(I):
        for j in range(J):
            CT[i, j] = model.cij[i, j] + model.DIS[i].sum() + model.F[j].sum()

    for j in range(J):
        candidates = CT[:, j]
        order = np.argsort(candidates)
        assigned = False
        for idx in order:
            if count[idx] + model.aij[idx, j] <= model.bi[idx]:
                X[idx, j] = 1
                count[idx] += model.aij[idx, j]
                assigned = True
                break
        if not assigned:
            idx = order[0]
            X[idx, j] = 1
            count[idx] += model.aij[idx, j]

    # Repair feasibility if necessary
    for i in range(I):
        while count[i] > model.bi[i] + 1e-9:
            assigned_jobs = np.where(X[i] == 1)[0]
            if assigned_jobs.size == 0:
                break
            job_to_move = assigned_jobs[np.argmax(model.aij[i, assigned_jobs])]
            X[i, job_to_move] = 0
            count[i] -= model.aij[i, job_to_move]
            target = np.argsort(model.aij[:, job_to_move])
            for new_i in target:
                if new_i == i:
                    continue
                if count[new_i] + model.aij[new_i, job_to_move] <= model.bi[new_i] + 1e-9:
                    X[new_i, job_to_move] = 1
                    count[new_i] += model.aij[new_i, job_to_move]
                    break
            else:
                X[i, job_to_move] = 1
                count[i] += model.aij[i, job_to_move]
                break

    permutation = np.argmax(X, axis=0)
    return evaluate_permutation(permutation, model)

