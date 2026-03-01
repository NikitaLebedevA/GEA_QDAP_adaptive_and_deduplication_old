from __future__ import annotations

import numpy as np

from gea_gqap_python import AlgorithmConfig, load_model, run_ga


def test_model_loader_smoke():
    model = load_model("c201535")
    assert model.cij.shape == (model.I, model.J)
    assert model.DIS.shape == (model.I, model.I)
    assert model.F.shape == (model.J, model.J)


def test_run_ga_produces_finite_cost():
    model = load_model("c201535")
    config = AlgorithmConfig(
        iterations=5,
        population_size=30,
        crossover_rate=0.6,
        mutation_rate=0.3,
        random_seed=42,
        time_limit=30.0,
    )
    result = run_ga(model, config=config)

    assert np.isfinite(result.best_cost)
    assert result.best_cost == min(result.stats.best_cost_trace)
    assert result.stats.best_cost_trace, "Cost trace should not be empty"

