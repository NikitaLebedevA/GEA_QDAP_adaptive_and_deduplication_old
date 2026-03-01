# GEA_GQAP_Python

Python translation of the Generalized Evolutionary Algorithm (GEA) for the generalized quadratic assignment problem (GQAP). The original implementation was written in MATLAB; this port reuses the published instance definitions and reproduces the algorithmic operators in Python using `numpy`.

## Features

- Parser that converts the MATLAB `.m` data sets into Python data structures at runtime.
- Direct translation of the GA workflow, including crossover, mutation, and scenario-based operators.
- Heuristic initialisation routine replicated from the MATLAB `Heuristic2` function.
- Lightweight result object that exposes the best individual, cost trace, operator contribution share, and runtime.

## Quick start

```bash
cd GEA_GQAP_Python
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

To execute the GA on dataset `c351595`:

```python
from gea_gqap_python import load_model, run_ga

model = load_model("c351595")
result = run_ga(model)
print(result.best_cost)
```

## Tests

The test suite runs the algorithm for a handful of iterations to ensure the run completes and produces finite cost values. Execute the suite with `pytest`.

