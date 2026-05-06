"""
Grid search and random search hyperparameter generators.
Returns a list of hyperparameter dicts ready for trial execution.
"""
from __future__ import annotations
import itertools
import math
import random
from typing import Any


def grid_search(param_grid: dict[str, list[Any]]) -> list[dict]:
    """Cartesian product of all parameter lists."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def random_search(
    param_space: dict[str, Any],
    n_trials: int,
    seed: int = 42,
) -> list[dict]:
    """
    Sample n_trials configurations from a parameter space definition.

    param_space values can be:
        {"type": "choice", "values": [0.001, 0.01, 0.1]}
        {"type": "log_uniform", "low": 1e-5, "high": 1e-1}
        {"type": "uniform", "low": 0.0, "high": 0.5}
        {"type": "int", "low": 32, "high": 512}
        {"type": "int_log", "low": 32, "high": 512}
        A plain list → treated as {"type": "choice", "values": list}
    """
    rng = random.Random(seed)
    configs = []

    for _ in range(n_trials):
        config = {}
        for name, spec in param_space.items():
            if isinstance(spec, list):
                config[name] = rng.choice(spec)
            elif spec["type"] == "choice":
                config[name] = rng.choice(spec["values"])
            elif spec["type"] == "log_uniform":
                log_low = math.log(spec["low"])
                log_high = math.log(spec["high"])
                config[name] = math.exp(rng.uniform(log_low, log_high))
            elif spec["type"] == "uniform":
                config[name] = rng.uniform(spec["low"], spec["high"])
            elif spec["type"] == "int":
                config[name] = rng.randint(spec["low"], spec["high"])
            elif spec["type"] == "int_log":
                log_low = math.log2(spec["low"])
                log_high = math.log2(spec["high"])
                config[name] = int(2 ** rng.uniform(log_low, log_high))
            else:
                raise ValueError(f"Unknown param spec type: {spec['type']}")
        configs.append(config)

    return configs
