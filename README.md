# Async Hyperparameter Search with Early Stopping

Concurrent hyperparameter search using **Trio** structured concurrency and **Successive Halving (SHA)** for early stopping. Underperforming trials are cancelled mid-run via Trio cancel scopes, freeing compute for promising ones.

```
hp_configs
    │
    ▼
┌──────────────────────────────────────────────────────┐
│                    Trio Nursery                      │
│                                                      │
│  [Trial 0] ──┐                                       │
│  [Trial 1] ──┤──  TrialReport  ──▶  [Monitor/SHA]   |
│  [Trial 2] ──┤   (memory channel)        │           │
│     ...    ──┘                    cancel_events      │
│                                   (trio.Event)       │
└──────────────────────────────────────────────────────┘
```

## Why Trio

- **Cancel scopes**: each trial wraps its training loop in a `CancelScope`. The monitor sets a `trio.Event` at a rung boundary — the trial checks it and calls `scope.cancel()`. Clean, no thread kills, no zombie tasks.
- **Nursery**: all trials and the monitor share a nursery. If any task raises an exception, all others are cancelled and the exception propagates — no silent failures.
- **CapacityLimiter**: caps concurrent trials without a semaphore.

## Successive Halving (SHA)

At each rung (milestone step), SHA keeps the top 1/η fraction of trials and cancels the rest:

```
Rung 0 (step 10):  12 trials → keep top 4 (1/3), cancel 8
Rung 1 (step 30):   4 trials → keep top 1 (1/3), cancel 3
Rung 2 (step 90):   1 trial  → full run
```

This concentrates compute on the best configs while spending minimal steps on bad ones.

## Quick Start

```bash
pip install -r requirements.txt

# MLP trainer, random search (12 trials, SHA early stopping)
python scripts/run_search.py --n-trials 12 --max-steps 80

# RL trainer (SAC hyperparams)
python scripts/run_search.py --trainer rl --n-trials 12

# Grid search
python scripts/run_search.py --strategy grid

# From config
python scripts/run_search.py --config configs/search_config.yaml
```

## Run Tests

```bash
pytest tests/ -v
```

## Trainers

| Trainer | Hyperparameters | Metric |
|---|---|---|
| `mlp` | `learning_rate`, `hidden_dim`, `dropout`, `weight_decay` | validation loss (↓) |
| `rl` | `learning_rate`, `tau`, `batch_size`, `gamma` | negative episode return (↓) |

Both use synthetic simulation — no GPU required. Swap in your real trainer by implementing `BaseTrainer.train()` as an async generator.

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `max_steps` | 80 | Total training steps per trial |
| `min_steps` | 10 | First SHA rung |
| `eta` | 3 | SHA reduction factor (keep 1/η each rung) |
| `max_concurrent_trials` | 4 | Max parallel trials |
| `n_trials` | 12 | Total trials to run |
