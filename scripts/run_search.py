#!/usr/bin/env python3
"""
Run async hyperparameter search with SHA early stopping.

Usage:
    # Random search, MLP trainer (default)
    python scripts/run_search.py --n-trials 12 --max-steps 80

    # Grid search
    python scripts/run_search.py --strategy grid

    # RL trainer (SAC hyperparams)
    python scripts/run_search.py --trainer rl --n-trials 12

    # From config
    python scripts/run_search.py --config configs/search_config.yaml
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import trio
import yaml

from src.search.searcher import SearchConfig, run_search
from src.strategies.successive_halving import SHAConfig
from src.strategies.grid_random import grid_search, random_search
from src.trainers.base_trainer import SyntheticMLPTrainer, SyntheticRLTrainer
from src.utils.logging import setup_logging

# ---------------------------------------------------------------------------
# Default parameter spaces
# ---------------------------------------------------------------------------

MLP_RANDOM_SPACE = {
    "learning_rate": {"type": "log_uniform", "low": 1e-5, "high": 1e-1},
    "hidden_dim":    {"type": "int_log",     "low": 16,   "high": 512},
    "dropout":       {"type": "uniform",     "low": 0.0,  "high": 0.5},
    "weight_decay":  {"type": "log_uniform", "low": 1e-6, "high": 1e-2},
}

MLP_GRID_SPACE = {
    "learning_rate": [1e-4, 1e-3, 1e-2],
    "hidden_dim":    [32, 128, 256],
    "dropout":       [0.0, 0.1, 0.3],
    "weight_decay":  [1e-5, 1e-4],
}

RL_RANDOM_SPACE = {
    "learning_rate": {"type": "log_uniform", "low": 1e-5, "high": 1e-2},
    "tau":           {"type": "log_uniform", "low": 1e-3, "high": 1e-1},
    "batch_size":    {"type": "int_log",     "low": 64,   "high": 1024},
    "gamma":         {"type": "uniform",     "low": 0.95, "high": 0.999},
}


def parse_args():
    p = argparse.ArgumentParser(description="Async Hyperparameter Search")
    p.add_argument("--config", type=Path)
    p.add_argument("--strategy", choices=["random", "grid"], default="random")
    p.add_argument("--trainer", choices=["mlp", "rl"], default="mlp")
    p.add_argument("--n-trials", type=int, default=12)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--max-concurrent", type=int, default=4)
    p.add_argument("--eta", type=int, default=3, help="SHA reduction factor")
    p.add_argument("--min-steps", type=int, default=10, help="SHA first rung")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("outputs/trials/results.json"))
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)

    if args.config and args.config.exists():
        with args.config.open() as f:
            cfg = yaml.safe_load(f)
        sha = SHAConfig(**cfg.get("sha", {}))
        search_config = SearchConfig(
            max_steps=cfg.get("max_steps", 80),
            sha=sha,
            max_concurrent_trials=cfg.get("max_concurrent_trials", 4),
        )
        strategy = cfg.get("strategy", "random")
        trainer_type = cfg.get("trainer", "mlp")
        n_trials = cfg.get("n_trials", 12)
        seed = cfg.get("seed", 42)
        space = cfg.get("param_space", MLP_RANDOM_SPACE)
    else:
        sha = SHAConfig(max_steps=args.max_steps, min_steps=args.min_steps, eta=args.eta)
        search_config = SearchConfig(
            max_steps=args.max_steps,
            sha=sha,
            max_concurrent_trials=args.max_concurrent,
        )
        strategy = args.strategy
        trainer_type = args.trainer
        n_trials = args.n_trials
        seed = args.seed
        space = RL_RANDOM_SPACE if trainer_type == "rl" else MLP_RANDOM_SPACE

    # Build hyperparameter configurations
    if strategy == "grid":
        grid_space = MLP_GRID_SPACE if trainer_type == "mlp" else MLP_GRID_SPACE
        hp_configs = grid_search(grid_space)[:n_trials]
    else:
        hp_configs = random_search(space, n_trials=n_trials, seed=seed)

    # Select trainer
    trainer = SyntheticRLTrainer() if trainer_type == "rl" else SyntheticMLPTrainer()

    print(f"\nRunning {len(hp_configs)} trials ({strategy} search, {trainer_type} trainer)")
    print(f"SHA rungs: min_steps={sha.min_steps}, eta={sha.eta}, max_steps={sha.max_steps}")
    print(f"Max concurrent: {search_config.max_concurrent_trials}\n")

    result = trio.run(run_search, hp_configs, trainer, search_config)

    # Print results
    print("\n" + "=" * 65)
    print("Search Complete")
    print("=" * 65)
    print(f"  Trials total     : {len(result.all_trials)}")
    print(f"  Completed        : {result.n_completed}")
    print(f"  Early-stopped    : {result.n_cancelled}")
    print(f"  Elapsed          : {result.elapsed_s:.1f}s")
    print(f"\n  Best trial       : #{result.best_trial.trial_id}")
    print(f"  Best metric      : {result.best_trial.best_metric:.4f}")
    print(f"  Best hyperparams :")
    for k, v in result.best_trial.hyperparams.items():
        print(f"    {k:20s} = {v}")

    print("\n  Top 5 trials:")
    print(f"  {'Rank':<5} {'Trial':<7} {'Metric':<10} {'Steps':<8} {'Cancelled'}")
    print("  " + "-" * 45)
    for rank, trial in enumerate(result.all_trials[:5], 1):
        print(f"  {rank:<5} #{trial.trial_id:<6} {trial.best_metric:<10.4f} "
              f"{trial.total_steps:<8} {'yes' if trial.was_cancelled else 'no'}")

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "best": {
            "trial_id": result.best_trial.trial_id,
            "metric": result.best_trial.best_metric,
            "hyperparams": result.best_trial.hyperparams,
        },
        "all_trials": [
            {
                "trial_id": t.trial_id,
                "best_metric": t.best_metric,
                "best_step": t.best_step,
                "total_steps": t.total_steps,
                "elapsed_s": round(t.elapsed_s, 3),
                "was_cancelled": t.was_cancelled,
                "hyperparams": t.hyperparams,
            }
            for t in result.all_trials
        ],
        "summary": {
            "n_trials": len(result.all_trials),
            "n_completed": result.n_completed,
            "n_cancelled": result.n_cancelled,
            "elapsed_s": round(result.elapsed_s, 3),
        },
    }
    args.output.write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved to: {args.output}")
    print("=" * 65)


if __name__ == "__main__":
    main()
