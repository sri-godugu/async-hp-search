import pytest
import trio

from src.search.searcher import SearchConfig, run_search
from src.strategies.successive_halving import SHAConfig, build_rungs, SuccessiveHalvingScheduler
from src.strategies.grid_random import grid_search, random_search
from src.trainers.base_trainer import SyntheticMLPTrainer, SyntheticRLTrainer


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------

def test_grid_search():
    space = {"lr": [0.01, 0.001], "hidden": [32, 64]}
    configs = grid_search(space)
    assert len(configs) == 4
    assert all("lr" in c and "hidden" in c for c in configs)


def test_random_search():
    space = {
        "learning_rate": {"type": "log_uniform", "low": 1e-5, "high": 1e-1},
        "hidden_dim": {"type": "int_log", "low": 16, "high": 256},
        "dropout": {"type": "uniform", "low": 0.0, "high": 0.5},
    }
    configs = random_search(space, n_trials=10, seed=0)
    assert len(configs) == 10
    for c in configs:
        assert 1e-5 <= c["learning_rate"] <= 1e-1
        assert 0.0 <= c["dropout"] <= 0.5


def test_build_rungs():
    sha = SHAConfig(max_steps=100, min_steps=10, eta=3)
    rungs = build_rungs(sha)
    assert rungs[0].step == 10
    assert all(r.step <= 100 for r in rungs)
    assert rungs[0].keep_fraction < 1.0


def test_sha_cancellation():
    sha = SHAConfig(max_steps=30, min_steps=10, eta=3)
    scheduler = SuccessiveHalvingScheduler(sha, n_trials=3)

    # Report rung step=10 for all 3 trials: worst should be cancelled
    cancelled = set()
    cancelled |= scheduler.report(0, 10, 0.5)   # best
    cancelled |= scheduler.report(1, 10, 1.5)   # worst
    cancelled |= scheduler.report(2, 10, 1.0)   # middle

    # With eta=3, keep top 1/3 = 1 trial, cancel 2
    assert len(cancelled) == 2
    assert 0 not in cancelled  # best trial should survive


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------

def test_mlp_trainer_yields_steps():
    async def _run():
        from src.trainers.base_trainer import TrialConfig
        trainer = SyntheticMLPTrainer()
        config = TrialConfig(trial_id=0, hyperparams={"learning_rate": 1e-3,
                                                       "hidden_dim": 64,
                                                       "dropout": 0.1,
                                                       "weight_decay": 1e-4},
                             max_steps=10)
        steps = []
        async for result in trainer.train(config):
            steps.append(result)
        assert len(steps) == 10
        assert all(r.metric > 0 for r in steps)
    trio.run(_run)


# ---------------------------------------------------------------------------
# Full search tests
# ---------------------------------------------------------------------------

def test_full_search_mlp():
    hp_configs = random_search({
        "learning_rate": {"type": "log_uniform", "low": 1e-4, "high": 1e-1},
        "hidden_dim": {"type": "int_log", "low": 16, "high": 128},
        "dropout": {"type": "uniform", "low": 0.0, "high": 0.3},
        "weight_decay": {"type": "log_uniform", "low": 1e-5, "high": 1e-3},
    }, n_trials=6, seed=0)

    sha = SHAConfig(max_steps=30, min_steps=10, eta=3)
    # max_concurrent_trials == n_trials so all hit each rung together,
    # letting SHA collect all reports and decide before any trial advances.
    config = SearchConfig(max_steps=30, sha=sha, max_concurrent_trials=6)

    result = trio.run(run_search, hp_configs, SyntheticMLPTrainer(), config)

    assert result.best_trial is not None
    assert len(result.all_trials) == 6
    assert result.n_completed + result.n_cancelled == 6
    assert result.n_cancelled > 0   # SHA should have cancelled some


def test_full_search_rl():
    hp_configs = random_search({
        "learning_rate": {"type": "log_uniform", "low": 1e-5, "high": 1e-2},
        "tau": {"type": "log_uniform", "low": 1e-3, "high": 1e-1},
        "batch_size": {"type": "int_log", "low": 64, "high": 512},
        "gamma": {"type": "uniform", "low": 0.95, "high": 0.999},
    }, n_trials=6, seed=1)

    sha = SHAConfig(max_steps=30, min_steps=10, eta=3)
    # max_concurrent_trials == n_trials so all hit each rung together.
    config = SearchConfig(max_steps=30, sha=sha, max_concurrent_trials=6)

    result = trio.run(run_search, hp_configs, SyntheticRLTrainer(), config)
    assert result.best_trial is not None
    # Best RL metric (negated reward) should be negative
    assert result.best_trial.best_metric < 0


def test_best_hyperparams_win():
    """Optimal hyperparams should consistently end up in top trials."""
    # Mix optimal with clearly suboptimal
    hp_configs = [
        {"learning_rate": 1e-3, "hidden_dim": 128, "dropout": 0.1, "weight_decay": 1e-4},  # optimal
        {"learning_rate": 1e-1, "hidden_dim": 16,  "dropout": 0.5, "weight_decay": 1e-1},  # bad
        {"learning_rate": 1e-1, "hidden_dim": 16,  "dropout": 0.5, "weight_decay": 1e-1},  # bad
        {"learning_rate": 1e-3, "hidden_dim": 128, "dropout": 0.1, "weight_decay": 1e-4},  # optimal
    ]
    sha = SHAConfig(max_steps=30, min_steps=10, eta=2)
    config = SearchConfig(max_steps=30, sha=sha, max_concurrent_trials=4)

    result = trio.run(run_search, hp_configs, SyntheticMLPTrainer(), config)
    # Best trial should have low metric (good hyperparams win)
    assert result.best_trial.best_metric < 1.0
