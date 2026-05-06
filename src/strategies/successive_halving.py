"""
Successive Halving (SHA) early stopping strategy.

At each rung (milestone step), SHA keeps only the top 1/eta fraction of
trials. The rest are cancelled via Trio cancel scopes.

Rungs example (max_steps=100, eta=3, min_steps=10):
    Rung 0:  step 10  — keep top 1/3  (33%)
    Rung 1:  step 30  — keep top 1/9  (11%)
    Rung 2:  step 90  — full run

This is also the foundation of Hyperband (runs multiple SHA brackets
with different resource budgets).
"""
from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Rung:
    step: int
    keep_fraction: float      # fraction of trials to keep past this rung


@dataclass
class SHAConfig:
    max_steps: int = 100
    min_steps: int = 10       # first rung
    eta: int = 3              # reduction factor (keep 1/eta each rung)


def build_rungs(config: SHAConfig) -> list[Rung]:
    """Compute rung schedule from config."""
    rungs = []
    n_rungs = math.floor(math.log(config.max_steps / config.min_steps, config.eta)) + 1
    for i in range(n_rungs):
        step = int(config.min_steps * (config.eta ** i))
        keep = 1.0 / (config.eta ** (i + 1))
        rungs.append(Rung(step=min(step, config.max_steps), keep_fraction=keep))
    return rungs


class SuccessiveHalvingScheduler:
    """
    Tracks running metrics for all trials and decides which to cancel.

    Usage:
        scheduler = SuccessiveHalvingScheduler(config, n_trials)
        # After each trial reports a metric at a given step:
        cancelled = scheduler.report(trial_id, step, metric)
        # cancelled is a set of trial_ids that should be stopped.
    """

    def __init__(self, config: SHAConfig, n_trials: int):
        self.config = config
        self.n_trials = n_trials
        self.rungs = build_rungs(config)
        self._best_at_rung: dict[int, list[tuple[float, int]]] = {
            r.step: [] for r in self.rungs
        }
        self._cancelled: set[int] = set()
        self._promoted: set[int] = set()
        self._decisions_made: set[int] = set()

    @property
    def rung_steps(self) -> set[int]:
        return {r.step for r in self.rungs}

    def report(self, trial_id: int, step: int, metric: float) -> set[int]:
        """
        Called when a trial reports a metric at a given step.
        Returns the set of trial_ids that should be cancelled NOW.
        """
        if step not in self.rung_steps:
            return set()

        rung = next(r for r in self.rungs if r.step == step)
        self._best_at_rung[step].append((metric, trial_id))

        # Only decide when we have enough data to compare
        # (all trials that haven't been cancelled yet have reported)
        active = self.n_trials - len(self._cancelled)
        reported_this_rung = len(self._best_at_rung[step])

        if reported_this_rung < active:
            return set()  # wait for more trials to report

        # Sort by metric (lower is better), keep top keep_fraction
        sorted_trials = sorted(self._best_at_rung[step], key=lambda x: x[0])
        n_keep = max(1, math.ceil(len(sorted_trials) * rung.keep_fraction))
        to_keep = {tid for _, tid in sorted_trials[:n_keep]}
        to_cancel = {tid for _, tid in sorted_trials[n_keep:]} - self._cancelled

        self._cancelled.update(to_cancel)
        self._promoted.update(to_keep)
        self._decisions_made.add(step)

        logger.info(
            "Rung step=%d | kept=%d trials | cancelled=%d trials | "
            "best_metric=%.4f | worst_kept=%.4f",
            step, len(to_keep), len(to_cancel),
            sorted_trials[0][0], sorted_trials[n_keep - 1][0],
        )

        return to_cancel

    def has_decided_at(self, step: int) -> bool:
        return step in self._decisions_made

    def is_cancelled(self, trial_id: int) -> bool:
        return trial_id in self._cancelled

    def summary(self) -> dict:
        return {
            "n_trials": self.n_trials,
            "n_cancelled": len(self._cancelled),
            "n_completed": self.n_trials - len(self._cancelled),
            "cancelled_ids": sorted(self._cancelled),
        }
