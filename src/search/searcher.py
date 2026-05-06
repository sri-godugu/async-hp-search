"""
Main search orchestration.

Architecture:
                          ┌─────────────────────────────────┐
                          │         Trio Nursery             │
                          │                                  │
  hp_configs ──▶  [Trial 0] ──┐                             │
                 [Trial 1] ──┤──  TrialReport  ──▶ [Monitor]│
                 [Trial 2] ──┤   (memory channel)     │     │
                    ...   ──┘                         │     │
                                                cancel_events│
                                                 (trio.Event)│
                          └─────────────────────────────────┘

cancel_events[i].set() → Trial i checks at its next rung and stops.
All tasks share a nursery — if any crash, all are torn down cleanly.
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import trio

from .trial import run_trial, TrialResult, TrialReport
from .monitor import monitor_task
from ..trainers.base_trainer import BaseTrainer, TrialConfig
from ..strategies.successive_halving import SHAConfig, build_rungs

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    max_steps: int = 100
    sha: SHAConfig = field(default_factory=SHAConfig)
    max_concurrent_trials: int = 4
    report_buffer: int = 256


@dataclass
class SearchResult:
    best_trial: TrialResult
    all_trials: list[TrialResult]
    n_cancelled: int
    n_completed: int
    elapsed_s: float
    sha_summary: dict


async def run_search(
    hp_configs: list[dict],
    trainer: BaseTrainer,
    search_config: SearchConfig,
) -> SearchResult:
    """
    Run concurrent hyperparameter search with SHA early stopping.
    Returns a SearchResult with the best trial and all statistics.
    """
    n_trials = len(hp_configs)
    rungs = build_rungs(search_config.sha)
    rung_steps = {r.step for r in rungs}

    report_send, report_recv = trio.open_memory_channel(search_config.report_buffer)
    cancel_events: dict[int, trio.Event] = {i: trio.Event() for i in range(n_trials)}
    rung_decision_events: dict[int, trio.Event] = {r.step: trio.Event() for r in rungs}
    limiter = trio.CapacityLimiter(search_config.max_concurrent_trials)

    trial_results: list[TrialResult] = []
    sha_summary: dict = {}

    t0 = time.monotonic()

    logger.info(
        "Search starting. %d trials | max_concurrent=%d | rungs=%s",
        n_trials,
        search_config.max_concurrent_trials,
        [r.step for r in rungs],
    )

    async with trio.open_nursery() as nursery:
        # Start monitor
        await nursery.start(
            monitor_task,
            report_recv,
            cancel_events,
            search_config.sha,
            n_trials,
            rung_decision_events,
        )

        # Start all trials concurrently (rate-limited by limiter)
        async def _run_one_trial(trial_id: int, hp: dict) -> None:
            async with limiter:
                config = TrialConfig(
                    trial_id=trial_id,
                    hyperparams=hp,
                    max_steps=search_config.max_steps,
                )
                async with report_send.clone() as ch:
                    result = await run_trial(
                        config, trainer, ch,
                        cancel_events[trial_id],
                        rung_steps,
                        rung_decision_events,
                    )
                trial_results.append(result)

        async with report_send:
            async with trio.open_nursery() as trial_nursery:
                for trial_id, hp in enumerate(hp_configs):
                    trial_nursery.start_soon(_run_one_trial, trial_id, hp)

        # After all trials finish, monitor's channel closes → monitor exits

    elapsed = time.monotonic() - t0
    completed = [r for r in trial_results if not r.was_cancelled]
    cancelled = [r for r in trial_results if r.was_cancelled]

    best = min(trial_results, key=lambda r: r.best_metric)

    logger.info(
        "Search complete. Best trial=%d metric=%.4f | completed=%d cancelled=%d | %.1fs",
        best.trial_id, best.best_metric, len(completed), len(cancelled), elapsed,
    )

    return SearchResult(
        best_trial=best,
        all_trials=sorted(trial_results, key=lambda r: r.best_metric),
        n_cancelled=len(cancelled),
        n_completed=len(completed),
        elapsed_s=elapsed,
        sha_summary=sha_summary,
    )
