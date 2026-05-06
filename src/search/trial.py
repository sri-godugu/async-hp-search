"""
Individual trial runner.

Each trial is a Trio task that:
  1. Runs the trainer (async generator, yields StepResult each step)
  2. Reports metrics to the monitor via a memory channel
  3. Checks at each rung whether it has been cancelled
  4. Uses a CancelScope so the monitor can cancel it from outside

The CancelScope is the key Trio primitive here:
    with trio.CancelScope() as scope:
        ...
        if scheduler.is_cancelled(trial_id):
            scope.cancel()
            return
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass

import trio

from ..trainers.base_trainer import TrialConfig, StepResult, BaseTrainer

logger = logging.getLogger(__name__)


@dataclass
class TrialReport:
    """Sent to the monitor after each step."""
    trial_id: int
    step: int
    metric: float
    elapsed_s: float
    cancelled: bool = False
    completed: bool = False


@dataclass
class TrialResult:
    trial_id: int
    hyperparams: dict
    best_metric: float
    best_step: int
    total_steps: int
    elapsed_s: float
    was_cancelled: bool


async def run_trial(
    config: TrialConfig,
    trainer: BaseTrainer,
    report_channel: trio.MemorySendChannel,
    cancel_event: trio.Event,       # set externally to cancel this trial
    rung_steps: set[int],
    rung_decision_events: dict[int, trio.Event],
    *,
    task_status=trio.TASK_STATUS_IGNORED,
) -> TrialResult:
    """
    Run a single trial.  Reports a TrialReport to report_channel after each step.
    Checks cancel_event at every rung step.
    """
    task_status.started()
    t0 = time.monotonic()
    best_metric = float("inf")
    best_step = 0
    steps_run = 0
    was_cancelled = False

    logger.debug("Trial %d starting: %s", config.trial_id, config.hyperparams)

    with trio.CancelScope() as scope:
        async for result in trainer.train(config):
            steps_run += 1

            if result.metric < best_metric:
                best_metric = result.metric
                best_step = result.step

            elapsed = time.monotonic() - t0

            # Send metric report to monitor
            try:
                await report_channel.send(TrialReport(
                    trial_id=config.trial_id,
                    step=result.step,
                    metric=result.metric,
                    elapsed_s=elapsed,
                ))
            except trio.ClosedResourceError:
                break

            # At rung boundaries: wait for the monitor to signal its decision
            # before checking cancel_event. This ensures the SHA scheduler has
            # seen all concurrent reports and set cancel events before we check.
            if result.step in rung_steps:
                await rung_decision_events[result.step].wait()

            # Check for cancellation at rung boundaries
            if result.step in rung_steps and cancel_event.is_set():
                logger.info(
                    "Trial %d cancelled at step %d (metric=%.4f)",
                    config.trial_id, result.step, result.metric,
                )
                was_cancelled = True
                scope.cancel()
                break

    elapsed = time.monotonic() - t0
    logger.debug(
        "Trial %d done. steps=%d best=%.4f cancelled=%s elapsed=%.1fs",
        config.trial_id, steps_run, best_metric, was_cancelled, elapsed,
    )

    return TrialResult(
        trial_id=config.trial_id,
        hyperparams=config.hyperparams,
        best_metric=best_metric,
        best_step=best_step,
        total_steps=steps_run,
        elapsed_s=elapsed,
        was_cancelled=was_cancelled,
    )
