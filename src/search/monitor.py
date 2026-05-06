"""
Monitor task — receives TrialReports, applies SHA scheduling,
and sets cancel_events for underperforming trials.

This is the "brain" of the early stopping system.
It runs as a single long-lived Trio task, consuming from the shared
report channel and maintaining SHA state.
"""
from __future__ import annotations
import logging
from typing import Sequence

import trio

from .trial import TrialReport
from ..strategies.successive_halving import SuccessiveHalvingScheduler, SHAConfig

logger = logging.getLogger(__name__)


async def monitor_task(
    report_channel: trio.MemoryReceiveChannel,
    cancel_events: dict[int, trio.Event],
    sha_config: SHAConfig,
    n_trials: int,
    rung_decision_events: dict[int, trio.Event],
    *,
    task_status=trio.TASK_STATUS_IGNORED,
) -> dict:
    """
    Monitors all trial reports and applies SHA early stopping.

    cancel_events: {trial_id: trio.Event}
        Setting the event signals the trial to stop at its next rung check.

    rung_decision_events: {rung_step: trio.Event}
        Set after SHA decides at each rung — trials wait on these before
        checking their cancel_event.

    Returns SHA summary statistics.
    """
    scheduler = SuccessiveHalvingScheduler(sha_config, n_trials)
    task_status.started()

    total_reports = 0
    total_cancellations = 0

    async with report_channel:
        async for report in report_channel:
            total_reports += 1

            # Ask scheduler if any trials should be cancelled at this rung
            to_cancel = scheduler.report(
                report.trial_id, report.step, report.metric
            )

            for tid in to_cancel:
                if tid in cancel_events and not cancel_events[tid].is_set():
                    logger.info(
                        "Monitor cancelling trial %d (SHA decision at step %d)",
                        tid, report.step,
                    )
                    cancel_events[tid].set()
                    total_cancellations += 1

            # Signal trials waiting at this rung that the decision is made.
            if scheduler.has_decided_at(report.step):
                evt = rung_decision_events.get(report.step)
                if evt is not None and not evt.is_set():
                    evt.set()

    summary = scheduler.summary()
    summary["total_reports"] = total_reports
    summary["total_cancellations"] = total_cancellations
    logger.info(
        "Monitor done. %d trials | %d cancelled | %d reports processed",
        n_trials, total_cancellations, total_reports,
    )
    return summary
