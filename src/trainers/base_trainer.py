"""
Abstract trainer interface and MLP trainer on synthetic regression.
Trainers are async generators that yield (step, metric) tuples,
allowing the scheduler to observe and cancel them mid-run.
"""
from __future__ import annotations
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncGenerator

import trio


@dataclass
class TrialConfig:
    trial_id: int
    hyperparams: dict
    max_steps: int = 100


@dataclass
class StepResult:
    trial_id: int
    step: int
    metric: float          # lower is better (validation loss)
    train_metric: float
    hyperparams: dict


class BaseTrainer(ABC):
    @abstractmethod
    def train(self, config: TrialConfig) -> AsyncGenerator[StepResult, None]:
        """Async generator: yields one StepResult per training step."""
        ...


# ---------------------------------------------------------------------------
# Synthetic MLP trainer — no GPU required
# ---------------------------------------------------------------------------

def _simulate_learning_curve(
    lr: float,
    hidden_dim: int,
    dropout: float,
    weight_decay: float,
    n_steps: int,
    noise: float = 0.02,
) -> list[float]:
    """
    Simulate a plausible validation loss curve without actual training.
    Optimal: lr~1e-3, hidden_dim~128, dropout~0.1, weight_decay~1e-4.
    """
    # Score how close hyperparams are to the optimum
    lr_score  = math.exp(-10 * (math.log10(lr) - math.log10(1e-3)) ** 2)
    dim_score = math.exp(-((hidden_dim - 128) / 128) ** 2)
    do_score  = math.exp(-((dropout - 0.1) / 0.2) ** 2)
    wd_score  = math.exp(-((math.log10(weight_decay + 1e-8) - math.log10(1e-4)) ** 2))

    quality = 0.4 * lr_score + 0.3 * dim_score + 0.15 * do_score + 0.15 * wd_score

    # Asymptotic loss: best config → ~0.05, worst → ~2.0
    final_loss = 2.0 - 1.95 * quality
    init_loss  = 2.0 + random.gauss(0, 0.1)

    # Exponential decay with noise
    curve = []
    for t in range(n_steps):
        decay = math.exp(-quality * 5 * t / n_steps)
        loss = final_loss + (init_loss - final_loss) * decay
        loss += random.gauss(0, noise)
        curve.append(max(0.01, loss))

    return curve


class SyntheticMLPTrainer(BaseTrainer):
    """
    Simulates training an MLP on a regression task.
    Hyperparameters: learning_rate, hidden_dim, dropout, weight_decay.
    """

    async def train(self, config: TrialConfig) -> AsyncGenerator[StepResult, None]:
        hp = config.hyperparams
        lr          = float(hp.get("learning_rate", 1e-3))
        hidden_dim  = int(hp.get("hidden_dim", 64))
        dropout     = float(hp.get("dropout", 0.1))
        weight_decay = float(hp.get("weight_decay", 1e-4))

        val_curve   = _simulate_learning_curve(lr, hidden_dim, dropout, weight_decay,
                                                config.max_steps)
        train_curve = [v * random.uniform(0.85, 0.95) for v in val_curve]

        for step, (val_loss, train_loss) in enumerate(zip(val_curve, train_curve)):
            # Checkpoint: yield control so Trio can cancel or schedule other tasks
            await trio.sleep(0)

            yield StepResult(
                trial_id=config.trial_id,
                step=step,
                metric=val_loss,
                train_metric=train_loss,
                hyperparams=config.hyperparams,
            )


class SyntheticRLTrainer(BaseTrainer):
    """
    Simulates RL training (e.g., SAC on locomotion).
    Hyperparameters: learning_rate, tau, batch_size, gamma.
    """

    async def train(self, config: TrialConfig) -> AsyncGenerator[StepResult, None]:
        hp = config.hyperparams
        lr         = float(hp.get("learning_rate", 3e-4))
        tau        = float(hp.get("tau", 0.005))
        batch_size = int(hp.get("batch_size", 256))
        gamma      = float(hp.get("gamma", 0.99))

        # Optimal: lr≈3e-4, tau≈0.005, batch≈256, gamma≈0.99
        lr_score    = math.exp(-10 * (math.log10(lr) - math.log10(3e-4)) ** 2)
        tau_score   = math.exp(-((math.log10(tau) - math.log10(0.005)) ** 2))
        batch_score = math.exp(-((math.log2(batch_size) - math.log2(256)) ** 2))
        gam_score   = math.exp(-((gamma - 0.99) / 0.02) ** 2)

        quality = 0.4 * lr_score + 0.2 * tau_score + 0.2 * batch_score + 0.2 * gam_score

        # RL reward: starts low, grows asymptotically (negated for "lower is better")
        max_reward = 4000 + quality * 1500
        for step in range(config.max_steps):
            await trio.sleep(0)
            progress = step / config.max_steps
            reward = max_reward * (1 - math.exp(-quality * 5 * progress))
            reward += random.gauss(0, 50)
            # Return negative reward as "loss" so lower = better convention holds
            yield StepResult(
                trial_id=config.trial_id,
                step=step,
                metric=-reward,            # negated: lower is better
                train_metric=-reward * 0.9,
                hyperparams=config.hyperparams,
            )
