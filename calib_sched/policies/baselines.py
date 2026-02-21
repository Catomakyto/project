from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np

from ..sim.env import ACTION_FULL, ACTION_IDLE, ACTION_PROBE, NUM_ACTIONS


class BasePolicy:
    def select_action(self, context: Mapping[str, float]) -> int:
        raise NotImplementedError

    def action_distribution(self, context: Mapping[str, float]) -> np.ndarray:
        probs = np.zeros(NUM_ACTIONS, dtype=float)
        action = self.select_action(context)
        probs[action] = 1.0
        return probs

    def update(self, context: Mapping[str, float], action: int, reward: float) -> None:
        del context, action, reward


@dataclass
class PeriodicPolicy(BasePolicy):
    period: int
    full_action: int = ACTION_FULL
    idle_action: int = ACTION_IDLE

    def select_action(self, context: Mapping[str, float]) -> int:
        t = int(context.get("timestep", 0.0))
        if self.period <= 0:
            return self.idle_action
        return self.full_action if (t % self.period) == 0 else self.idle_action


@dataclass
class AlwaysIdlePolicy(BasePolicy):
    idle_action: int = ACTION_IDLE

    def select_action(self, context: Mapping[str, float]) -> int:
        del context
        return self.idle_action


@dataclass
class AlwaysProbePolicy(BasePolicy):
    probe_action: int = ACTION_PROBE

    def select_action(self, context: Mapping[str, float]) -> int:
        del context
        return self.probe_action


@dataclass
class ThresholdPolicy(BasePolicy):
    threshold: float
    full_action: int = ACTION_FULL
    idle_action: int = ACTION_IDLE

    def select_action(self, context: Mapping[str, float]) -> int:
        score = (
            float(context.get("readout_mean_error_zero", 0.0))
            + float(context.get("readout_mean_error_one", 0.0))
            + float(context.get("coherent_mean_anomaly", 0.0))
            + float(context.get("crosstalk_mean_odd_parity", 0.0))
        )
        return self.full_action if score >= self.threshold else self.idle_action


@dataclass
class EpsilonGreedyPolicy(BasePolicy):
    base_policy: BasePolicy
    epsilon: float

    def select_action(self, context: Mapping[str, float]) -> int:
        return int(self.base_policy.select_action(context))

    def action_distribution(self, context: Mapping[str, float]) -> np.ndarray:
        eps = min(1.0, max(0.0, float(self.epsilon)))
        probs = np.full(NUM_ACTIONS, eps / NUM_ACTIONS, dtype=float)
        base_action = int(self.base_policy.select_action(context))
        probs[base_action] += 1.0 - eps
        return probs

    def update(self, context: Mapping[str, float], action: int, reward: float) -> None:
        self.base_policy.update(context, action, reward)


def sample_action(
    policy: BasePolicy,
    context: Mapping[str, float],
    rng: np.random.Generator,
) -> Dict[str, object]:
    probs = policy.action_distribution(context)
    action = int(rng.choice(np.arange(NUM_ACTIONS), p=probs))
    return {
        "action": action,
        "propensity": float(probs[action]),
        "probs": probs,
    }
