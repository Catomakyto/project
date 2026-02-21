from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Sequence

import numpy as np

from ..sim.env import ACTION_IDLE, ACTION_PROBE, NUM_ACTIONS
from .baselines import BasePolicy, PeriodicPolicy


@dataclass
class ConservativeBanditPolicy(BasePolicy):
    """Conservative uncertainty-aware contextual bandit with probe-first abstention."""

    ridge: float = 1.0
    beta: float = 1.5
    uncertainty_threshold: float = 1.0
    risk_threshold: float = 0.12
    improvement_margin: float = 0.0
    exploration_epsilon: float = 0.02
    safe_period: int = 24
    feature_keys: Optional[List[str]] = None

    safe_policy: BasePolicy = field(default_factory=lambda: PeriodicPolicy(period=24))
    _A: Optional[List[np.ndarray]] = field(default=None, init=False, repr=False)
    _A_inv: Optional[List[np.ndarray]] = field(default=None, init=False, repr=False)
    _b: Optional[List[np.ndarray]] = field(default=None, init=False, repr=False)
    _theta: Optional[List[np.ndarray]] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.safe_policy = PeriodicPolicy(period=self.safe_period)

    def _infer_feature_keys(self, context: Mapping[str, float]) -> List[str]:
        numeric = [k for k, v in context.items() if isinstance(v, (float, int))]
        preferred = sorted(k for k in numeric if k != "timestep")
        if "timestep" in numeric:
            preferred.append("timestep")
        return preferred

    def _ensure_model(self, context: Mapping[str, float]) -> None:
        if self.feature_keys is None:
            self.feature_keys = self._infer_feature_keys(context)
        if self._A is not None:
            return

        d = 1 + len(self.feature_keys)
        self._A = [self.ridge * np.eye(d, dtype=float) for _ in range(NUM_ACTIONS)]
        self._A_inv = [np.linalg.inv(a) for a in self._A]
        self._b = [np.zeros(d, dtype=float) for _ in range(NUM_ACTIONS)]
        self._theta = [np.zeros(d, dtype=float) for _ in range(NUM_ACTIONS)]

    def _x(self, context: Mapping[str, float]) -> np.ndarray:
        self._ensure_model(context)
        assert self.feature_keys is not None
        values = [float(context.get(k, 0.0)) for k in self.feature_keys]
        return np.asarray([1.0] + values, dtype=float)

    def predict_reward(self, context: Mapping[str, float], action: int) -> float:
        self._ensure_model(context)
        assert self._theta is not None
        x = self._x(context)
        return float(np.dot(self._theta[action], x))

    def predict_uncertainty(self, context: Mapping[str, float], action: int) -> float:
        self._ensure_model(context)
        assert self._A_inv is not None
        x = self._x(context)
        val = float(x.T @ self._A_inv[action] @ x)
        return float(np.sqrt(max(0.0, val)))

    def predict_silent_failure_risk(self, context: Mapping[str, float]) -> float:
        ro = float(context.get("readout_mean_error_zero", 0.0)) + float(
            context.get("readout_mean_error_one", 0.0)
        )
        coh = float(context.get("coherent_mean_anomaly", 0.0))
        xt = float(context.get("crosstalk_mean_odd_parity", 0.0))
        score = 12.0 * ro + 6.0 * coh + 3.0 * xt - 0.08
        return float(min(1.0, max(0.0, score)))

    def _deterministic_action(self, context: Mapping[str, float]) -> int:
        self._ensure_model(context)
        assert self._theta is not None and self._A_inv is not None

        baseline_action = int(self.safe_policy.select_action(context))
        risk = self.predict_silent_failure_risk(context)

        means = np.asarray([self.predict_reward(context, a) for a in range(NUM_ACTIONS)], dtype=float)
        unc = np.asarray([self.predict_uncertainty(context, a) for a in range(NUM_ACTIONS)], dtype=float)

        if risk >= self.risk_threshold or float(np.max(unc)) >= self.uncertainty_threshold:
            return ACTION_PROBE

        lcbs = means - self.beta * unc
        best_action = int(np.argmax(lcbs))
        baseline_lcb = float(lcbs[baseline_action])
        best_lcb = float(lcbs[best_action])

        chosen = best_action if (best_lcb > baseline_lcb + self.improvement_margin) else baseline_action

        if chosen == ACTION_IDLE and risk > 0.5 * self.risk_threshold:
            return ACTION_PROBE
        return chosen

    def select_action(self, context: Mapping[str, float]) -> int:
        return self._deterministic_action(context)

    def action_distribution(self, context: Mapping[str, float]) -> np.ndarray:
        action = self._deterministic_action(context)
        eps = min(1.0, max(0.0, float(self.exploration_epsilon)))
        probs = np.full(NUM_ACTIONS, eps / NUM_ACTIONS, dtype=float)
        probs[action] += 1.0 - eps
        return probs

    def update(self, context: Mapping[str, float], action: int, reward: float) -> None:
        self._ensure_model(context)
        assert self._A is not None and self._A_inv is not None and self._b is not None and self._theta is not None

        x = self._x(context)
        self._A[action] = self._A[action] + np.outer(x, x)
        self._b[action] = self._b[action] + reward * x
        self._A_inv[action] = np.linalg.pinv(self._A[action])
        self._theta[action] = self._A_inv[action] @ self._b[action]

    def fit_batch(
        self,
        contexts: Sequence[Mapping[str, float]],
        actions: Sequence[int],
        rewards: Sequence[float],
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        if not contexts:
            return
        self._ensure_model(contexts[0])
        assert self._A is not None and self._A_inv is not None and self._b is not None and self._theta is not None

        d = self._A[0].shape[0]
        w = np.ones(len(contexts), dtype=float) if weights is None else np.asarray(weights, dtype=float)

        for a in range(NUM_ACTIONS):
            X_rows = []
            y_vals = []
            wt_vals = []
            for i, (ctx, act, rew) in enumerate(zip(contexts, actions, rewards)):
                if int(act) == a:
                    X_rows.append(self._x(ctx))
                    y_vals.append(float(rew))
                    wt_vals.append(float(w[i]))

            A = self.ridge * np.eye(d, dtype=float)
            b = np.zeros(d, dtype=float)
            if X_rows:
                X = np.vstack(X_rows)
                y = np.asarray(y_vals, dtype=float)
                W = np.diag(np.asarray(wt_vals, dtype=float))
                A = A + X.T @ W @ X
                b = b + X.T @ W @ y

            A_inv = np.linalg.pinv(A)
            theta = A_inv @ b

            self._A[a] = A
            self._A_inv[a] = A_inv
            self._b[a] = b
            self._theta[a] = theta
