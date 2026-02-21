from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .hw_drift_fit import fit_and_save
from .hw_grounded_env import (
    ACTION_FULL_RECAL,
    ACTION_NO_ACTION,
    ACTION_PARTIAL_RECAL,
    ACTION_PROBE,
    HWGroundedEnv,
    HWGroundedEnvConfig,
    load_channel_params_from_fit,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

CHANNELS: Tuple[str, str, str, str] = ("z", "o", "c", "x")
POLICY_ORDER: Tuple[str, str, str, str] = (
    "periodic_full",
    "periodic_partial",
    "threshold",
    "conservative_bandit",
)
PLOT_POLICIES: Tuple[str, str, str] = (
    "periodic_full",
    "periodic_partial",
    "conservative_bandit",
)


@dataclass
class TrajectoryOutcome:
    """Single trajectory outcome under one policy."""

    full_count: int
    partial_count: int
    probe_count: int
    cumulative_reward: float
    time_to_failure: int
    unnecessary_recal_rate: float
    total_cost: float
    post_shock_recal_cost: float
    performance_trace: Optional[np.ndarray]
    cumulative_recal_trace: Optional[np.ndarray]


@dataclass
class PolicyEvalResult:
    """Monte Carlo aggregate arrays for one evaluated policy."""

    name: str
    full_counts: np.ndarray
    partial_counts: np.ndarray
    probe_counts: np.ndarray
    cumulative_rewards: np.ndarray
    time_to_failure: np.ndarray
    unnecessary_rates: np.ndarray
    total_costs: np.ndarray
    post_shock_recal_costs: np.ndarray
    performance_traces: Optional[np.ndarray]
    cumulative_recal_traces: Optional[np.ndarray]


class PolicyAdapter:
    """Minimal policy API for the grounded simulator."""

    def select_action(self, context: Mapping[str, float]) -> int:
        raise NotImplementedError

    def update(self, context: Mapping[str, float], action: int, reward: float) -> None:
        del context, action, reward


class PeriodicPolicy(PolicyAdapter):
    """Periodic baseline policy."""

    def __init__(self, period: int, action: int) -> None:
        self.period = int(period)
        self.action = int(action)

    def select_action(self, context: Mapping[str, float]) -> int:
        if self.period <= 0:
            return ACTION_NO_ACTION
        t = int(context.get("timestep", 0.0))
        return self.action if (t % self.period) == 0 else ACTION_NO_ACTION


class ThresholdEscalationPolicy(PolicyAdapter):
    """Threshold baseline with probe-based monitoring."""

    def __init__(
        self,
        partial_threshold: float,
        full_threshold: float,
        probe_interval: int = 6,
        stale_probe_age: int = 8,
    ) -> None:
        self.partial_threshold = float(partial_threshold)
        self.full_threshold = float(full_threshold)
        self.probe_interval = max(1, int(probe_interval))
        self.stale_probe_age = max(1, int(stale_probe_age))

    def select_action(self, context: Mapping[str, float]) -> int:
        t = int(context.get("timestep", 0.0))
        obs_age = int(context.get("obs_age", 0.0))
        probe_remaining = int(context.get("probe_cap_remaining", 0.0))

        channels = _extract_channels(context)
        max_val = float(max(channels.values()))
        score = float(sum(channels.values()))

        if max_val >= self.full_threshold or score >= 2.0 * self.full_threshold:
            return ACTION_FULL_RECAL
        if max_val >= self.partial_threshold or score >= 2.0 * self.partial_threshold:
            return ACTION_PARTIAL_RECAL
        if probe_remaining > 0 and (obs_age >= self.stale_probe_age or (t % self.probe_interval) == 0):
            return ACTION_PROBE
        return ACTION_NO_ACTION


class ConservativeGuardedPolicy(PolicyAdapter):
    """Budget-aware conservative policy with escalation safeguards."""

    def __init__(
        self,
        safe_period: int,
        reference_partial_period: int,
        beta: float,
        danger_k: int,
        failure_threshold: float,
        weights: Mapping[str, float],
        baseline_means: Mapping[str, float],
        baseline_sigmas: Mapping[str, float],
        probe_shots: int,
        lambda_cost: float,
        costs: Mapping[int, float],
        stale_probe_age: int = 40,
        safety_margin: float = 0.0,
    ) -> None:
        self.safe_period = max(1, int(safe_period))
        self.reference_partial_period = max(1, int(reference_partial_period))
        self.beta = float(beta)
        self.danger_k = max(1, int(danger_k))
        self.failure_threshold = float(failure_threshold)
        self.weights = {k: float(v) for k, v in weights.items()}
        self.baseline_means = {k: float(v) for k, v in baseline_means.items()}
        self.baseline_sigmas = {k: max(1e-6, float(v)) for k, v in baseline_sigmas.items()}
        self.probe_shots = max(1, int(probe_shots))
        self.lambda_cost = float(lambda_cost)
        self.costs = {int(k): float(v) for k, v in costs.items()}
        self.stale_probe_age = max(1, int(stale_probe_age))
        self.safety_margin = float(safety_margin)

        self.force_action: Optional[int] = None
        self.danger_probe_streak = 0
        self.danger_mode = False
        self.after_partial_monitoring = False
        self.post_partial_danger_probe_streak = 0

    def select_action(self, context: Mapping[str, float]) -> int:
        channels = _extract_channels(context)
        obs_age = int(context.get("obs_age", 0.0))
        t = int(context.get("timestep", 0.0))
        probe_remaining = int(context.get("probe_cap_remaining", 0.0))
        budget_remaining = float(context.get("budget_remaining", 0.0))

        if self.force_action is not None:
            forced_cost = self.costs.get(self.force_action, 0.0)
            if forced_cost <= budget_remaining + 1e-12:
                return int(self.force_action)

        _, _, perf_lcb_now = self._perf_lcb(channels, obs_age)
        action_reward_lcb = {
            action: self._predicted_reward_lcb(channels, obs_age, action)
            for action in [ACTION_NO_ACTION, ACTION_PROBE, ACTION_PARTIAL_RECAL, ACTION_FULL_RECAL]
        }

        if probe_remaining > 0 and self.costs[ACTION_PROBE] <= budget_remaining + 1e-12:
            if self.after_partial_monitoring:
                return ACTION_PROBE
            if self.danger_mode:
                return ACTION_PROBE

        if (
            probe_remaining > 0
            and self.costs[ACTION_PROBE] <= budget_remaining + 1e-12
            and obs_age >= self.stale_probe_age
        ):
            return ACTION_PROBE

        baseline_action = ACTION_FULL_RECAL if (t % self.safe_period) == 0 else ACTION_NO_ACTION
        baseline_reward_lcb = float(action_reward_lcb[baseline_action])
        candidate_actions: List[int] = []
        for action in [ACTION_NO_ACTION, ACTION_PROBE, ACTION_PARTIAL_RECAL, ACTION_FULL_RECAL]:
            if action == ACTION_PROBE and probe_remaining <= 0:
                continue
            action_cost = self.costs.get(action, 0.0)
            if action_cost <= budget_remaining + 1e-12:
                candidate_actions.append(action)

        if not candidate_actions:
            del perf_lcb_now
            return ACTION_NO_ACTION

        best_action = max(candidate_actions, key=lambda a: float(action_reward_lcb[a]))
        if float(action_reward_lcb[best_action]) >= baseline_reward_lcb - self.safety_margin:
            del perf_lcb_now
            return int(best_action)

        del perf_lcb_now
        return int(baseline_action if baseline_action in candidate_actions else ACTION_NO_ACTION)

    def update(self, context: Mapping[str, float], action: int, reward: float) -> None:
        del reward
        channels = _extract_channels(context)
        obs_age = int(context.get("obs_age", 0.0))
        _, _, perf_lcb = self._perf_lcb(channels, obs_age)

        if action == ACTION_PROBE:
            if perf_lcb < self.failure_threshold:
                self.danger_probe_streak += 1
                self.danger_mode = True
                if self.after_partial_monitoring:
                    self.post_partial_danger_probe_streak += 1
            else:
                self.danger_probe_streak = 0
                self.danger_mode = False
                if self.after_partial_monitoring:
                    self.post_partial_danger_probe_streak = 0
                    self.after_partial_monitoring = False

            if self.after_partial_monitoring and self.post_partial_danger_probe_streak >= self.danger_k:
                self.force_action = ACTION_FULL_RECAL
                self.after_partial_monitoring = False
                self.post_partial_danger_probe_streak = 0
                self.danger_probe_streak = 0
                self.danger_mode = False
            elif self.danger_probe_streak >= self.danger_k and self.force_action is None:
                self.force_action = ACTION_PARTIAL_RECAL
                self.danger_probe_streak = 0
        else:
            self.danger_probe_streak = 0

            if action == ACTION_PARTIAL_RECAL:
                forced_partial = self.force_action == ACTION_PARTIAL_RECAL
                if forced_partial:
                    self.force_action = None
                self.after_partial_monitoring = bool(forced_partial)
                self.post_partial_danger_probe_streak = 0
                self.danger_mode = False
            elif action == ACTION_FULL_RECAL:
                if self.force_action == ACTION_FULL_RECAL:
                    self.force_action = None
                self.after_partial_monitoring = False
                self.post_partial_danger_probe_streak = 0
                self.danger_mode = False

    def _predicted_reward_lcb(self, channels: Mapping[str, float], obs_age: int, action: int) -> float:
        predicted = dict(channels)
        reset_channels: Sequence[str] = ()

        if action == ACTION_PARTIAL_RECAL:
            reset_channels = _partial_targets(predicted, self.weights)
        elif action == ACTION_FULL_RECAL:
            reset_channels = CHANNELS

        for channel in reset_channels:
            predicted[channel] = float(self.baseline_means[channel])

        perf_hat = _performance_from_channels(predicted, self.weights)
        perf_se = _performance_se(
            predicted,
            self.weights,
            probe_shots=self.probe_shots,
            obs_age=obs_age,
            baseline_sigmas=self.baseline_sigmas,
            reset_channels=reset_channels,
        )
        perf_lcb = perf_hat - self.beta * perf_se
        return float(perf_lcb - self.lambda_cost * self.costs.get(action, 0.0))

    def _perf_lcb(self, channels: Mapping[str, float], obs_age: int) -> Tuple[float, float, float]:
        perf_hat = _performance_from_channels(channels, self.weights)
        perf_se = _performance_se(
            channels,
            self.weights,
            probe_shots=self.probe_shots,
            obs_age=obs_age,
            baseline_sigmas=self.baseline_sigmas,
            reset_channels=(),
        )
        perf_lcb = perf_hat - self.beta * perf_se
        return perf_hat, perf_se, perf_lcb


def _extract_channels(context: Mapping[str, float]) -> Dict[str, float]:
    return {
        "z": float(context.get("readout_mean_error_zero", 0.0)),
        "o": float(context.get("readout_mean_error_one", 0.0)),
        "c": float(context.get("coherent_mean_anomaly", 0.0)),
        "x": float(context.get("crosstalk_mean_anomaly", 0.0)),
    }


def _partial_targets(channels: Mapping[str, float], weights: Mapping[str, float]) -> Sequence[str]:
    contributions = {
        channel: float(weights.get(channel, 1.0)) * float(channels[channel])
        for channel in CHANNELS
    }
    worst = max(contributions, key=lambda c: contributions[c])
    if worst in {"z", "o"}:
        return ("z", "o")
    if worst == "c":
        return ("c",)
    return ("x",)


def _performance_from_channels(channels: Mapping[str, float], weights: Mapping[str, float]) -> float:
    penalty = sum(float(weights.get(channel, 1.0)) * float(channels[channel]) for channel in CHANNELS)
    return float(np.clip(1.0 - penalty, 0.0, 1.0))


def _performance_se(
    channels: Mapping[str, float],
    weights: Mapping[str, float],
    probe_shots: int,
    obs_age: int,
    baseline_sigmas: Mapping[str, float],
    reset_channels: Sequence[str],
) -> float:
    shots = max(1, int(probe_shots))
    reset_set = set(reset_channels)
    var = 0.0
    for channel in CHANNELS:
        p = float(np.clip(channels[channel], 1e-6, 1.0 - 1e-6))
        shot_se = float(np.sqrt(p * (1.0 - p) / shots))
        if channel in reset_set:
            channel_se = max(shot_se, float(baseline_sigmas.get(channel, shot_se)))
        else:
            channel_se = shot_se
        w = float(weights.get(channel, 1.0))
        var += (w * channel_se) ** 2

    inflation = 1.0 + 0.05 * max(0, int(obs_age))
    return float(np.sqrt(var) * inflation)


def _mean_ci95(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    mean = float(np.mean(arr))
    if arr.size == 1:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean}
    se = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    delta = 1.96 * se
    return {"mean": mean, "ci95_low": mean - delta, "ci95_high": mean + delta}


def _period_for_budget(action_cost: float, steps: int, budget: float) -> int:
    if action_cost <= 0.0:
        return 0
    best_period = 0
    best_total = 0.0
    for period in range(1, steps + 1):
        count = 1 + (steps - 1) // period
        total_cost = float(count * action_cost)
        if total_cost <= budget and total_cost >= best_total:
            best_total = total_cost
            best_period = period
    return best_period


def _fallback_order(action: int) -> Sequence[int]:
    if action == ACTION_FULL_RECAL:
        return (ACTION_FULL_RECAL, ACTION_PARTIAL_RECAL, ACTION_NO_ACTION, ACTION_PROBE)
    if action == ACTION_PARTIAL_RECAL:
        return (ACTION_PARTIAL_RECAL, ACTION_NO_ACTION, ACTION_PROBE)
    if action == ACTION_PROBE:
        return (ACTION_PROBE, ACTION_NO_ACTION)
    return (ACTION_NO_ACTION,)


def _enforce_budget(action: int, budget_remaining: float, env: HWGroundedEnv) -> int:
    """Enforce budget fairly with fallback to cheaper valid actions."""
    for candidate in _fallback_order(action):
        if candidate == ACTION_PROBE and not env.can_probe():
            continue
        if env.action_cost(candidate) <= budget_remaining + 1e-12:
            return int(candidate)
    return ACTION_NO_ACTION


def _run_one_trajectory(
    policy: PolicyAdapter,
    env_cfg: HWGroundedEnvConfig,
    steps: int,
    budget: float,
    seed: int,
    lookahead_steps: int,
    collect_traces: bool,
    shock_start: Optional[int],
    shock_window: int,
) -> TrajectoryOutcome:
    env = HWGroundedEnv(env_cfg, seed=seed)
    budget_remaining = float(budget)

    perf_trace = np.zeros(steps, dtype=float) if collect_traces else None
    recal_trace = np.zeros(steps, dtype=float) if collect_traces else None

    cumulative_reward = 0.0
    total_cost = 0.0
    full_count = 0
    partial_count = 0
    probe_count = 0
    unnecessary_recal = 0
    total_recal = 0
    time_to_failure = steps + 1
    cumulative_recal = 0
    post_shock_recal_cost = 0.0

    for t in range(steps):
        context = env.get_context(budget_remaining=budget_remaining)
        proposed_action = int(policy.select_action(context))
        action = _enforce_budget(proposed_action, budget_remaining=budget_remaining, env=env)

        counterfactual_fail = False
        if action in {ACTION_PARTIAL_RECAL, ACTION_FULL_RECAL}:
            snapshot = env.snapshot()
            counterfactual_fail = env.counterfactual_would_fail_without_recal(
                snapshot=snapshot,
                lookahead_steps=lookahead_steps,
            )

        step_result = env.step(action)
        executed_action = int(step_result.action)
        budget_remaining -= step_result.cost
        total_cost += step_result.cost
        cumulative_reward += step_result.reward

        if executed_action == ACTION_FULL_RECAL:
            full_count += 1
            total_recal += 1
            cumulative_recal += 1
            if not counterfactual_fail:
                unnecessary_recal += 1
        elif executed_action == ACTION_PARTIAL_RECAL:
            partial_count += 1
            total_recal += 1
            cumulative_recal += 1
            if not counterfactual_fail:
                unnecessary_recal += 1
        elif executed_action == ACTION_PROBE:
            probe_count += 1

        if (
            shock_start is not None
            and int(shock_start) <= t < int(shock_start) + int(shock_window)
            and executed_action in {ACTION_PARTIAL_RECAL, ACTION_FULL_RECAL}
        ):
            post_shock_recal_cost += float(step_result.cost)

        policy.update(context, executed_action, step_result.reward)

        if perf_trace is not None:
            perf_trace[t] = step_result.performance
        if recal_trace is not None:
            recal_trace[t] = cumulative_recal

        if step_result.done and time_to_failure == steps + 1:
            time_to_failure = int(step_result.timestep + 1)

    unnecessary_rate = float(unnecessary_recal / total_recal) if total_recal > 0 else 0.0

    return TrajectoryOutcome(
        full_count=full_count,
        partial_count=partial_count,
        probe_count=probe_count,
        cumulative_reward=float(cumulative_reward),
        time_to_failure=int(time_to_failure),
        unnecessary_recal_rate=float(unnecessary_rate),
        total_cost=float(total_cost),
        post_shock_recal_cost=float(post_shock_recal_cost),
        performance_trace=perf_trace,
        cumulative_recal_trace=recal_trace,
    )


def _evaluate_policy(
    name: str,
    policy_factory: Callable[[], PolicyAdapter],
    env_cfg: HWGroundedEnvConfig,
    steps: int,
    seeds: np.ndarray,
    budget: float,
    lookahead_steps: int,
    collect_traces: bool,
    verbose: bool,
    shock_start: Optional[int],
    shock_window: int,
) -> PolicyEvalResult:
    mc = int(seeds.size)
    full_counts = np.zeros(mc, dtype=float)
    partial_counts = np.zeros(mc, dtype=float)
    probe_counts = np.zeros(mc, dtype=float)
    cumulative_rewards = np.zeros(mc, dtype=float)
    time_to_failure = np.zeros(mc, dtype=float)
    unnecessary_rates = np.zeros(mc, dtype=float)
    total_costs = np.zeros(mc, dtype=float)
    post_shock_recal_costs = np.zeros(mc, dtype=float)

    performance_traces = np.zeros((mc, steps), dtype=float) if collect_traces else None
    cumulative_recal_traces = np.zeros((mc, steps), dtype=float) if collect_traces else None

    if verbose:
        print(f"[eval] Running policy={name} over {mc} trajectories ...")
    checkpoint = max(1, mc // 5)
    for i, seed in enumerate(seeds):
        policy = policy_factory()
        outcome = _run_one_trajectory(
            policy=policy,
            env_cfg=env_cfg,
            steps=steps,
            budget=budget,
            seed=int(seed),
            lookahead_steps=lookahead_steps,
            collect_traces=collect_traces,
            shock_start=shock_start,
            shock_window=shock_window,
        )

        full_counts[i] = outcome.full_count
        partial_counts[i] = outcome.partial_count
        probe_counts[i] = outcome.probe_count
        cumulative_rewards[i] = outcome.cumulative_reward
        time_to_failure[i] = outcome.time_to_failure
        unnecessary_rates[i] = outcome.unnecessary_recal_rate
        total_costs[i] = outcome.total_cost
        post_shock_recal_costs[i] = outcome.post_shock_recal_cost

        if collect_traces:
            assert outcome.performance_trace is not None and outcome.cumulative_recal_trace is not None
            performance_traces[i, :] = outcome.performance_trace
            cumulative_recal_traces[i, :] = outcome.cumulative_recal_trace

        if verbose and ((i + 1) % checkpoint == 0 or (i + 1) == mc):
            print(f"[eval] policy={name} progress {i + 1}/{mc}")

    return PolicyEvalResult(
        name=name,
        full_counts=full_counts,
        partial_counts=partial_counts,
        probe_counts=probe_counts,
        cumulative_rewards=cumulative_rewards,
        time_to_failure=time_to_failure,
        unnecessary_rates=unnecessary_rates,
        total_costs=total_costs,
        post_shock_recal_costs=post_shock_recal_costs,
        performance_traces=performance_traces,
        cumulative_recal_traces=cumulative_recal_traces,
    )


def _policy_factories(
    env_cfg: HWGroundedEnvConfig,
    steps: int,
    budget: float,
    threshold_partial: float,
    threshold_full: float,
    beta: float,
    danger_k: int,
) -> Tuple[Dict[str, Callable[[], PolicyAdapter]], Dict[str, object]]:
    full_period = _period_for_budget(env_cfg.full_cost, steps=steps, budget=budget)
    partial_period = _period_for_budget(env_cfg.partial_cost, steps=steps, budget=budget)

    baseline_means = {ch: float(env_cfg.channels[ch].initial_mean) for ch in CHANNELS}
    baseline_sigmas = {ch: float(env_cfg.channels[ch].initial_sigma) for ch in CHANNELS}
    costs = {
        ACTION_NO_ACTION: 0.0,
        ACTION_PROBE: float(env_cfg.probe_cost),
        ACTION_PARTIAL_RECAL: float(env_cfg.partial_cost),
        ACTION_FULL_RECAL: float(env_cfg.full_cost),
    }

    safe_period = max(1, full_period if full_period > 0 else max(1, steps // 10))

    factories: Dict[str, Callable[[], PolicyAdapter]] = {
        "periodic_full": lambda: PeriodicPolicy(period=full_period, action=ACTION_FULL_RECAL),
        "periodic_partial": lambda: PeriodicPolicy(period=partial_period, action=ACTION_PARTIAL_RECAL),
        "threshold": lambda: ThresholdEscalationPolicy(
            partial_threshold=threshold_partial,
            full_threshold=threshold_full,
            probe_interval=6,
            stale_probe_age=8,
        ),
        "conservative_bandit": lambda: ConservativeGuardedPolicy(
            safe_period=safe_period,
            reference_partial_period=max(1, partial_period if partial_period > 0 else safe_period),
            beta=beta,
            danger_k=danger_k,
            failure_threshold=float(env_cfg.failure_threshold),
            weights=env_cfg.channel_weights,
            baseline_means=baseline_means,
            baseline_sigmas=baseline_sigmas,
            probe_shots=int(env_cfg.probe_shots),
            lambda_cost=float(env_cfg.lambda_cost),
            costs=costs,
            stale_probe_age=50,
            safety_margin=0.0,
        ),
    }

    periods: Dict[str, object] = {
        "periodic_full_period": int(full_period),
        "periodic_partial_period": int(partial_period),
        "conservative_policy_impl": "conservative_guarded_lcb",
        "conservative_beta": float(beta),
        "danger_k": int(danger_k),
        "max_probes_per_horizon": int(env_cfg.max_probes_per_horizon),
    }
    return factories, periods


def _sample_policy_seeds(seed: int, mc: int, seed_offset: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed + seed_offset))
    return rng.integers(low=1, high=2**31 - 1, size=int(mc), dtype=np.int64)


def _shock_metrics(
    time_to_failure: np.ndarray,
    post_shock_recal_costs: np.ndarray,
    steps: int,
    shock_start: Optional[int],
    shock_window: int = 20,
) -> Dict[str, float]:
    if shock_start is None:
        return {}
    ttf = np.asarray(time_to_failure, dtype=float)
    post_shock_ttf = np.maximum(ttf - float(shock_start), 0.0)

    fail_after_shock = (ttf <= float(steps)) & (ttf > float(shock_start))
    collapse_mask = fail_after_shock & ((ttf - float(shock_start)) <= float(shock_window))
    collapse_rate = float(np.mean(collapse_mask.astype(float))) if ttf.size > 0 else 0.0
    post_shock_recal_cost = float(np.mean(np.asarray(post_shock_recal_costs, dtype=float)))

    return {
        "post_shock_ttf_median": float(np.median(post_shock_ttf)),
        "collapse_rate_20": collapse_rate,
        "recal_cost_spent_post_shock": post_shock_recal_cost,
    }


def _summarize_results(
    results: Mapping[str, PolicyEvalResult],
    steps: int,
    shock_start: Optional[int],
    shock_window: int = 20,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    summary: Dict[str, object] = {}
    rows: List[Dict[str, object]] = []

    for name, result in results.items():
        ttf = np.asarray(result.time_to_failure, dtype=float)
        ttf_median = float(np.median(ttf))
        ttf_q1 = float(np.percentile(ttf, 25))
        ttf_q3 = float(np.percentile(ttf, 75))
        ttf_iqr = float(ttf_q3 - ttf_q1)

        cost_ci = _mean_ci95(result.total_costs)
        probes_ci = _mean_ci95(result.probe_counts)
        partial_ci = _mean_ci95(result.partial_counts)
        full_ci = _mean_ci95(result.full_counts)
        reward_ci = _mean_ci95(result.cumulative_rewards)
        unnecessary_ci = _mean_ci95(result.unnecessary_rates)
        shock_metrics = _shock_metrics(
            ttf,
            post_shock_recal_costs=result.post_shock_recal_costs,
            steps=steps,
            shock_start=shock_start,
            shock_window=shock_window,
        )

        policy_summary: Dict[str, object] = {
            "total_cost_spent": cost_ci,
            "probes": probes_ci,
            "partial_recal": partial_ci,
            "full_recal": full_ci,
            "cumulative_reward": reward_ci,
            "unnecessary_recalibration_rate": unnecessary_ci,
            "time_to_failure": {
                "median": ttf_median,
                "iqr_q1": ttf_q1,
                "iqr_q3": ttf_q3,
                "iqr": ttf_iqr,
            },
        }
        policy_summary.update(shock_metrics)
        summary[name] = policy_summary

        row: Dict[str, object] = {
            "policy": name,
            "total_cost_spent": cost_ci["mean"],
            "total_cost_spent_ci95_low": cost_ci["ci95_low"],
            "total_cost_spent_ci95_high": cost_ci["ci95_high"],
            "probes": probes_ci["mean"],
            "probes_ci95_low": probes_ci["ci95_low"],
            "probes_ci95_high": probes_ci["ci95_high"],
            "partial_recal": partial_ci["mean"],
            "partial_recal_ci95_low": partial_ci["ci95_low"],
            "partial_recal_ci95_high": partial_ci["ci95_high"],
            "full_recal": full_ci["mean"],
            "full_recal_ci95_low": full_ci["ci95_low"],
            "full_recal_ci95_high": full_ci["ci95_high"],
            "cumulative_reward_mean": reward_ci["mean"],
            "cumulative_reward_ci95_low": reward_ci["ci95_low"],
            "cumulative_reward_ci95_high": reward_ci["ci95_high"],
            "time_to_failure_median": ttf_median,
            "time_to_failure_IQR": ttf_iqr,
            "time_to_failure_iqr_q1": ttf_q1,
            "time_to_failure_iqr_q3": ttf_q3,
            "unnecessary_recalibration_rate": unnecessary_ci["mean"],
            "unnecessary_recalibration_rate_ci95_low": unnecessary_ci["ci95_low"],
            "unnecessary_recalibration_rate_ci95_high": unnecessary_ci["ci95_high"],
        }
        row.update(shock_metrics)
        rows.append(row)

    return summary, pd.DataFrame(rows)


def _result_row(
    policy: str,
    budget: float,
    beta: float,
    result: PolicyEvalResult,
    steps: int,
    shock_start: Optional[int],
    shock_window: int = 20,
) -> Dict[str, object]:
    cost_ci = _mean_ci95(result.total_costs)
    probe_ci = _mean_ci95(result.probe_counts)
    partial_ci = _mean_ci95(result.partial_counts)
    full_ci = _mean_ci95(result.full_counts)
    reward_ci = _mean_ci95(result.cumulative_rewards)
    unnecessary_ci = _mean_ci95(result.unnecessary_rates)

    ttf = np.asarray(result.time_to_failure, dtype=float)
    ttf_median = float(np.median(ttf))
    ttf_q1 = float(np.percentile(ttf, 25))
    ttf_q3 = float(np.percentile(ttf, 75))

    row: Dict[str, object] = {
        "policy": str(policy),
        "budget": float(budget),
        "beta": float(beta),
        "total_cost_spent": float(cost_ci["mean"]),
        "total_cost_spent_ci95_low": float(cost_ci["ci95_low"]),
        "total_cost_spent_ci95_high": float(cost_ci["ci95_high"]),
        "probes": float(probe_ci["mean"]),
        "probes_ci95_low": float(probe_ci["ci95_low"]),
        "probes_ci95_high": float(probe_ci["ci95_high"]),
        "partial_recal": float(partial_ci["mean"]),
        "partial_recal_ci95_low": float(partial_ci["ci95_low"]),
        "partial_recal_ci95_high": float(partial_ci["ci95_high"]),
        "full_recal": float(full_ci["mean"]),
        "full_recal_ci95_low": float(full_ci["ci95_low"]),
        "full_recal_ci95_high": float(full_ci["ci95_high"]),
        "cumulative_reward_mean": float(reward_ci["mean"]),
        "cumulative_reward_ci95_low": float(reward_ci["ci95_low"]),
        "cumulative_reward_ci95_high": float(reward_ci["ci95_high"]),
        "time_to_failure_median": ttf_median,
        "time_to_failure_IQR": float(ttf_q3 - ttf_q1),
        "time_to_failure_iqr_q1": ttf_q1,
        "time_to_failure_iqr_q3": ttf_q3,
        "unnecessary_recalibration_rate": float(unnecessary_ci["mean"]),
        "unnecessary_recalibration_rate_ci95_low": float(unnecessary_ci["ci95_low"]),
        "unnecessary_recalibration_rate_ci95_high": float(unnecessary_ci["ci95_high"]),
    }
    row.update(
        _shock_metrics(
            ttf,
            post_shock_recal_costs=result.post_shock_recal_costs,
            steps=steps,
            shock_start=shock_start,
            shock_window=shock_window,
        )
    )
    return row


def _save_eval_traces(
    out_path: Path,
    results: Mapping[str, PolicyEvalResult],
    steps: int,
    shock_start: Optional[int],
) -> None:
    event_step = int(-1 if shock_start is None else shock_start)
    arrays: Dict[str, np.ndarray] = {
        "steps": np.asarray([int(steps)], dtype=np.int64),
        "shock_start": np.asarray([event_step], dtype=np.int64),
        "switch_step": np.asarray([event_step], dtype=np.int64),
        "policy_names": np.asarray(list(results.keys()), dtype="<U64"),
    }

    for name, result in results.items():
        if result.performance_traces is not None:
            arrays[f"{name}__performance_traces"] = np.asarray(result.performance_traces, dtype=float)
        if result.cumulative_recal_traces is not None:
            arrays[f"{name}__cumulative_recal_traces"] = np.asarray(
                result.cumulative_recal_traces,
                dtype=float,
            )
        arrays[f"{name}__time_to_failure"] = np.asarray(result.time_to_failure, dtype=float)
        arrays[f"{name}__cumulative_rewards"] = np.asarray(result.cumulative_rewards, dtype=float)

    np.savez(out_path, **arrays)


def _save_shock_traces(
    out_path: Path,
    results: Mapping[str, PolicyEvalResult],
    steps: int,
    shock_start: int,
    shock_duration: int,
) -> None:
    arrays: Dict[str, np.ndarray] = {
        "steps": np.asarray([int(steps)], dtype=np.int64),
        "shock_start": np.asarray([int(shock_start)], dtype=np.int64),
        "shock_duration": np.asarray([int(shock_duration)], dtype=np.int64),
        "policy_names": np.asarray(list(results.keys()), dtype="<U64"),
    }

    for name, result in results.items():
        if result.performance_traces is not None:
            traces = np.asarray(result.performance_traces, dtype=float)
            arrays[f"{name}__performance_mean"] = np.mean(traces, axis=0)
            if traces.shape[0] > 1:
                arrays[f"{name}__performance_se"] = np.std(traces, axis=0, ddof=1) / np.sqrt(traces.shape[0])
            else:
                arrays[f"{name}__performance_se"] = np.zeros(traces.shape[1], dtype=float)
            arrays[f"{name}__performance_traces"] = traces
        arrays[f"{name}__time_to_failure"] = np.asarray(result.time_to_failure, dtype=float)
        arrays[f"{name}__full_counts"] = np.asarray(result.full_counts, dtype=float)
        arrays[f"{name}__partial_counts"] = np.asarray(result.partial_counts, dtype=float)
        arrays[f"{name}__probe_counts"] = np.asarray(result.probe_counts, dtype=float)
        arrays[f"{name}__post_shock_recal_costs"] = np.asarray(result.post_shock_recal_costs, dtype=float)

    np.savez(out_path, **arrays)


def _print_robustness_check(shock_df: pd.DataFrame) -> None:
    def _row(policy: str) -> Optional[pd.Series]:
        match = shock_df[shock_df["policy"] == policy]
        return None if match.empty else match.iloc[0]

    cons = _row("conservative_bandit")
    pfull = _row("periodic_full")
    ppart = _row("periodic_partial")
    if cons is None or pfull is None or ppart is None:
        print("ROBUSTNESS CHECK: insufficient policy rows for comparison.")
        return

    cons_collapse = float(cons["collapse_rate_20"])
    cons_post_ttf = float(cons["post_shock_ttf_median"])
    full_collapse = float(pfull["collapse_rate_20"])
    full_post_ttf = float(pfull["post_shock_ttf_median"])
    part_collapse = float(ppart["collapse_rate_20"])
    part_post_ttf = float(ppart["post_shock_ttf_median"])

    print(
        "ROBUSTNESS CHECK: "
        f"cons(collapse={cons_collapse:.4f}, post_shock_ttf={cons_post_ttf:.2f}) vs "
        f"periodic_full(collapse={full_collapse:.4f}, post_shock_ttf={full_post_ttf:.2f}) and "
        f"periodic_partial(collapse={part_collapse:.4f}, post_shock_ttf={part_post_ttf:.2f})"
    )

    strictly_better_vs_full = (cons_collapse < full_collapse) and (cons_post_ttf > full_post_ttf)
    if not strictly_better_vs_full:
        print("FAIL: shock did not differentiate.")

    competitive_vs_partial = (cons_collapse <= part_collapse) or (cons_post_ttf >= part_post_ttf)
    if not competitive_vs_partial:
        print("NOTE: periodic_partial still best; try profile readout_spike_short or increase shock_duration to 20.")


def _plot_performance_vs_time(
    results: Mapping[str, PolicyEvalResult],
    out_path: Path,
    policies: Sequence[str],
    switch_step: Optional[int] = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for name in policies:
        result = results.get(name)
        if result is None or result.performance_traces is None:
            continue
        traces = np.asarray(result.performance_traces, dtype=float)
        mean = np.mean(traces, axis=0)
        if traces.shape[0] > 1:
            se = np.std(traces, axis=0, ddof=1) / np.sqrt(traces.shape[0])
        else:
            se = np.zeros_like(mean)
        ci = 1.96 * se
        x = np.arange(traces.shape[1])
        ax.plot(x, mean, linewidth=2.1, label=name)
        ax.fill_between(x, mean - ci, mean + ci, alpha=0.16)

    if switch_step is not None:
        ax.axvline(float(switch_step), linestyle="--", linewidth=1.5, color="black", alpha=0.7)

    ax.set_title("Performance vs Time (Mean +/- 95% CI)", fontsize=13)
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Performance", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_cumulative_recalibrations(
    results: Mapping[str, PolicyEvalResult],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for name, result in results.items():
        if result.cumulative_recal_traces is None:
            continue
        traces = result.cumulative_recal_traces
        mean = np.mean(traces, axis=0)
        if traces.shape[0] > 1:
            se = np.std(traces, axis=0, ddof=1) / np.sqrt(traces.shape[0])
        else:
            se = np.zeros_like(mean)
        ci = 1.96 * se
        x = np.arange(traces.shape[1])
        ax.plot(x, mean, linewidth=2.1, label=name)
        ax.fill_between(x, np.maximum(0.0, mean - ci), mean + ci, alpha=0.16)

    ax.set_title("Cumulative Recalibrations vs Time", fontsize=13)
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Cumulative Recalibrations", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_time_to_failure_boxplot(
    results: Mapping[str, PolicyEvalResult],
    steps: int,
    out_path: Path,
    policies: Sequence[str],
) -> None:
    import matplotlib.pyplot as plt

    labels = [p for p in policies if p in results]
    data = [np.asarray(results[name].time_to_failure, dtype=float) for name in labels]
    if not data:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylim(0, steps + 2)
    ax.set_title("Time-to-Failure Distribution by Policy", fontsize=13)
    ax.set_xlabel("Policy", fontsize=11)
    ax.set_ylabel("Time to Failure (steps)", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_budget_vs_performance(
    sweep_df: pd.DataFrame,
    conservative_best_df: pd.DataFrame,
    dominance_df: pd.DataFrame,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    if sweep_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    for policy in ["periodic_full", "periodic_partial"]:
        rows = sweep_df[sweep_df["policy"] == policy]
        if rows.empty:
            continue
        base_beta = float(rows["beta"].min())
        rows = rows[rows["beta"] == base_beta].sort_values("budget")
        budgets = rows["budget"].to_numpy(dtype=float)
        means = rows["cumulative_reward_mean"].to_numpy(dtype=float)
        lows = rows["cumulative_reward_ci95_low"].to_numpy(dtype=float)
        highs = rows["cumulative_reward_ci95_high"].to_numpy(dtype=float)
        ax.plot(budgets, means, marker="o", linewidth=2.0, label=policy)
        ax.fill_between(budgets, lows, highs, alpha=0.16)

    if not conservative_best_df.empty:
        best_rows = conservative_best_df.sort_values("budget")
        budgets = best_rows["budget"].to_numpy(dtype=float)
        means = best_rows["cumulative_reward_mean"].to_numpy(dtype=float)
        lows = best_rows["cumulative_reward_ci95_low"].to_numpy(dtype=float)
        highs = best_rows["cumulative_reward_ci95_high"].to_numpy(dtype=float)
        ax.plot(budgets, means, marker="s", linewidth=2.2, label="conservative_best")
        ax.fill_between(budgets, lows, highs, alpha=0.16)

    if not dominance_df.empty and "budget" in dominance_df.columns:
        for budget in sorted({float(v) for v in dominance_df["budget"].to_list()}):
            ax.axvline(budget, color="gray", alpha=0.18, linewidth=1.2)

    ax.set_title("Budget vs Cumulative Reward", fontsize=13)
    ax.set_xlabel("Budget", fontsize=11)
    ax.set_ylabel("Cumulative Reward Mean", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _parse_float_grid(value: str, arg_name: str) -> List[float]:
    items = [part.strip() for part in str(value).split(",") if part.strip()]
    if not items:
        raise ValueError(f"{arg_name} produced an empty list.")
    parsed: List[float] = []
    for item in items:
        parsed.append(float(item))
    return parsed


def _build_env_config(
    fitted_params: Mapping[str, object],
    seed: int,
    probe_shots: int,
    probe_cost: float,
    partial_cost: float,
    full_cost: float,
    lambda_cost: float,
    failure_threshold: float,
    failure_consecutive: int,
    max_probes_per_horizon: int,
    weights: Mapping[str, float],
) -> HWGroundedEnvConfig:
    channels = load_channel_params_from_fit(fitted_params)
    return HWGroundedEnvConfig(
        channels=channels,
        seed=seed,
        probe_shots=probe_shots,
        probe_cost=probe_cost,
        partial_cost=partial_cost,
        full_cost=full_cost,
        lambda_cost=lambda_cost,
        channel_weights=weights,
        failure_threshold=failure_threshold,
        failure_consecutive=failure_consecutive,
        max_probes_per_horizon=max_probes_per_horizon,
        partial_strategy="worst_group",
    )


def _validate_shock_profile(profile: str) -> str:
    profile_name = str(profile).strip().lower()
    if profile_name not in {"coherent_burst_short", "readout_spike_short"}:
        raise ValueError(
            "Unsupported --shock-profile. Supported: coherent_burst_short, readout_spike_short"
        )
    return profile_name


def _run_policy_suite(
    env_cfg: HWGroundedEnvConfig,
    steps: int,
    mc: int,
    budget: float,
    seed: int,
    lookahead: int,
    threshold_partial: float,
    threshold_full: float,
    beta: float,
    danger_k: int,
    collect_traces: bool,
    verbose: bool,
    seed_offset: int,
    shock_start: Optional[int],
    shock_window: int,
) -> Tuple[Dict[str, PolicyEvalResult], Dict[str, object]]:
    factories, periods_info = _policy_factories(
        env_cfg=env_cfg,
        steps=steps,
        budget=budget,
        threshold_partial=threshold_partial,
        threshold_full=threshold_full,
        beta=beta,
        danger_k=danger_k,
    )

    policy_seeds = _sample_policy_seeds(seed=seed, mc=mc, seed_offset=seed_offset)

    results: Dict[str, PolicyEvalResult] = {}
    for name in POLICY_ORDER:
        results[name] = _evaluate_policy(
            name=name,
            policy_factory=factories[name],
            env_cfg=env_cfg,
            steps=steps,
            seeds=policy_seeds,
            budget=budget,
            lookahead_steps=lookahead,
            collect_traces=collect_traces,
            verbose=verbose,
            shock_start=shock_start,
            shock_window=shock_window,
        )
    return results, periods_info


def _build_summary_payload(
    args: argparse.Namespace,
    outdir: Path,
    periods_info: Mapping[str, object],
    summary_body: Mapping[str, object],
    shock_mode: bool,
    shock_start: Optional[int],
    shock_duration: Optional[int],
    shock_profile: Optional[str],
) -> Dict[str, object]:
    config: Dict[str, object] = {
        "csv": str(args.csv),
        "outdir": str(outdir),
        "steps": int(args.steps),
        "mc": int(args.mc),
        "budget": float(args.budget),
        "seed": int(args.seed),
        "lookahead": int(args.lookahead),
        "probe_shots": int(args.probe_shots),
        "probe_cost": float(args.probe_cost),
        "partial_cost": float(args.partial_cost),
        "full_cost": float(args.full_cost),
        "lambda_cost": float(args.lambda_cost),
        "failure_threshold": float(args.failure_threshold),
        "failure_consecutive": int(args.failure_consecutive),
        "max_probes": int(args.max_probes),
        "beta": float(args.beta),
        "danger_k": int(args.danger_k),
        "threshold_partial": float(args.threshold_partial),
        "threshold_full": float(args.threshold_full),
    }
    if shock_mode:
        config["shock"] = True
        config["shock_start"] = int(shock_start if shock_start is not None else -1)
        config["shock_duration"] = int(shock_duration if shock_duration is not None else -1)
        config["shock_profile"] = str(shock_profile) if shock_profile is not None else ""

    return {
        "config": config,
        "budget_matching": dict(periods_info),
        "results": dict(summary_body),
    }


def _run_budget_beta_sweep(
    env_cfg: HWGroundedEnvConfig,
    steps: int,
    mc: int,
    seed: int,
    budgets: Sequence[float],
    betas: Sequence[float],
    threshold_partial: float,
    threshold_full: float,
    danger_k: int,
    lookahead: int,
    shock_start: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []

    for bidx, budget in enumerate(budgets):
        print(f"[sweep] Budget={float(budget):.1f}")
        for beta_idx, beta in enumerate(betas):
            combo_seed_offset = 3_000_000 + bidx * 100_000 + beta_idx * 10_000
            combo_seeds = _sample_policy_seeds(seed=seed, mc=mc, seed_offset=combo_seed_offset)

            factories, _ = _policy_factories(
                env_cfg=env_cfg,
                steps=steps,
                budget=float(budget),
                threshold_partial=threshold_partial,
                threshold_full=threshold_full,
                beta=float(beta),
                danger_k=danger_k,
            )

            combo_rows: Dict[str, Dict[str, object]] = {}
            for policy_name in POLICY_ORDER:
                result = _evaluate_policy(
                    name=policy_name,
                    policy_factory=factories[policy_name],
                    env_cfg=env_cfg,
                    steps=steps,
                    seeds=combo_seeds,
                    budget=float(budget),
                    lookahead_steps=lookahead,
                    collect_traces=False,
                    verbose=False,
                    shock_start=shock_start,
                    shock_window=20,
                )
                row = _result_row(
                    policy=policy_name,
                    budget=float(budget),
                    beta=float(beta),
                    result=result,
                    steps=steps,
                    shock_start=shock_start,
                )
                rows.append(row)
                combo_rows[policy_name] = row

    sweep_df = pd.DataFrame(rows)

    if sweep_df.empty:
        empty_dom_cols = [
            "policy",
            "budget",
            "beta",
            "periodic_full_reward_mean",
            "periodic_partial_reward_mean",
            "periodic_full_ttf_median",
            "periodic_partial_ttf_median",
            "dominates",
        ]
        return sweep_df, pd.DataFrame(columns=sweep_df.columns), pd.DataFrame(columns=empty_dom_cols)

    conservative_rows = sweep_df[sweep_df["policy"] == "conservative_bandit"].copy()
    conservative_best_rows: List[Dict[str, object]] = []
    sweep_cols = list(sweep_df.columns)
    for budget in sorted({float(v) for v in conservative_rows["budget"].to_list()}):
        g = conservative_rows[conservative_rows["budget"] == budget].copy()
        if g.empty:
            continue
        best = g.sort_values(
            ["cumulative_reward_mean", "time_to_failure_median"],
            ascending=[False, False],
        ).iloc[0]
        conservative_best_rows.append(dict(best))
    if conservative_best_rows:
        conservative_best_df = (
            pd.DataFrame(conservative_best_rows, columns=sweep_cols)
            .sort_values("budget")
            .reset_index(drop=True)
        )
    else:
        conservative_best_df = pd.DataFrame(columns=sweep_cols)

    dominance_rows: List[Dict[str, object]] = []
    for _, best in conservative_best_df.iterrows():
        budget = float(best["budget"])
        beta = float(best["beta"])

        periodic_at_combo = sweep_df[
            (sweep_df["budget"] == budget)
            & (sweep_df["beta"] == beta)
            & (sweep_df["policy"].isin(["periodic_full", "periodic_partial"]))
        ]
        if len(periodic_at_combo) < 2:
            periodic_at_combo = sweep_df[
                (sweep_df["budget"] == budget)
                & (sweep_df["policy"].isin(["periodic_full", "periodic_partial"]))
            ].sort_values("beta")

        full_row = periodic_at_combo[periodic_at_combo["policy"] == "periodic_full"].iloc[0]
        partial_row = periodic_at_combo[periodic_at_combo["policy"] == "periodic_partial"].iloc[0]

        max_periodic_reward = max(
            float(full_row["cumulative_reward_mean"]),
            float(partial_row["cumulative_reward_mean"]),
        )
        max_periodic_ci_high = max(
            float(full_row["cumulative_reward_ci95_high"]),
            float(partial_row["cumulative_reward_ci95_high"]),
        )
        max_periodic_ttf = max(
            float(full_row["time_to_failure_median"]),
            float(partial_row["time_to_failure_median"]),
        )

        reward_dominates = (
            float(best["cumulative_reward_mean"]) >= max_periodic_reward + 0.002
        ) or (
            float(best["cumulative_reward_ci95_low"]) > max_periodic_ci_high
        )
        ttf_ok = float(best["time_to_failure_median"]) >= (max_periodic_ttf - 2.0)
        dominates = bool(reward_dominates and ttf_ok)
        if dominates:
            dom_row = dict(best)
            dom_row["periodic_full_reward_mean"] = float(full_row["cumulative_reward_mean"])
            dom_row["periodic_partial_reward_mean"] = float(partial_row["cumulative_reward_mean"])
            dom_row["periodic_full_ttf_median"] = float(full_row["time_to_failure_median"])
            dom_row["periodic_partial_ttf_median"] = float(partial_row["time_to_failure_median"])
            dom_row["dominates"] = True
            dominance_rows.append(dom_row)

    dominance_cols = sweep_cols + [
        "periodic_full_reward_mean",
        "periodic_partial_reward_mean",
        "periodic_full_ttf_median",
        "periodic_partial_ttf_median",
        "dominates",
    ]
    if dominance_rows:
        dominance_df = pd.DataFrame(dominance_rows).reindex(columns=dominance_cols)
    else:
        dominance_df = pd.DataFrame(columns=dominance_cols)
    return sweep_df, conservative_best_df, dominance_df


def _saved_file(path: Path, saved_files: List[Path]) -> None:
    if path.exists():
        saved_files.append(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate calibration scheduling in a hardware-grounded drift simulator."
    )
    parser.add_argument("--csv", default="data/hardware/hardware_timeseries.csv")
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--mc", type=int, default=1000)
    parser.add_argument("--budget", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--lookahead", type=int, default=10)
    parser.add_argument("--probe-shots", type=int, default=1500)
    parser.add_argument("--probe-cost", type=float, default=2.0)
    parser.add_argument("--partial-cost", type=float, default=4.0)
    parser.add_argument("--full-cost", type=float, default=10.0)
    parser.add_argument("--lambda-cost", type=float, default=0.1)
    parser.add_argument("--failure-threshold", type=float, default=0.8)
    parser.add_argument("--failure-consecutive", type=int, default=3)
    parser.add_argument("--max-probes", type=int, default=40)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--danger-k", type=int, default=5)
    parser.add_argument("--threshold-partial", type=float, default=0.03)
    parser.add_argument("--threshold-full", type=float, default=0.08)

    parser.add_argument("--shock", action="store_true", help="Run the temporary shock robustness experiment.")
    parser.add_argument("--shock-start", type=int, default=None)
    parser.add_argument("--shock-duration", type=int, default=15)
    parser.add_argument("--shock-profile", type=str, default="coherent_burst_short")

    parser.add_argument("--budget-sweep", action="store_true", help="Run budget/beta sweep and dominance analysis.")
    parser.add_argument(
        "--budget-grid",
        type=str,
        default="10,15,20,25,30,35,40,50,60,80,100,120",
    )
    parser.add_argument("--beta-grid", type=str, default="0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--mc-sweep", type=int, default=500)

    parser.add_argument("--make-all", action="store_true", help="Run fit + standard eval + sweep + shock eval + figures.")
    parser.add_argument("--sweep", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be > 0.")
    if args.mc <= 0:
        raise ValueError("--mc must be > 0.")
    if args.mc_sweep <= 0:
        raise ValueError("--mc-sweep must be > 0.")
    if args.budget <= 0.0:
        raise ValueError("--budget must be > 0.")
    if args.beta <= 0.0:
        raise ValueError("--beta must be > 0.")
    if int(args.shock_duration) <= 0:
        raise ValueError("--shock-duration must be > 0.")

    shock_start = args.shock_start
    if shock_start is None:
        shock_start = int(args.steps // 2)
    if int(shock_start) < 0:
        raise ValueError("--shock-start must be >= 0.")
    shock_profile = _validate_shock_profile(args.shock_profile)

    budget_grid = _parse_float_grid(args.budget_grid, "--budget-grid")
    beta_grid = _parse_float_grid(args.beta_grid, "--beta-grid")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fitted_path = outdir / "fitted_drift_params.json"
    saved_files: List[Path] = []

    print(f"[run] Fitting drift model from {args.csv}")
    fitted_params = fit_and_save(csv_path=args.csv, out_path=str(fitted_path))
    print(f"[run] Fitted params saved to {fitted_path}")
    _saved_file(fitted_path, saved_files)

    weights = {"z": 1.0, "o": 1.0, "c": 1.0, "x": 1.0}
    env_cfg = _build_env_config(
        fitted_params=fitted_params,
        seed=args.seed,
        probe_shots=args.probe_shots,
        probe_cost=args.probe_cost,
        partial_cost=args.partial_cost,
        full_cost=args.full_cost,
        lambda_cost=args.lambda_cost,
        failure_threshold=args.failure_threshold,
        failure_consecutive=args.failure_consecutive,
        max_probes_per_horizon=args.max_probes,
        weights=weights,
    )

    print("[run] Standard evaluation")
    standard_results, periods_info = _run_policy_suite(
        env_cfg=env_cfg,
        steps=args.steps,
        mc=args.mc,
        budget=args.budget,
        seed=args.seed,
        lookahead=args.lookahead,
        threshold_partial=args.threshold_partial,
        threshold_full=args.threshold_full,
        beta=args.beta,
        danger_k=args.danger_k,
        collect_traces=True,
        verbose=True,
        seed_offset=100_000,
        shock_start=None,
        shock_window=20,
    )

    standard_summary_body, standard_summary_df = _summarize_results(
        standard_results,
        steps=args.steps,
        shock_start=None,
        shock_window=20,
    )
    standard_payload = _build_summary_payload(
        args=args,
        outdir=outdir,
        periods_info=periods_info,
        summary_body=standard_summary_body,
        shock_mode=False,
        shock_start=None,
        shock_duration=None,
        shock_profile=None,
    )

    summary_json = outdir / "hw_grounded_summary.json"
    summary_csv = outdir / "hw_grounded_summary.csv"
    summary_json.write_text(json.dumps(standard_payload, indent=2), encoding="utf-8")
    standard_summary_df.to_csv(summary_csv, index=False)
    _saved_file(summary_json, saved_files)
    _saved_file(summary_csv, saved_files)

    eval_traces = outdir / "eval_traces.npz"
    _save_eval_traces(
        out_path=eval_traces,
        results=standard_results,
        steps=args.steps,
        shock_start=None,
    )
    _saved_file(eval_traces, saved_files)

    perf_plot = outdir / "performance_vs_time.png"
    ttf_plot = outdir / "time_to_failure_boxplot.png"
    recal_plot = outdir / "recalibrations_cumulative.png"
    _plot_performance_vs_time(standard_results, perf_plot, policies=PLOT_POLICIES)
    _plot_time_to_failure_boxplot(
        standard_results,
        steps=args.steps,
        out_path=ttf_plot,
        policies=PLOT_POLICIES,
    )
    _plot_cumulative_recalibrations(standard_results, recal_plot)
    _saved_file(perf_plot, saved_files)
    _saved_file(ttf_plot, saved_files)
    _saved_file(recal_plot, saved_files)

    run_budget_sweep = bool(args.budget_sweep or args.sweep or args.make_all)
    run_shock = bool(args.shock or args.make_all)

    budget_plot = outdir / "budget_vs_performance.png"
    if run_budget_sweep:
        sweep_env_cfg = env_cfg
        sweep_shock_start: Optional[int] = None
        if args.shock:
            sweep_env_cfg = replace(
                env_cfg,
                shock_start=int(shock_start),
                shock_duration=int(args.shock_duration),
                shock_profile=shock_profile,
            )
            sweep_shock_start = int(shock_start)

        print(
            f"[run] Budget sweep (budgets={budget_grid}, betas={beta_grid}, mc={int(args.mc_sweep)})"
        )
        sweep_df, conservative_best_df, dominance_df = _run_budget_beta_sweep(
            env_cfg=sweep_env_cfg,
            steps=args.steps,
            mc=int(args.mc_sweep),
            seed=args.seed,
            budgets=budget_grid,
            betas=beta_grid,
            threshold_partial=args.threshold_partial,
            threshold_full=args.threshold_full,
            danger_k=args.danger_k,
            lookahead=args.lookahead,
            shock_start=sweep_shock_start,
        )

        sweep_csv = outdir / "budget_sweep_table.csv"
        conservative_best_csv = outdir / "budget_conservative_best.csv"
        dominance_csv = outdir / "budget_dominance_region.csv"
        sweep_df.to_csv(sweep_csv, index=False)
        conservative_best_df.to_csv(conservative_best_csv, index=False)
        dominance_df.to_csv(dominance_csv, index=False)
        _plot_budget_vs_performance(sweep_df, conservative_best_df, dominance_df, budget_plot)

        # Backward-compatible alias from previous script output.
        legacy_sweep_csv = outdir / "sweep_table.csv"
        sweep_df.to_csv(legacy_sweep_csv, index=False)

        _saved_file(sweep_csv, saved_files)
        _saved_file(conservative_best_csv, saved_files)
        _saved_file(dominance_csv, saved_files)
        _saved_file(legacy_sweep_csv, saved_files)
        _saved_file(budget_plot, saved_files)

        if dominance_df.empty:
            print("[sweep] No dominance region found; saved empty dominance table.")
        else:
            print(f"[sweep] Dominance rows: {len(dominance_df)}")
    else:
        quick_mc = min(250, max(40, int(args.mc // 2)))
        quick_budgets = (
            np.asarray([0.5, 0.75, 1.0, 1.25, 1.5], dtype=float) * float(args.budget)
        ).tolist()
        quick_df, quick_best_df, quick_dom_df = _run_budget_beta_sweep(
            env_cfg=env_cfg,
            steps=args.steps,
            mc=quick_mc,
            seed=args.seed,
            budgets=quick_budgets,
            betas=[float(args.beta)],
            threshold_partial=args.threshold_partial,
            threshold_full=args.threshold_full,
            danger_k=args.danger_k,
            lookahead=args.lookahead,
            shock_start=None,
        )
        _plot_budget_vs_performance(quick_df, quick_best_df, quick_dom_df, budget_plot)
        _saved_file(budget_plot, saved_files)

    if run_shock:
        print("[run] Temporary shock robustness evaluation")
        shock_cfg = replace(
            env_cfg,
            shock_start=int(shock_start),
            shock_duration=int(args.shock_duration),
            shock_profile=shock_profile,
        )

        shock_results, shock_periods = _run_policy_suite(
            env_cfg=shock_cfg,
            steps=args.steps,
            mc=args.mc,
            budget=args.budget,
            seed=args.seed,
            lookahead=args.lookahead,
            threshold_partial=args.threshold_partial,
            threshold_full=args.threshold_full,
            beta=args.beta,
            danger_k=args.danger_k,
            collect_traces=True,
            verbose=True,
            seed_offset=700_000,
            shock_start=int(shock_start),
            shock_window=20,
        )

        shock_summary_body, shock_summary_df = _summarize_results(
            shock_results,
            steps=args.steps,
            shock_start=int(shock_start),
            shock_window=20,
        )
        shock_payload = _build_summary_payload(
            args=args,
            outdir=outdir,
            periods_info=shock_periods,
            summary_body=shock_summary_body,
            shock_mode=True,
            shock_start=int(shock_start),
            shock_duration=int(args.shock_duration),
            shock_profile=shock_profile,
        )

        shock_json = outdir / "shock_summary.json"
        shock_csv = outdir / "shock_summary.csv"
        shock_json.write_text(json.dumps(shock_payload, indent=2), encoding="utf-8")
        shock_summary_df.to_csv(shock_csv, index=False)

        shock_traces = outdir / "shock_traces.npz"
        _save_shock_traces(
            out_path=shock_traces,
            results=shock_results,
            steps=args.steps,
            shock_start=int(shock_start),
            shock_duration=int(args.shock_duration),
        )

        shock_perf_plot = outdir / "shock_performance_vs_time.png"
        _plot_performance_vs_time(
            shock_results,
            shock_perf_plot,
            policies=PLOT_POLICIES,
            switch_step=int(shock_start),
        )

        _print_robustness_check(shock_summary_df)

        _saved_file(shock_json, saved_files)
        _saved_file(shock_csv, saved_files)
        _saved_file(shock_traces, saved_files)
        _saved_file(shock_perf_plot, saved_files)

    if args.make_all:
        print("[run] Generating figure set")
        from .make_figures import generate_figures

        figure_paths = generate_figures(csv_path=str(args.csv), outdir=str(outdir))
        for figure_path in figure_paths:
            _saved_file(Path(figure_path), saved_files)

    print("[done] Saved files:")
    for path in sorted({str(p.resolve()) for p in saved_files}):
        print(f"  - {path}")

    print("")
    print("WHAT TO USE IN SLIDES")
    print("1) results/hardware_drift_timeseries.png: Real hardware channels drift and burst over time.")
    print("2) results/performance_vs_time.png: Conservative tracks periodic quality under matched cost on baseline drift.")
    if run_shock:
        print(
            "3) results/shock_performance_vs_time.png: During temporary shock, conservative avoids collapse/overreaction better than fixed schedules."
        )
    else:
        print(
            "3) results/time_to_failure_boxplot.png: Time-to-failure spread highlights policy robustness under matched budgets."
        )


if __name__ == "__main__":
    main()
