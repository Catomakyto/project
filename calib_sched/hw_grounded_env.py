from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

ACTION_NO_ACTION = 0
ACTION_PROBE = 1
ACTION_PARTIAL_RECAL = 2
ACTION_FULL_RECAL = 3
NUM_ACTIONS = 4

CHANNELS: Tuple[str, str, str, str] = ("z", "o", "c", "x")


@dataclass(frozen=True)
class ChannelDriftParams:
    """Drift process parameters for one anomaly channel."""

    mu: float
    sigma: float
    p_burst: float
    sigma_burst: float
    initial_mean: float
    initial_sigma: float
    min_value: float = 0.0
    max_value: float = 1.0


@dataclass
class HWGroundedEnvConfig:
    """Configuration for the hardware-grounded drift environment."""

    channels: Mapping[str, ChannelDriftParams]
    seed: int = 0
    probe_shots: int = 1500
    probe_cost: float = 0.5
    partial_cost: float = 4.0
    full_cost: float = 10.0
    lambda_cost: float = 0.1
    channel_weights: Mapping[str, float] = field(
        default_factory=lambda: {"z": 1.0, "o": 1.0, "c": 1.0, "x": 1.0}
    )
    failure_threshold: float = 0.8
    failure_consecutive: int = 3
    max_probes_per_horizon: int = 40
    partial_groups: Mapping[str, Tuple[str, ...]] = field(
        default_factory=lambda: {
            "readout": ("z", "o"),
            "coherent": ("c", "x"),
        }
    )
    partial_strategy: str = "worst_group"
    recal_residual_scale: float = 0.35
    shock_start: Optional[int] = None
    shock_duration: int = 15
    shock_profile: str = "coherent_burst_short"

    def validate(self) -> None:
        missing = [ch for ch in CHANNELS if ch not in self.channels]
        if missing:
            raise ValueError(f"Missing channel params for: {missing}")
        if self.probe_shots <= 0:
            raise ValueError("probe_shots must be > 0.")
        if self.failure_consecutive <= 0:
            raise ValueError("failure_consecutive must be > 0.")
        if self.max_probes_per_horizon < 0:
            raise ValueError("max_probes_per_horizon must be >= 0.")
        if self.partial_strategy not in {"worst_group", "readout", "coherent"}:
            raise ValueError(
                "partial_strategy must be one of {'worst_group', 'readout', 'coherent'}."
            )
        if self.shock_start is not None and int(self.shock_start) < 0:
            raise ValueError("shock_start must be >= 0 when provided.")
        if int(self.shock_duration) <= 0:
            raise ValueError("shock_duration must be > 0.")
        if self.shock_profile not in {"coherent_burst_short", "readout_spike_short"}:
            raise ValueError("shock_profile must be one of {'coherent_burst_short', 'readout_spike_short'}.")


@dataclass
class EnvSnapshot:
    """Serializable simulation snapshot for deterministic counterfactual replay."""

    timestep: int
    drift: Dict[str, float]
    consecutive_failures: int
    time_to_failure: Optional[int]
    last_observation: Optional[Dict[str, float]]
    last_probe_step: int
    rng_state: MutableMapping[str, object]


@dataclass
class StepResult:
    """Single-step transition result."""

    timestep: int
    action: int
    cost: float
    performance: float
    reward: float
    failure: bool
    done: bool
    observation: Optional[Dict[str, float]]
    hidden_drift: Dict[str, float]


def load_channel_params_from_fit(fitted: Mapping[str, object]) -> Dict[str, ChannelDriftParams]:
    """Convert fitted drift JSON payload to typed channel params."""

    if "channels" not in fitted:
        raise ValueError("Fitted payload missing 'channels' key.")
    channels_obj = fitted["channels"]
    if not isinstance(channels_obj, Mapping):
        raise ValueError("Fitted payload 'channels' must be a mapping.")

    out: Dict[str, ChannelDriftParams] = {}
    for channel in CHANNELS:
        raw = channels_obj.get(channel)
        if not isinstance(raw, Mapping):
            raise ValueError(f"Missing or invalid fitted channel '{channel}'.")
        out[channel] = ChannelDriftParams(
            mu=float(raw.get("mu", 0.0)),
            sigma=max(1e-6, float(raw.get("sigma", 1e-4))),
            p_burst=float(np.clip(float(raw.get("p_burst", 0.1)), 0.0, 0.95)),
            sigma_burst=max(1e-6, float(raw.get("sigma_burst", 3e-4))),
            initial_mean=max(0.0, float(raw.get("initial_mean", 0.0))),
            initial_sigma=max(1e-6, float(raw.get("initial_sigma", 1e-4))),
            min_value=float(raw.get("min_value", 0.0)),
            max_value=float(raw.get("max_value", 1.0)),
        )
    return out


class HWGroundedEnv:
    """Lightweight simulator grounded by fitted hardware drift channels."""

    def __init__(self, cfg: HWGroundedEnvConfig, seed: Optional[int] = None) -> None:
        self.cfg = cfg
        self.cfg.validate()

        self.seed = int(cfg.seed if seed is None else seed)
        self.rng = np.random.default_rng(self.seed)

        self.t = 0
        self._drift: Dict[str, float] = {}
        self._consecutive_failures = 0
        self.time_to_failure: Optional[int] = None
        self.last_observation: Optional[Dict[str, float]] = None
        self.last_probe_step = -1
        self.action_counts = {a: 0 for a in range(NUM_ACTIONS)}
        self.reset(seed=self.seed)

    @property
    def drift(self) -> Dict[str, float]:
        return dict(self._drift)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.default_rng(self.seed)

        self.t = 0
        self._consecutive_failures = 0
        self.time_to_failure = None
        self.action_counts = {a: 0 for a in range(NUM_ACTIONS)}

        self._drift = {ch: self._sample_baseline(ch) for ch in CHANNELS}
        self.last_observation = self._observe(self._drift)
        self.last_probe_step = 0

    def action_cost(self, action: int) -> float:
        if action == ACTION_NO_ACTION:
            return 0.0
        if action == ACTION_PROBE:
            return float(self.cfg.probe_cost)
        if action == ACTION_PARTIAL_RECAL:
            return float(self.cfg.partial_cost)
        if action == ACTION_FULL_RECAL:
            return float(self.cfg.full_cost)
        raise ValueError(f"Unsupported action: {action}")

    def can_probe(self) -> bool:
        """Return whether another probe action is allowed under probe cap."""
        return int(self.action_counts.get(ACTION_PROBE, 0)) < int(self.cfg.max_probes_per_horizon)

    def get_context(self, budget_remaining: Optional[float] = None) -> Dict[str, float]:
        obs = self.last_observation if self.last_observation is not None else self._observe(self._drift)
        obs_age = max(0, self.t - self.last_probe_step)

        x_anomaly = float(obs["x"])
        x_odd_parity = float(np.clip(0.5 + x_anomaly, 0.0, 1.0))

        context = {
            "readout_mean_error_zero": float(obs["z"]),
            "readout_mean_error_one": float(obs["o"]),
            "coherent_mean_anomaly": float(obs["c"]),
            "crosstalk_mean_anomaly": float(x_anomaly),
            "crosstalk_mean_odd_parity": float(x_odd_parity),
            "obs_age": float(obs_age),
            "timestep": float(self.t),
            "probe_cap_remaining": float(
                max(0, int(self.cfg.max_probes_per_horizon) - int(self.action_counts.get(ACTION_PROBE, 0)))
            ),
        }
        if budget_remaining is not None:
            context["budget_remaining"] = float(budget_remaining)
        return context

    def snapshot(self) -> EnvSnapshot:
        return EnvSnapshot(
            timestep=int(self.t),
            drift=dict(self._drift),
            consecutive_failures=int(self._consecutive_failures),
            time_to_failure=self.time_to_failure,
            last_observation=None if self.last_observation is None else dict(self.last_observation),
            last_probe_step=int(self.last_probe_step),
            rng_state=copy.deepcopy(self.rng.bit_generator.state),
        )

    def restore(self, snapshot: EnvSnapshot) -> None:
        self.t = int(snapshot.timestep)
        self._drift = dict(snapshot.drift)
        self._consecutive_failures = int(snapshot.consecutive_failures)
        self.time_to_failure = snapshot.time_to_failure
        self.last_observation = (
            None if snapshot.last_observation is None else dict(snapshot.last_observation)
        )
        self.last_probe_step = int(snapshot.last_probe_step)
        self.rng.bit_generator.state = copy.deepcopy(snapshot.rng_state)

    def counterfactual_would_fail_without_recal(
        self,
        snapshot: EnvSnapshot,
        lookahead_steps: int,
    ) -> bool:
        """Return True if no recalibration for next lookahead steps hits failure."""

        if lookahead_steps <= 0:
            return False

        clone = HWGroundedEnv(self.cfg, seed=self.seed)
        clone.restore(snapshot)

        for _ in range(lookahead_steps):
            clone._evolve_drift()
            performance = clone._compute_performance(clone._drift)
            if performance < clone.cfg.failure_threshold:
                clone._consecutive_failures += 1
            else:
                clone._consecutive_failures = 0

            if clone._consecutive_failures >= clone.cfg.failure_consecutive:
                return True
            clone.t += 1
        return False

    def step(self, action: int) -> StepResult:
        if action not in {ACTION_NO_ACTION, ACTION_PROBE, ACTION_PARTIAL_RECAL, ACTION_FULL_RECAL}:
            raise ValueError(f"Unsupported action: {action}")

        actual_action = int(action)
        if action == ACTION_PROBE and not self.can_probe():
            actual_action = ACTION_NO_ACTION

        self._evolve_drift()
        if actual_action == ACTION_PARTIAL_RECAL:
            self._apply_partial_recalibration()
        elif actual_action == ACTION_FULL_RECAL:
            self._apply_full_recalibration()

        observation: Optional[Dict[str, float]] = None
        if actual_action == ACTION_PROBE:
            observation = self._observe(self._drift)
            self.last_observation = dict(observation)
            self.last_probe_step = int(self.t)

        performance = self._compute_performance(self._drift)
        cost = self.action_cost(actual_action)
        reward = float(performance - self.cfg.lambda_cost * cost)

        failure = performance < self.cfg.failure_threshold
        if failure:
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0

        done = False
        if self.time_to_failure is None and self._consecutive_failures >= self.cfg.failure_consecutive:
            self.time_to_failure = int(self.t + 1)
            done = True

        self.action_counts[actual_action] += 1
        result = StepResult(
            timestep=int(self.t),
            action=actual_action,
            cost=float(cost),
            performance=float(performance),
            reward=float(reward),
            failure=bool(failure),
            done=bool(done),
            observation=observation,
            hidden_drift=dict(self._drift),
        )
        self.t += 1
        return result

    def _sample_baseline(self, channel: str) -> float:
        p = self.cfg.channels[channel]
        value = self.rng.normal(loc=p.initial_mean, scale=p.initial_sigma)
        return float(np.clip(value, p.min_value, p.max_value))

    def _observe(self, drift: Mapping[str, float]) -> Dict[str, float]:
        shots = int(self.cfg.probe_shots)
        obs: Dict[str, float] = {}
        for channel in CHANNELS:
            true_val = float(np.clip(drift[channel], 0.0, 1.0))
            successes = int(self.rng.binomial(shots, true_val))
            # Wilson-like smoothing via add-two correction for stability near 0/1.
            p_hat = (successes + 2.0) / (shots + 4.0)
            obs[channel] = float(np.clip(p_hat, 0.0, 1.0))
        return obs

    def _evolve_drift(self) -> None:
        for channel in CHANNELS:
            p = self.cfg.channels[channel]
            sigma, p_burst, sigma_burst = self._effective_drift_params(channel)
            increment = self.rng.normal(loc=p.mu, scale=sigma)
            if self.rng.random() < p_burst:
                increment += self.rng.normal(loc=0.0, scale=sigma_burst)
            next_val = self._drift[channel] + float(increment)
            self._drift[channel] = float(np.clip(next_val, p.min_value, p.max_value))

    def _effective_drift_params(self, channel: str) -> Tuple[float, float, float]:
        p = self.cfg.channels[channel]
        sigma = max(1e-9, float(p.sigma))
        p_burst = float(np.clip(float(p.p_burst), 0.0, 1.0))
        sigma_burst = max(1e-9, float(p.sigma_burst))

        if not self._is_in_shock_window():
            return sigma, p_burst, sigma_burst

        profile = self.cfg.shock_profile
        if profile == "coherent_burst_short":
            if channel == "c":
                sigma *= 4.0
                sigma_burst *= 4.0
                p_burst = min(0.8, p_burst * 4.0)
            elif channel == "z":
                sigma *= 1.3
        elif profile == "readout_spike_short":
            if channel in {"z", "o"}:
                sigma *= 2.5
                p_burst = min(0.6, p_burst * 3.0)

        sigma = max(1e-9, float(sigma))
        p_burst = float(np.clip(p_burst, 0.0, 1.0))
        sigma_burst = max(1e-9, float(sigma_burst))
        return sigma, p_burst, sigma_burst

    def _is_in_shock_window(self) -> bool:
        if self.cfg.shock_start is None:
            return False
        start = int(self.cfg.shock_start)
        end = start + int(self.cfg.shock_duration)
        return start <= int(self.t) < end

    def _apply_partial_recalibration(self) -> None:
        for channel in self._select_partial_channels():
            self._drift[channel] = self._sample_recalibrated(channel)

    def _apply_full_recalibration(self) -> None:
        for channel in CHANNELS:
            self._drift[channel] = self._sample_recalibrated(channel)

    def _sample_recalibrated(self, channel: str) -> float:
        p = self.cfg.channels[channel]
        residual_sigma = max(1e-6, self.cfg.recal_residual_scale * p.initial_sigma)
        value = self.rng.normal(loc=p.initial_mean, scale=residual_sigma)
        return float(np.clip(value, p.min_value, p.max_value))

    def _select_partial_channels(self) -> Sequence[str]:
        if self.cfg.partial_strategy == "readout":
            return self.cfg.partial_groups.get("readout", ("z", "o"))
        if self.cfg.partial_strategy == "coherent":
            return self.cfg.partial_groups.get("coherent", ("c", "x"))

        contributions = {
            channel: float(self.cfg.channel_weights.get(channel, 1.0)) * float(self._drift.get(channel, 0.0))
            for channel in CHANNELS
        }
        worst_channel = max(contributions, key=lambda ch: contributions[ch])
        if worst_channel in {"z", "o"}:
            return ("z", "o")
        if worst_channel == "c":
            return ("c",)
        return ("x",)

    def _compute_performance(self, drift: Mapping[str, float]) -> float:
        penalty = 0.0
        for channel in CHANNELS:
            penalty += float(self.cfg.channel_weights.get(channel, 1.0)) * float(drift[channel])
        return float(np.clip(1.0 - penalty, 0.0, 1.0))
