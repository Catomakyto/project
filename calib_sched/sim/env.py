from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .executor import compute_sentinel_stats, simulate_workload_performance
from .hidden_state import DriftParams, HiddenState

ACTION_IDLE = 0
ACTION_PROBE = 1
ACTION_PARTIAL = 2
ACTION_FULL = 3
NUM_ACTIONS = 4


@dataclass
class WorkloadDistribution:
    qubit_pool: List[int]
    min_qubits: int
    max_qubits: int
    depth_range: Tuple[int, int]
    two_qubit_range: Tuple[int, int]


@dataclass
class UtilityConfig:
    lambda_weight: float
    mu_penalty: float
    silent_failure_tau: float


@dataclass
class EnvConfig:
    seed: int
    num_qubits: int
    monitor_qubits: List[int]
    monitor_pair: Tuple[int, int]
    sentinel_shots: int
    coherent_repeats: List[int]
    action_costs: Dict[int, float]
    workload: WorkloadDistribution
    utility: UtilityConfig
    drift: DriftParams
    sensitivity: Dict[str, float]


class CalibEnv:
    def __init__(self, cfg: EnvConfig, seed: int | None = None) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed if seed is None else seed)
        self.state = HiddenState(
            num_qubits=cfg.num_qubits,
            crosstalk_pairs=[cfg.monitor_pair],
            drift=cfg.drift,
            rng=self.rng,
        )
        self.t = 0
        self.last_full_cal = 0
        self.last_partial_cal = 0

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = HiddenState(
            num_qubits=self.cfg.num_qubits,
            crosstalk_pairs=[self.cfg.monitor_pair],
            drift=self.cfg.drift,
            rng=self.rng,
        )
        self.t = 0
        self.last_full_cal = 0
        self.last_partial_cal = 0

    def sample_workload(self) -> Dict[str, object]:
        pool = [int(q) for q in self.cfg.workload.qubit_pool]
        max_q = max(1, min(self.cfg.workload.max_qubits, len(pool)))
        min_q = max(1, min(self.cfg.workload.min_qubits, max_q))
        n_qubits = int(self.rng.integers(min_q, max_q + 1))
        chosen = sorted(self.rng.choice(pool, size=n_qubits, replace=False).tolist())
        depth_lo, depth_hi = self.cfg.workload.depth_range
        two_lo, two_hi = self.cfg.workload.two_qubit_range
        depth = int(self.rng.integers(depth_lo, depth_hi + 1))
        two_qubit_count = int(self.rng.integers(two_lo, two_hi + 1))
        return {
            "qubits": chosen,
            "depth": depth,
            "two_qubit_count": two_qubit_count,
        }

    def observe_context(self, workload: Mapping[str, object]) -> Dict[str, float]:
        stats = compute_sentinel_stats(
            state=self.state,
            monitor_qubits=self.cfg.monitor_qubits,
            monitor_pair=self.cfg.monitor_pair,
            shots=self.cfg.sentinel_shots,
            coherent_repeats=self.cfg.coherent_repeats,
            rng=self.rng,
        )
        context = dict(stats)
        context["time_since_full_cal"] = float(self.t - self.last_full_cal)
        context["time_since_partial_cal"] = float(self.t - self.last_partial_cal)
        context["workload_num_qubits"] = float(len(workload["qubits"]))  # type: ignore[index]
        context["workload_depth"] = float(workload["depth"])  # type: ignore[index]
        context["workload_two_qubit_count"] = float(workload["two_qubit_count"])  # type: ignore[index]
        context["timestep"] = float(self.t)
        return context

    def step(self, action: int, workload: Mapping[str, object]) -> Dict[str, float | int]:
        if action not in {ACTION_IDLE, ACTION_PROBE, ACTION_PARTIAL, ACTION_FULL}:
            raise ValueError(f"Unsupported action {action}")

        if action == ACTION_PARTIAL:
            self.state.calibrate_partial()
            self.last_partial_cal = self.t
        elif action == ACTION_FULL:
            self.state.calibrate_full()
            self.last_partial_cal = self.t
            self.last_full_cal = self.t

        cost = float(self.cfg.action_costs.get(action, 0.0))
        performance = simulate_workload_performance(
            state=self.state,
            workload=workload,
            sensitivity=self.cfg.sensitivity,
            rng=self.rng,
        )
        silent_failure = int(action == ACTION_IDLE and performance < self.cfg.utility.silent_failure_tau)

        utility = (
            performance
            - self.cfg.utility.lambda_weight * cost
            - self.cfg.utility.mu_penalty * float(silent_failure)
        )

        # Hidden drift evolves after each decision/workload event.
        self.state.update()
        self.t += 1

        return {
            "utility": float(utility),
            "performance": float(performance),
            "cost": float(cost),
            "silent_failure": int(silent_failure),
        }
