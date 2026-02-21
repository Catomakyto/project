from __future__ import annotations

import csv
import json
import pathlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ..policies.baselines import BasePolicy, sample_action
from ..sim.env import CalibEnv, EnvConfig


@dataclass
class EpisodeMetrics:
    episode: int
    cumulative_utility: float
    mean_utility: float
    mean_performance: float
    silent_failure_rate: float
    mean_cost: float


def _flatten_context(context: Mapping[str, float]) -> Dict[str, float]:
    return {f"ctx_{k}": float(v) for k, v in context.items() if isinstance(v, (int, float))}


def run_episode(
    env: CalibEnv,
    policy: BasePolicy,
    horizon: int,
    episode_index: int,
    collect_logs: bool,
) -> Tuple[EpisodeMetrics, List[Dict[str, float | int | str]]]:
    records: List[Dict[str, float | int | str]] = []

    cumulative_utility = 0.0
    perf = []
    costs = []
    silent = 0

    for t in range(horizon):
        workload = env.sample_workload()
        context = env.observe_context(workload)

        sampled = sample_action(policy, context, env.rng)
        action = int(sampled["action"])
        propensity = float(sampled["propensity"])

        step = env.step(action=action, workload=workload)
        reward = float(step["utility"])

        policy.update(context=context, action=action, reward=reward)

        cumulative_utility += reward
        perf.append(float(step["performance"]))
        costs.append(float(step["cost"]))
        silent += int(step["silent_failure"])

        if collect_logs:
            row: Dict[str, float | int | str] = {
                "episode": int(episode_index),
                "step": int(t),
                "action": int(action),
                "reward": float(reward),
                "performance": float(step["performance"]),
                "cost": float(step["cost"]),
                "silent_failure": int(step["silent_failure"]),
                "propensity": float(propensity),
                "workload_qubits": "|".join(str(q) for q in workload["qubits"]),  # type: ignore[index]
                "workload_depth": int(workload["depth"]),  # type: ignore[index]
                "workload_two_qubit_count": int(workload["two_qubit_count"]),  # type: ignore[index]
            }
            row.update(_flatten_context(context))
            records.append(row)

    metrics = EpisodeMetrics(
        episode=episode_index,
        cumulative_utility=float(cumulative_utility),
        mean_utility=float(cumulative_utility / max(1, horizon)),
        mean_performance=float(np.mean(perf) if perf else 0.0),
        silent_failure_rate=float(silent / max(1, horizon)),
        mean_cost=float(np.mean(costs) if costs else 0.0),
    )
    return metrics, records


def run_policy_episodes(
    env_cfg: EnvConfig,
    policy_factory: Callable[[], BasePolicy],
    horizon: int,
    episodes: int,
    seed: int,
    collect_logs: bool,
) -> Tuple[List[EpisodeMetrics], List[Dict[str, float | int | str]]]:
    metrics: List[EpisodeMetrics] = []
    all_records: List[Dict[str, float | int | str]] = []

    for ep in range(episodes):
        env = CalibEnv(env_cfg, seed=seed + ep)
        policy = policy_factory()
        ep_metrics, ep_records = run_episode(
            env=env,
            policy=policy,
            horizon=horizon,
            episode_index=ep,
            collect_logs=collect_logs,
        )
        metrics.append(ep_metrics)
        all_records.extend(ep_records)

    return metrics, all_records


def save_records(records: Sequence[Mapping[str, object]], path: str) -> None:
    if not records:
        return
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    suffix = out.suffix.lower()
    if suffix == ".parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception as exc:  # pragma: no cover - depends on optional dependency
            raise RuntimeError("pyarrow is required to write parquet logs") from exc

        keys = sorted({k for rec in records for k in rec.keys()})
        data = {k: [rec.get(k) for rec in records] for k in keys}
        table = pa.table(data)
        pq.write_table(table, out)
        return

    if suffix == ".jsonl":
        with out.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
        return

    # default CSV
    keys = sorted({k for rec in records for k in rec.keys()})
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in keys})


def save_episode_metrics(metrics: Sequence[EpisodeMetrics], path: str) -> None:
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [m.__dict__ for m in metrics]
    if not rows:
        return
    keys = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
