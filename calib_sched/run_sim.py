from __future__ import annotations

import argparse
import json
import os
import pathlib
from typing import Any, Dict, List

import yaml

from .eval.protocol import run_policy_episodes, save_episode_metrics, save_records
from .policies.baselines import EpsilonGreedyPolicy, PeriodicPolicy
from .policies.conservative_bandit import ConservativeBanditPolicy
from .sim.env import EnvConfig, UtilityConfig, WorkloadDistribution
from .sim.hidden_state import DriftParams


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _build_env_config(cfg: Dict[str, Any], seed: int) -> EnvConfig:
    sim = cfg["simulation"]
    hardware = cfg["hardware"]

    drift = DriftParams(**sim["drift"])
    util = UtilityConfig(
        lambda_weight=float(sim["utility"]["lambda_weight"]),
        mu_penalty=float(sim["utility"]["mu_penalty"]),
        silent_failure_tau=float(sim["utility"]["silent_failure_tau"]),
    )
    workload = WorkloadDistribution(
        qubit_pool=[int(q) for q in sim["workload"]["qubit_pool"]],
        min_qubits=int(sim["workload"]["min_qubits"]),
        max_qubits=int(sim["workload"]["max_qubits"]),
        depth_range=(
            int(sim["workload"]["depth_range"][0]),
            int(sim["workload"]["depth_range"][1]),
        ),
        two_qubit_range=(
            int(sim["workload"]["two_qubit_range"][0]),
            int(sim["workload"]["two_qubit_range"][1]),
        ),
    )
    action_costs = {
        0: float(sim["action_costs"]["idle"]),
        1: float(sim["action_costs"]["probe"]),
        2: float(sim["action_costs"]["partial"]),
        3: float(sim["action_costs"]["full"]),
    }

    return EnvConfig(
        seed=seed,
        num_qubits=int(sim["num_qubits"]),
        monitor_qubits=[int(q) for q in hardware["monitor_qubits"]],
        monitor_pair=(int(hardware["monitor_pair"][0]), int(hardware["monitor_pair"][1])),
        sentinel_shots=int(hardware["sentinel_suite"]["shots_per_circuit"]),
        coherent_repeats=[int(r) for r in hardware["sentinel_suite"]["coherent_repeats"]],
        action_costs=action_costs,
        workload=workload,
        utility=util,
        drift=drift,
        sensitivity={k: float(v) for k, v in sim["sensitivity"].items()},
    )


def _mean_metric(rows: List[Dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(float(r[key]) for r in rows) / len(rows))


def _plot_metrics(output_dir: pathlib.Path, periodic_csv: pathlib.Path, bandit_csv: pathlib.Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import csv

    import matplotlib.pyplot as plt

    def read_rows(path: pathlib.Path) -> List[Dict[str, float]]:
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return [{k: float(v) for k, v in row.items()} for row in reader]

    periodic = read_rows(periodic_csv)
    bandit = read_rows(bandit_csv)

    labels = ["mean_utility", "mean_performance", "silent_failure_rate", "mean_cost"]
    p_vals = [_mean_metric(periodic, k) for k in labels]
    b_vals = [_mean_metric(bandit, k) for k in labels]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([i - width / 2 for i in x], p_vals, width=width, label="periodic")
    ax.bar([i + width / 2 for i in x], b_vals, width=width, label="conservative_bandit")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Simulation Policy Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "sim_metrics.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulation evaluation for calibration scheduling.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="Output directory for metrics/plots")
    parser.add_argument("--log-out", default="", help="Optional logged bandit data path (.parquet/.csv/.jsonl)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    sim_cfg = cfg["simulation"]

    seed = int(sim_cfg["seed"] if args.seed is None else args.seed)
    horizon = int(sim_cfg["horizon"])
    episodes = int(sim_cfg["episodes"])

    env_cfg = _build_env_config(cfg, seed=seed)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    periodic_period = int(sim_cfg["baseline"]["period"])
    bandit_params = dict(sim_cfg["conservative_bandit"])

    def periodic_factory() -> PeriodicPolicy:
        return PeriodicPolicy(period=periodic_period)

    def bandit_factory() -> ConservativeBanditPolicy:
        return ConservativeBanditPolicy(
            ridge=float(bandit_params["ridge"]),
            beta=float(bandit_params["beta"]),
            uncertainty_threshold=float(bandit_params["uncertainty_threshold"]),
            risk_threshold=float(bandit_params["risk_threshold"]),
            improvement_margin=float(bandit_params["improvement_margin"]),
            exploration_epsilon=float(bandit_params["exploration_epsilon"]),
            safe_period=periodic_period,
        )

    periodic_metrics, _ = run_policy_episodes(
        env_cfg=env_cfg,
        policy_factory=periodic_factory,
        horizon=horizon,
        episodes=episodes,
        seed=seed,
        collect_logs=False,
    )
    bandit_metrics, _ = run_policy_episodes(
        env_cfg=env_cfg,
        policy_factory=bandit_factory,
        horizon=horizon,
        episodes=episodes,
        seed=seed + 10_000,
        collect_logs=False,
    )

    periodic_csv = out_dir / "periodic_metrics.csv"
    bandit_csv = out_dir / "conservative_bandit_metrics.csv"
    save_episode_metrics(periodic_metrics, str(periodic_csv))
    save_episode_metrics(bandit_metrics, str(bandit_csv))
    _plot_metrics(out_dir, periodic_csv=periodic_csv, bandit_csv=bandit_csv)

    summary = {
        "seed": seed,
        "horizon": horizon,
        "episodes": episodes,
        "periodic": {
            "mean_utility": _mean_metric([m.__dict__ for m in periodic_metrics], "mean_utility"),
            "mean_performance": _mean_metric([m.__dict__ for m in periodic_metrics], "mean_performance"),
            "silent_failure_rate": _mean_metric([m.__dict__ for m in periodic_metrics], "silent_failure_rate"),
            "mean_cost": _mean_metric([m.__dict__ for m in periodic_metrics], "mean_cost"),
        },
        "conservative_bandit": {
            "mean_utility": _mean_metric([m.__dict__ for m in bandit_metrics], "mean_utility"),
            "mean_performance": _mean_metric([m.__dict__ for m in bandit_metrics], "mean_performance"),
            "silent_failure_rate": _mean_metric([m.__dict__ for m in bandit_metrics], "silent_failure_rate"),
            "mean_cost": _mean_metric([m.__dict__ for m in bandit_metrics], "mean_cost"),
        },
    }

    if args.log_out:
        behavior_eps = float(sim_cfg["logging"]["behavior_epsilon"])
        log_episodes = int(sim_cfg["logging"]["episodes"])

        def behavior_factory() -> EpsilonGreedyPolicy:
            return EpsilonGreedyPolicy(
                base_policy=PeriodicPolicy(period=periodic_period),
                epsilon=behavior_eps,
            )

        _, logs = run_policy_episodes(
            env_cfg=env_cfg,
            policy_factory=behavior_factory,
            horizon=horizon,
            episodes=log_episodes,
            seed=seed + 20_000,
            collect_logs=True,
        )
        save_records(logs, args.log_out)
        summary["logged_data"] = {
            "path": args.log_out,
            "rows": len(logs),
            "behavior": f"epsilon-greedy periodic (epsilon={behavior_eps})",
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote simulation summary: {summary_path}")
    print(f"Wrote plot: {out_dir / 'sim_metrics.png'}")
    if args.log_out:
        print(f"Wrote logged data: {args.log_out}")


if __name__ == "__main__":
    main()
