from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Mapping, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calib_sched.eval_hw_grounded import _build_env_config, _policy_factories
from calib_sched.hw_drift_fit import fit_drift_params
from calib_sched.hw_grounded_env import (
    ACTION_FULL_RECAL,
    ACTION_NO_ACTION,
    ACTION_PARTIAL_RECAL,
    ACTION_PROBE,
    HWGroundedEnv,
    HWGroundedEnvConfig,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

POLICY_ORDER: Tuple[str, ...] = ("periodic_full", "periodic_partial", "conservative_bandit")
BETA_GRID: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)


@dataclass
class PolicyRun:
    traces: np.ndarray
    time_to_failure: np.ndarray
    cumulative_reward: np.ndarray


def _fallback_order(action: int) -> Sequence[int]:
    if action == ACTION_FULL_RECAL:
        return (ACTION_FULL_RECAL, ACTION_PARTIAL_RECAL, ACTION_NO_ACTION, ACTION_PROBE)
    if action == ACTION_PARTIAL_RECAL:
        return (ACTION_PARTIAL_RECAL, ACTION_NO_ACTION, ACTION_PROBE)
    if action == ACTION_PROBE:
        return (ACTION_PROBE, ACTION_NO_ACTION)
    return (ACTION_NO_ACTION,)


def _enforce_budget(action: int, budget_remaining: float, env: HWGroundedEnv) -> int:
    for candidate in _fallback_order(int(action)):
        if candidate == ACTION_PROBE and not env.can_probe():
            continue
        if env.action_cost(candidate) <= float(budget_remaining) + 1e-12:
            return int(candidate)
    return ACTION_NO_ACTION


def _load_or_fit_params(
    fitted_json: Path,
    csv_path: Path,
    fallback_out: Path,
) -> Dict[str, object]:
    if fitted_json.exists():
        return json.loads(fitted_json.read_text(encoding="utf-8"))
    params = fit_drift_params(str(csv_path))
    fallback_out.parent.mkdir(parents=True, exist_ok=True)
    fallback_out.write_text(json.dumps(params, indent=2), encoding="utf-8")
    return params


def _run_policy(
    policy_factory: Callable[[], object],
    env_cfg: HWGroundedEnvConfig,
    horizon: int,
    seeds: np.ndarray,
    budget: float,
) -> PolicyRun:
    n = int(seeds.size)
    traces = np.zeros((n, horizon), dtype=float)
    time_to_failure = np.full(n, float(horizon + 1), dtype=float)
    cumulative_reward = np.zeros(n, dtype=float)

    for i, seed in enumerate(seeds):
        env = HWGroundedEnv(env_cfg, seed=int(seed))
        policy = policy_factory()
        budget_remaining = float(budget)
        total_reward = 0.0
        ttf = int(horizon + 1)

        for _t in range(horizon):
            context = env.get_context(budget_remaining=budget_remaining)
            proposed_action = int(policy.select_action(context))
            action = _enforce_budget(proposed_action, budget_remaining=budget_remaining, env=env)

            step = env.step(action)
            budget_remaining -= float(step.cost)
            total_reward += float(step.reward)

            policy.update(context, int(step.action), float(step.reward))
            traces[i, int(step.timestep)] = float(step.performance)

            if bool(step.done) and ttf == horizon + 1:
                ttf = int(step.timestep + 1)

        cumulative_reward[i] = float(total_reward)
        time_to_failure[i] = float(ttf)

    return PolicyRun(
        traces=traces,
        time_to_failure=time_to_failure,
        cumulative_reward=cumulative_reward,
    )


def _metrics(run: PolicyRun, horizon: int, collapse_threshold: float) -> Dict[str, float]:
    ttf = np.asarray(run.time_to_failure, dtype=float)
    reward = np.asarray(run.cumulative_reward, dtype=float)
    del horizon
    collapse_rate = float(np.mean((run.traces < float(collapse_threshold)).astype(float)))
    return {
        "median_time_to_failure": float(np.median(ttf)),
        "collapse_rate": collapse_rate,
        "cumulative_reward_mean": float(np.mean(reward)),
    }


def _trace_ci(run: PolicyRun) -> Tuple[np.ndarray, np.ndarray]:
    traces = np.asarray(run.traces, dtype=float)
    mean = np.mean(traces, axis=0)
    if traces.shape[0] > 1:
        se = np.std(traces, axis=0, ddof=1) / np.sqrt(traces.shape[0])
    else:
        se = np.zeros(traces.shape[1], dtype=float)
    return mean, 1.96 * se


def _plot_traces(
    runs: Mapping[str, PolicyRun],
    labels: Mapping[str, str],
    out_path: Path,
    title: str,
    horizon: int,
    shock_start: int,
    shock_duration: int,
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "legend.fontsize": 13,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    )

    fig, ax = plt.subplots(figsize=(12.0, 7.0))

    for key in labels:
        run = runs.get(key)
        if run is None:
            continue
        mean, ci = _trace_ci(run)
        x = np.arange(horizon, dtype=float)
        ax.plot(x, mean, linewidth=2.4, label=labels[key])
        ax.fill_between(
            x,
            np.clip(mean - ci, 0.0, 1.0),
            np.clip(mean + ci, 0.0, 1.0),
            alpha=0.16,
        )

    x0 = float(shock_start)
    x1 = float(shock_start + shock_duration)
    ax.axvspan(x0, x1, color="#f2c9b1", alpha=0.30, label="Shock region")
    y_text = 0.98
    ax.text(
        x0 + 1.0,
        y_text,
        f"shock [{shock_start}, {shock_start + shock_duration})",
        fontsize=12,
        ha="left",
        va="top",
    )

    ax.set_xlim(0, max(1, horizon - 1))
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Fidelity (performance proxy)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _select_best_beta(
    env_cfg: HWGroundedEnvConfig,
    horizon: int,
    seeds: np.ndarray,
    budget: float,
    threshold_partial: float,
    threshold_full: float,
    danger_k: int,
) -> float:
    best_beta = float(BETA_GRID[0])
    best_reward = -np.inf
    best_ttf = -np.inf

    for beta in BETA_GRID:
        factories, _ = _policy_factories(
            env_cfg=env_cfg,
            steps=horizon,
            budget=float(budget),
            threshold_partial=float(threshold_partial),
            threshold_full=float(threshold_full),
            beta=float(beta),
            danger_k=int(danger_k),
        )
        run = _run_policy(
            policy_factory=factories["conservative_bandit"],
            env_cfg=env_cfg,
            horizon=horizon,
            seeds=seeds,
            budget=float(budget),
        )
        reward_mean = float(np.mean(run.cumulative_reward))
        ttf_median = float(np.median(run.time_to_failure))

        if (reward_mean > best_reward) or (
            np.isclose(reward_mean, best_reward) and ttf_median > best_ttf
        ):
            best_beta = float(beta)
            best_reward = reward_mean
            best_ttf = ttf_median

    return float(best_beta)


def _evaluate_policy_set(
    env_cfg: HWGroundedEnvConfig,
    horizon: int,
    seeds: np.ndarray,
    budget: float,
    beta: float,
    threshold_partial: float,
    threshold_full: float,
    danger_k: int,
    policy_keys: Sequence[str],
) -> Dict[str, PolicyRun]:
    factories, _ = _policy_factories(
        env_cfg=env_cfg,
        steps=horizon,
        budget=float(budget),
        threshold_partial=float(threshold_partial),
        threshold_full=float(threshold_full),
        beta=float(beta),
        danger_k=int(danger_k),
    )
    out: Dict[str, PolicyRun] = {}
    for key in policy_keys:
        out[key] = _run_policy(
            policy_factory=factories[key],
            env_cfg=env_cfg,
            horizon=horizon,
            seeds=seeds,
            budget=float(budget),
        )
    return out


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reproducible failure-case artifacts.")
    parser.add_argument("--outdir", default="results/failure_cases")
    parser.add_argument("--csv", default="data/hardware/hardware_timeseries.csv")
    parser.add_argument("--fitted-json", default="results/fitted_drift_params.json")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--budget", type=float, default=100.0)
    parser.add_argument("--low-budget", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--shock-start", type=int, default=100)
    parser.add_argument("--shock-duration", type=int, default=15)
    parser.add_argument("--shock-profile", type=str, default="coherent_burst_short")
    parser.add_argument("--probe-shots", type=int, default=1500)
    parser.add_argument("--probe-cost", type=float, default=2.0)
    parser.add_argument("--partial-cost", type=float, default=4.0)
    parser.add_argument("--full-cost", type=float, default=10.0)
    parser.add_argument("--lambda-cost", type=float, default=0.1)
    parser.add_argument("--failure-threshold", type=float, default=0.8)
    parser.add_argument("--failure-consecutive", type=int, default=3)
    parser.add_argument("--max-probes", type=int, default=40)
    parser.add_argument("--threshold-partial", type=float, default=0.03)
    parser.add_argument("--threshold-full", type=float, default=0.08)
    parser.add_argument("--danger-k", type=int, default=5)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fitted_json = Path(args.fitted_json)
    csv_path = Path(args.csv)
    params = _load_or_fit_params(
        fitted_json=fitted_json,
        csv_path=csv_path,
        fallback_out=outdir / "fitted_drift_params.json",
    )

    weights = {"z": 1.0, "o": 1.0, "c": 1.0, "x": 1.0}
    base_env_cfg = _build_env_config(
        fitted_params=params,
        seed=int(args.seed),
        probe_shots=int(args.probe_shots),
        probe_cost=float(args.probe_cost),
        partial_cost=float(args.partial_cost),
        full_cost=float(args.full_cost),
        lambda_cost=float(args.lambda_cost),
        failure_threshold=float(args.failure_threshold),
        failure_consecutive=int(args.failure_consecutive),
        max_probes_per_horizon=int(args.max_probes),
        weights=weights,
    )

    shock_env_cfg = replace(
        base_env_cfg,
        shock_start=int(args.shock_start),
        shock_duration=int(args.shock_duration),
        shock_profile=str(args.shock_profile),
    )

    rng = np.random.default_rng(int(args.seed))
    seeds = rng.integers(low=1, high=2**31 - 1, size=int(args.episodes), dtype=np.int64)

    conservative_best_beta = _select_best_beta(
        env_cfg=base_env_cfg,
        horizon=int(args.horizon),
        seeds=seeds,
        budget=float(args.budget),
        threshold_partial=float(args.threshold_partial),
        threshold_full=float(args.threshold_full),
        danger_k=int(args.danger_k),
    )
    collapse_threshold = float(args.failure_threshold)

    summary_rows: list[Dict[str, object]] = []

    # 1) Periodic collapse under shock.
    periodic_runs = _evaluate_policy_set(
        env_cfg=shock_env_cfg,
        horizon=int(args.horizon),
        seeds=seeds,
        budget=float(args.budget),
        beta=float(conservative_best_beta),
        threshold_partial=float(args.threshold_partial),
        threshold_full=float(args.threshold_full),
        danger_k=int(args.danger_k),
        policy_keys=POLICY_ORDER,
    )
    periodic_labels = {
        "periodic_full": "periodic_full",
        "periodic_partial": "periodic_partial",
        "conservative_bandit": f"conservative_best (beta={conservative_best_beta:.2f})",
    }
    _plot_traces(
        runs=periodic_runs,
        labels=periodic_labels,
        out_path=outdir / "failure_periodic_shock.png",
        title="Failure Case: Periodic Collapse Under Regime Shock",
        horizon=int(args.horizon),
        shock_start=int(args.shock_start),
        shock_duration=int(args.shock_duration),
    )
    for key in POLICY_ORDER:
        m = _metrics(
            periodic_runs[key],
            horizon=int(args.horizon),
            collapse_threshold=collapse_threshold,
        )
        summary_rows.append(
            {
                "scenario": "periodic_collapse_under_shock",
                "policy": "conservative_best" if key == "conservative_bandit" else key,
                "beta": float(conservative_best_beta) if key == "conservative_bandit" else "",
                "budget": float(args.budget),
                **m,
            }
        )

    # 2) Low-beta instability.
    low_beta = 0.01
    low_beta_runs = _evaluate_policy_set(
        env_cfg=shock_env_cfg,
        horizon=int(args.horizon),
        seeds=seeds,
        budget=float(args.budget),
        beta=float(low_beta),
        threshold_partial=float(args.threshold_partial),
        threshold_full=float(args.threshold_full),
        danger_k=int(args.danger_k),
        policy_keys=("conservative_bandit",),
    )
    low_plot_runs = {
        "low_beta": low_beta_runs["conservative_bandit"],
        "periodic_partial_ref": periodic_runs["periodic_partial"],
    }
    _plot_traces(
        runs=low_plot_runs,
        labels={
            "low_beta": "conservative beta=0.01",
            "periodic_partial_ref": "periodic_partial (reference)",
        },
        out_path=outdir / "failure_low_beta.png",
        title="Failure Case: Low-Beta Instability (vs periodic_partial)",
        horizon=int(args.horizon),
        shock_start=int(args.shock_start),
        shock_duration=int(args.shock_duration),
    )
    low_m = _metrics(
        low_beta_runs["conservative_bandit"],
        horizon=int(args.horizon),
        collapse_threshold=collapse_threshold,
    )
    low_ref_m = _metrics(
        periodic_runs["periodic_partial"],
        horizon=int(args.horizon),
        collapse_threshold=collapse_threshold,
    )
    summary_rows.append(
        {
            "scenario": "low_beta_instability",
            "policy": "conservative_bandit",
            "beta": float(low_beta),
            "budget": float(args.budget),
            **low_m,
        }
    )
    summary_rows.append(
        {
            "scenario": "low_beta_instability_reference",
            "policy": "periodic_partial",
            "beta": "",
            "budget": float(args.budget),
            **low_ref_m,
        }
    )

    # 3) Over-conservative stagnation.
    high_beta = 1.0
    high_beta_runs = _evaluate_policy_set(
        env_cfg=shock_env_cfg,
        horizon=int(args.horizon),
        seeds=seeds,
        budget=float(args.budget),
        beta=float(high_beta),
        threshold_partial=float(args.threshold_partial),
        threshold_full=float(args.threshold_full),
        danger_k=int(args.danger_k),
        policy_keys=("conservative_bandit",),
    )
    high_plot_runs = {
        "high_beta": high_beta_runs["conservative_bandit"],
        "periodic_partial_ref": periodic_runs["periodic_partial"],
    }
    _plot_traces(
        runs=high_plot_runs,
        labels={
            "high_beta": "conservative beta=1.00",
            "periodic_partial_ref": "periodic_partial (reference)",
        },
        out_path=outdir / "failure_high_beta.png",
        title="Failure Case: Over-Conservative Stagnation (vs periodic_partial)",
        horizon=int(args.horizon),
        shock_start=int(args.shock_start),
        shock_duration=int(args.shock_duration),
    )
    high_m = _metrics(
        high_beta_runs["conservative_bandit"],
        horizon=int(args.horizon),
        collapse_threshold=collapse_threshold,
    )
    high_ref_m = _metrics(
        periodic_runs["periodic_partial"],
        horizon=int(args.horizon),
        collapse_threshold=collapse_threshold,
    )
    summary_rows.append(
        {
            "scenario": "over_conservative_stagnation",
            "policy": "conservative_bandit",
            "beta": float(high_beta),
            "budget": float(args.budget),
            **high_m,
        }
    )
    summary_rows.append(
        {
            "scenario": "over_conservative_stagnation_reference",
            "policy": "periodic_partial",
            "beta": "",
            "budget": float(args.budget),
            **high_ref_m,
        }
    )

    # 4) Low-budget breakdown.
    low_budget_runs = _evaluate_policy_set(
        env_cfg=shock_env_cfg,
        horizon=int(args.horizon),
        seeds=seeds,
        budget=float(args.low_budget),
        beta=float(conservative_best_beta),
        threshold_partial=float(args.threshold_partial),
        threshold_full=float(args.threshold_full),
        danger_k=int(args.danger_k),
        policy_keys=POLICY_ORDER,
    )
    _plot_traces(
        runs=low_budget_runs,
        labels=periodic_labels,
        out_path=outdir / "failure_low_budget.png",
        title=f"Failure Case: Low-Budget Breakdown (budget={float(args.low_budget):.1f})",
        horizon=int(args.horizon),
        shock_start=int(args.shock_start),
        shock_duration=int(args.shock_duration),
    )

    low_budget_table_rows: list[Dict[str, object]] = []
    for key in POLICY_ORDER:
        m = _metrics(
            low_budget_runs[key],
            horizon=int(args.horizon),
            collapse_threshold=collapse_threshold,
        )
        policy_name = "conservative_best" if key == "conservative_bandit" else key
        low_budget_table_rows.append(
            {
                "policy": policy_name,
                "collapse_rate": m["collapse_rate"],
                "median_time_to_failure": m["median_time_to_failure"],
                "cumulative_reward_mean": m["cumulative_reward_mean"],
            }
        )
        summary_rows.append(
            {
                "scenario": "low_budget_breakdown",
                "policy": policy_name,
                "beta": float(conservative_best_beta) if key == "conservative_bandit" else "",
                "budget": float(args.low_budget),
                **m,
            }
        )

    _write_csv(outdir / "collapse_rate_table.csv", low_budget_table_rows)
    _write_csv(outdir / "failure_summary.csv", summary_rows)

    print(f"[done] conservative_best_beta={conservative_best_beta:.2f}")
    print("[done] saved:")
    print(f"  - {outdir / 'failure_periodic_shock.png'}")
    print(f"  - {outdir / 'failure_low_beta.png'}")
    print(f"  - {outdir / 'failure_high_beta.png'}")
    print(f"  - {outdir / 'failure_low_budget.png'}")
    print(f"  - {outdir / 'collapse_rate_table.csv'}")
    print(f"  - {outdir / 'failure_summary.csv'}")


if __name__ == "__main__":
    main()
