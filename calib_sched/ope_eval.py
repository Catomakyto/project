from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict, List

import yaml

from .eval.ope import (
    LinearRewardModel,
    compare_candidate_vs_baseline,
    extract_context,
    infer_feature_keys,
    load_logged_data,
)
from .policies.baselines import PeriodicPolicy, ThresholdPolicy
from .policies.conservative_bandit import ConservativeBanditPolicy


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _build_candidate(policy_name: str, cfg: Dict[str, Any], feature_keys: List[str], records: List[Dict[str, object]]):
    sim = cfg["simulation"]
    baseline_period = int(sim["baseline"]["period"])

    if policy_name == "conservative_bandit":
        p = sim["conservative_bandit"]
        policy = ConservativeBanditPolicy(
            ridge=float(p["ridge"]),
            beta=float(p["beta"]),
            uncertainty_threshold=float(p["uncertainty_threshold"]),
            risk_threshold=float(p["risk_threshold"]),
            improvement_margin=float(p["improvement_margin"]),
            exploration_epsilon=0.0,
            safe_period=baseline_period,
            feature_keys=feature_keys,
        )
        contexts = [extract_context(r) for r in records]
        actions = [int(float(r["action"])) for r in records]
        rewards = [float(r["reward"]) for r in records]
        policy.fit_batch(contexts=contexts, actions=actions, rewards=rewards)
        return policy

    if policy_name == "periodic":
        return PeriodicPolicy(period=baseline_period)

    if policy_name == "threshold":
        return ThresholdPolicy(threshold=float(sim["baseline"]["threshold"]))

    raise ValueError(f"Unsupported policy '{policy_name}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Off-policy evaluation (IPS + DR + LCB) for calibration policies.")
    parser.add_argument("--log", required=True, help="Logged bandit data (.parquet/.csv/.jsonl)")
    parser.add_argument("--policy", default="conservative_bandit")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="results/ope/ope_report.json")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    records = load_logged_data(args.log)
    if not records:
        raise RuntimeError("No records found in logged data")

    feature_keys = infer_feature_keys(records)

    ope_cfg = cfg["ope"]
    reward_model = LinearRewardModel(
        feature_keys=feature_keys,
        ridge=float(ope_cfg["reward_model_ridge"]),
    )
    reward_model.fit(records)

    baseline_policy = PeriodicPolicy(period=int(cfg["simulation"]["baseline"]["period"]))
    candidate_policy = _build_candidate(args.policy, cfg, feature_keys, records)

    report = compare_candidate_vs_baseline(
        records=records,
        candidate_policy=candidate_policy,
        baseline_policy=baseline_policy,
        reward_model=reward_model,
        delta=float(ope_cfg["delta"]),
        clip_weight=float(ope_cfg["clip_weight"]),
    )

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote OPE report: {out_path}")
    print(f"Candidate DR mean={report['candidate']['dr_mean']:.6f}, LCB={report['candidate']['dr_lcb']:.6f}")
    print(f"Baseline  DR mean={report['baseline']['dr_mean']:.6f}, LCB={report['baseline']['dr_lcb']:.6f}")
    print(f"Accept candidate: {report['accept_candidate']}")


if __name__ == "__main__":
    main()
