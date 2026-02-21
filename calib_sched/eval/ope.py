from __future__ import annotations

import csv
import json
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from ..policies.baselines import BasePolicy
from ..sim.env import NUM_ACTIONS
from .metrics import empirical_bernstein_lcb


@dataclass
class OpeEstimate:
    ips_mean: float
    ips_lcb: float
    dr_mean: float
    dr_lcb: float


class LinearRewardModel:
    def __init__(self, feature_keys: Sequence[str], ridge: float = 1.0) -> None:
        self.feature_keys = list(feature_keys)
        self.ridge = float(ridge)
        self._A: List[np.ndarray] = []
        self._A_inv: List[np.ndarray] = []
        self._b: List[np.ndarray] = []
        self._theta: List[np.ndarray] = []
        self._init_params()

    def _init_params(self) -> None:
        d = 1 + len(self.feature_keys)
        self._A = [self.ridge * np.eye(d, dtype=float) for _ in range(NUM_ACTIONS)]
        self._A_inv = [np.linalg.inv(a) for a in self._A]
        self._b = [np.zeros(d, dtype=float) for _ in range(NUM_ACTIONS)]
        self._theta = [np.zeros(d, dtype=float) for _ in range(NUM_ACTIONS)]

    def _x(self, context: Mapping[str, float]) -> np.ndarray:
        return np.asarray([1.0] + [float(context.get(k, 0.0)) for k in self.feature_keys], dtype=float)

    def fit(self, records: Sequence[Mapping[str, object]], reward_key: str = "reward") -> None:
        d = 1 + len(self.feature_keys)
        self._init_params()

        for action in range(NUM_ACTIONS):
            rows = []
            ys = []
            for rec in records:
                if int(float(rec["action"])) != action:
                    continue
                context = extract_context(rec)
                rows.append(self._x(context))
                ys.append(float(rec[reward_key]))

            A = self.ridge * np.eye(d, dtype=float)
            b = np.zeros(d, dtype=float)
            if rows:
                X = np.vstack(rows)
                y = np.asarray(ys, dtype=float)
                A = A + X.T @ X
                b = b + X.T @ y

            A_inv = np.linalg.pinv(A)
            theta = A_inv @ b
            self._A[action] = A
            self._A_inv[action] = A_inv
            self._b[action] = b
            self._theta[action] = theta

    def predict(self, context: Mapping[str, float], action: int) -> float:
        x = self._x(context)
        return float(np.dot(self._theta[action], x))


def _coerce(value: object) -> object:
    if value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s == "":
            return s
        try:
            if "." in s or "e" in s.lower():
                return float(s)
            return int(s)
        except ValueError:
            return s
    return value


def load_logged_data(path: str) -> List[Dict[str, object]]:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    suffix = p.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(p)
        data = table.to_pydict()
        n = len(next(iter(data.values()))) if data else 0
        rows: List[Dict[str, object]] = []
        for i in range(n):
            rec = {k: _coerce(v[i]) for k, v in data.items()}
            rows.append(rec)
        return rows

    if suffix == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append({k: _coerce(v) for k, v in json.loads(line).items()})
        return rows

    rows = []
    with p.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({k: _coerce(v) for k, v in row.items()})
    return rows


def extract_context(record: Mapping[str, object]) -> Dict[str, float]:
    context: Dict[str, float] = {}
    for k, v in record.items():
        if not k.startswith("ctx_"):
            continue
        try:
            context[k[4:]] = float(v)  # type: ignore[arg-type]
        except Exception:
            continue
    if "timestep" not in context and "step" in record:
        context["timestep"] = float(record["step"])  # type: ignore[arg-type]
    return context


def infer_feature_keys(records: Sequence[Mapping[str, object]]) -> List[str]:
    keys = set()
    for rec in records:
        for k in rec.keys():
            if k.startswith("ctx_"):
                keys.add(k[4:])
    ordered = sorted(k for k in keys if k != "timestep")
    if "timestep" in keys:
        ordered.append("timestep")
    return ordered


def ips_contributions(
    records: Sequence[Mapping[str, object]],
    target_policy: BasePolicy,
    clip_weight: float | None = None,
) -> np.ndarray:
    vals = []
    for rec in records:
        context = extract_context(rec)
        action = int(float(rec["action"]))
        reward = float(rec["reward"])
        propensity = max(1e-12, float(rec["propensity"]))

        pi = target_policy.action_distribution(context)
        weight = float(pi[action]) / propensity
        if clip_weight is not None:
            weight = min(clip_weight, weight)
        vals.append(weight * reward)
    return np.asarray(vals, dtype=float)


def dr_contributions(
    records: Sequence[Mapping[str, object]],
    target_policy: BasePolicy,
    reward_model: LinearRewardModel,
    clip_weight: float | None = None,
) -> np.ndarray:
    vals = []
    for rec in records:
        context = extract_context(rec)
        action = int(float(rec["action"]))
        reward = float(rec["reward"])
        propensity = max(1e-12, float(rec["propensity"]))

        pi = target_policy.action_distribution(context)
        q = np.asarray([reward_model.predict(context, a) for a in range(NUM_ACTIONS)], dtype=float)

        q_pi = float(np.dot(pi, q))
        correction = float(pi[action]) / propensity
        if clip_weight is not None:
            correction = min(clip_weight, correction)
        vals.append(q_pi + correction * (reward - q[action]))
    return np.asarray(vals, dtype=float)


def evaluate_policy_ope(
    records: Sequence[Mapping[str, object]],
    policy: BasePolicy,
    reward_model: LinearRewardModel,
    delta: float,
    clip_weight: float | None = None,
) -> OpeEstimate:
    ips_vals = ips_contributions(records, policy, clip_weight=clip_weight)
    dr_vals = dr_contributions(records, policy, reward_model, clip_weight=clip_weight)

    ips_mean, ips_lcb, _ = empirical_bernstein_lcb(ips_vals.tolist(), delta=delta)
    dr_mean, dr_lcb, _ = empirical_bernstein_lcb(dr_vals.tolist(), delta=delta)

    return OpeEstimate(
        ips_mean=float(ips_mean),
        ips_lcb=float(ips_lcb),
        dr_mean=float(dr_mean),
        dr_lcb=float(dr_lcb),
    )


def compare_candidate_vs_baseline(
    records: Sequence[Mapping[str, object]],
    candidate_policy: BasePolicy,
    baseline_policy: BasePolicy,
    reward_model: LinearRewardModel,
    delta: float,
    clip_weight: float | None = None,
) -> Dict[str, object]:
    candidate = evaluate_policy_ope(
        records=records,
        policy=candidate_policy,
        reward_model=reward_model,
        delta=delta,
        clip_weight=clip_weight,
    )
    baseline = evaluate_policy_ope(
        records=records,
        policy=baseline_policy,
        reward_model=reward_model,
        delta=delta,
        clip_weight=clip_weight,
    )

    accept = bool(candidate.dr_lcb > baseline.dr_lcb)
    return {
        "candidate": candidate.__dict__,
        "baseline": baseline.__dict__,
        "accept_candidate": accept,
        "rule": "accept iff DR_LCB(candidate) > DR_LCB(baseline)",
    }
