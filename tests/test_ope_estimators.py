import unittest

import numpy as np

from calib_sched.eval.ope import (
    LinearRewardModel,
    dr_contributions,
    ips_contributions,
)
from calib_sched.policies.baselines import BasePolicy


class AlwaysActionPolicy(BasePolicy):
    def __init__(self, action: int) -> None:
        self.action = action

    def select_action(self, context):  # type: ignore[override]
        return self.action

    def action_distribution(self, context):  # type: ignore[override]
        probs = np.zeros(4, dtype=float)
        probs[self.action] = 1.0
        return probs


class TestOpeEstimators(unittest.TestCase):
    def test_ips_and_dr_toy_case(self) -> None:
        records = [
            {"action": 1, "reward": 1.0, "propensity": 0.5, "ctx_x": 0.0},
            {"action": 1, "reward": 1.0, "propensity": 0.5, "ctx_x": 1.0},
            {"action": 0, "reward": 0.0, "propensity": 0.5, "ctx_x": 0.0},
            {"action": 0, "reward": 0.0, "propensity": 0.5, "ctx_x": 1.0},
        ]

        policy = AlwaysActionPolicy(action=1)

        ips = ips_contributions(records, policy)
        self.assertAlmostEqual(float(np.mean(ips)), 1.0, places=6)

        model = LinearRewardModel(feature_keys=["x"], ridge=1e-6)
        model.fit(records)
        dr = dr_contributions(records, policy, model)
        self.assertAlmostEqual(float(np.mean(dr)), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
