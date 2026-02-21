import unittest
from unittest import mock

import calib_sched.braket_runs.parse as parse_mod
from calib_sched.braket_runs.parse import jeffreys_interval, wilson_interval


class TestIntervals(unittest.TestCase):
    def test_wilson_known_case(self) -> None:
        lo, hi = wilson_interval(k=5, n=10, alpha=0.05)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)

    def test_wilson_empty(self) -> None:
        lo, hi = wilson_interval(k=0, n=0, alpha=0.05)
        self.assertEqual(lo, 0.0)
        self.assertEqual(hi, 1.0)

    def test_jeffreys_bounds(self) -> None:
        if parse_mod.SCIPY_AVAILABLE:
            lo, hi = jeffreys_interval(k=1, n=10, alpha=0.05)
            self.assertAlmostEqual(lo, 0.0110116738, places=6)
            self.assertAlmostEqual(hi, 0.3813147711, places=6)
        else:
            with self.assertRaises(RuntimeError):
                jeffreys_interval(k=1, n=10, alpha=0.05)

    def test_wilson_fallback_alpha_guard(self) -> None:
        # Simulate SciPy-unavailable mode and assert only whitelisted alphas work.
        with mock.patch.object(parse_mod, "SCIPY_AVAILABLE", False):
            lo, hi = wilson_interval(k=5, n=10, alpha=0.05)
            self.assertLess(lo, hi)
            with self.assertRaises(ValueError):
                wilson_interval(k=5, n=10, alpha=0.02)


if __name__ == "__main__":
    unittest.main()
