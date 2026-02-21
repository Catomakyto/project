import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import calib_sched.braket_runs.parse as parse_mod


class _FakeResult:
    def __init__(self, counts):
        self.measurement_counts = counts


class _FakeTask:
    _counts_by_arn = {
        "arn_ro0": {"00": 90, "01": 5, "10": 4, "11": 1},
        "arn_ro1": {"11": 88, "10": 5, "01": 4, "00": 3},
        "arn_rx4": {"0": 52, "1": 48},
        "arn_rx8": {"0": 51, "1": 49},
        "arn_xt": {"00": 70, "01": 10, "10": 15, "11": 5},
    }

    def __init__(self, arn):
        self.arn = arn

    def result(self):
        return _FakeResult(self._counts_by_arn[self.arn])


class TestHardwareParsingDeterministic(unittest.TestCase):
    def test_parse_from_descriptors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            metadata_path = tmp / "meta.json"
            out_csv = tmp / "hardware_timeseries.csv"

            metadata = {
                "created_at": "2026-02-08T00:00:00+00:00",
                "device_arn": "arn:test",
                "region": "us-west-1",
                "dry_run": False,
                "batched": False,
                "task_arns": ["arn_ro0", "arn_ro1", "arn_rx4", "arn_rx8", "arn_xt"],
                "circuit_descriptors": [
                    {
                        "circuit_index": 0,
                        "name": "RO0",
                        "circuit_type": "readout_zero",
                        "shots": 100,
                        "qubits": [0, 1],
                        "pair": [],
                        "repeats": 0,
                        "measured_qubits": [0, 1],
                        "bit_order": "measured_qubits_left_to_right",
                        "task_arn": "arn_ro0",
                        "result_index": 0,
                    },
                    {
                        "circuit_index": 1,
                        "name": "RO1",
                        "circuit_type": "readout_one",
                        "shots": 100,
                        "qubits": [0, 1],
                        "pair": [],
                        "repeats": 0,
                        "measured_qubits": [0, 1],
                        "bit_order": "measured_qubits_left_to_right",
                        "task_arn": "arn_ro1",
                        "result_index": 0,
                    },
                    {
                        "circuit_index": 2,
                        "name": "RX4_q0",
                        "circuit_type": "coherent_rx",
                        "shots": 100,
                        "qubits": [0],
                        "pair": [],
                        "repeats": 4,
                        "measured_qubits": [0],
                        "bit_order": "measured_qubits_left_to_right",
                        "task_arn": "arn_rx4",
                        "result_index": 0,
                    },
                    {
                        "circuit_index": 3,
                        "name": "RX8_q0",
                        "circuit_type": "coherent_rx",
                        "shots": 100,
                        "qubits": [0],
                        "pair": [],
                        "repeats": 8,
                        "measured_qubits": [0],
                        "bit_order": "measured_qubits_left_to_right",
                        "task_arn": "arn_rx8",
                        "result_index": 0,
                    },
                    {
                        "circuit_index": 4,
                        "name": "XT_01",
                        "circuit_type": "crosstalk_pair",
                        "shots": 100,
                        "qubits": [],
                        "pair": [0, 1],
                        "repeats": 0,
                        "measured_qubits": [0, 1],
                        "bit_order": "measured_qubits_left_to_right",
                        "task_arn": "arn_xt",
                        "result_index": 0,
                    },
                ],
            }
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            with mock.patch.object(parse_mod, "AwsQuantumTask", _FakeTask):
                row = parse_mod.process_hardware_metadata(
                    metadata_path=str(metadata_path),
                    output_csv=str(out_csv),
                    interval_method="wilson",
                )

            self.assertAlmostEqual(row["readout_mean_error_zero"], 0.055, places=6)
            self.assertAlmostEqual(row["readout_mean_error_one"], 0.075, places=6)
            self.assertAlmostEqual(row["coherent_mean_anomaly"], 0.485, places=6)
            self.assertAlmostEqual(row["crosstalk_mean_odd_parity"], 0.25, places=6)
            self.assertTrue(out_csv.exists())
            self.assertGreater(len(out_csv.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
