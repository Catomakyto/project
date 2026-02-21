from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class DriftParams:
    readout_sigma: float = 4e-4
    coherent_kappa: float = 0.03
    coherent_sigma: float = 2e-4
    crosstalk_rho: float = 0.96
    crosstalk_sigma: float = 2e-4
    p_burst_up: float = 0.01
    p_burst_down: float = 0.18
    burst_amplitude: float = 0.004
    residual_error: float = 1e-4


class HiddenState:
    """Hidden drift state with readout, coherent, and crosstalk processes."""

    def __init__(
        self,
        num_qubits: int,
        crosstalk_pairs: Sequence[Tuple[int, int]],
        drift: DriftParams,
        rng: np.random.Generator,
    ) -> None:
        self.num_qubits = int(num_qubits)
        self.crosstalk_pairs = [tuple(sorted((int(a), int(b)))) for a, b in crosstalk_pairs]
        self.drift = drift
        self.rng = rng

        self.readout_p10 = np.full(self.num_qubits, 1e-3, dtype=float)
        self.readout_p01 = np.full(self.num_qubits, 1e-3, dtype=float)
        self.coherent_delta = np.zeros(self.num_qubits, dtype=float)

        self.crosstalk_amp: Dict[Tuple[int, int], float] = {
            pair: 0.0 for pair in self.crosstalk_pairs
        }
        self.crosstalk_burst: Dict[Tuple[int, int], bool] = {
            pair: False for pair in self.crosstalk_pairs
        }

    def update(self) -> None:
        self.readout_p10 += self.rng.normal(0.0, self.drift.readout_sigma, size=self.num_qubits)
        self.readout_p01 += self.rng.normal(0.0, self.drift.readout_sigma, size=self.num_qubits)
        np.clip(self.readout_p10, 0.0, 0.08, out=self.readout_p10)
        np.clip(self.readout_p01, 0.0, 0.08, out=self.readout_p01)

        noise = self.rng.normal(0.0, self.drift.coherent_sigma, size=self.num_qubits)
        self.coherent_delta = (1.0 - self.drift.coherent_kappa) * self.coherent_delta + noise

        for pair in self.crosstalk_pairs:
            in_burst = self.crosstalk_burst[pair]
            if in_burst and self.rng.random() < self.drift.p_burst_down:
                self.crosstalk_burst[pair] = False
            elif (not in_burst) and self.rng.random() < self.drift.p_burst_up:
                self.crosstalk_burst[pair] = True

            amp = self.crosstalk_amp[pair]
            amp = self.drift.crosstalk_rho * amp + self.rng.normal(0.0, self.drift.crosstalk_sigma)
            if self.crosstalk_burst[pair]:
                amp += self.drift.burst_amplitude
            self.crosstalk_amp[pair] = float(amp)

    def calibrate_partial(self) -> None:
        self.readout_p10 = np.abs(self.rng.normal(0.0, self.drift.residual_error, size=self.num_qubits))
        self.readout_p01 = np.abs(self.rng.normal(0.0, self.drift.residual_error, size=self.num_qubits))
        self.coherent_delta = self.rng.normal(0.0, self.drift.residual_error, size=self.num_qubits)

    def calibrate_full(self) -> None:
        self.calibrate_partial()
        for pair in self.crosstalk_pairs:
            self.crosstalk_amp[pair] = float(self.rng.normal(0.0, self.drift.residual_error))
            self.crosstalk_burst[pair] = False

    def snapshot(self) -> Dict[str, object]:
        return {
            "readout_p10": self.readout_p10.copy(),
            "readout_p01": self.readout_p01.copy(),
            "coherent_delta": self.coherent_delta.copy(),
            "crosstalk_amp": dict(self.crosstalk_amp),
            "crosstalk_burst": dict(self.crosstalk_burst),
        }
