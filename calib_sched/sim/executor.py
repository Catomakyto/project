from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from .hidden_state import HiddenState


def _binomial(rng: np.random.Generator, n: int, p: float) -> int:
    p_clamped = min(1.0, max(0.0, p))
    return int(rng.binomial(int(n), p_clamped))


def compute_sentinel_stats(
    state: HiddenState,
    monitor_qubits: Sequence[int],
    monitor_pair: Tuple[int, int],
    shots: int,
    coherent_repeats: Sequence[int],
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Simulate sentinel outcomes used as context features."""
    qubits = [int(q) for q in monitor_qubits]
    pair = tuple(sorted((int(monitor_pair[0]), int(monitor_pair[1]))))

    ro0 = []
    ro1 = []
    for q in qubits:
        ro0.append(_binomial(rng, shots, float(state.readout_p10[q])) / shots)
        ro1.append(_binomial(rng, shots, float(state.readout_p01[q])) / shots)

    coherent_values = []
    for repeats in coherent_repeats:
        q = qubits[0]
        theta = repeats * (math.pi / 4.0 + float(state.coherent_delta[q]) / 2.0)
        p_one = math.sin(theta) ** 2
        p_hat = _binomial(rng, shots, p_one) / shots
        expected = math.sin(repeats * math.pi / 4.0) ** 2
        coherent_values.append(abs(p_hat - expected))

    c_amp = abs(float(state.crosstalk_amp.get(pair, 0.0)))
    odd_parity_true = min(1.0, max(0.0, math.tanh(40.0 * c_amp)))
    odd_parity_hat = _binomial(rng, shots, odd_parity_true) / shots

    return {
        "readout_mean_error_zero": float(np.mean(ro0) if ro0 else 0.0),
        "readout_mean_error_one": float(np.mean(ro1) if ro1 else 0.0),
        "coherent_mean_anomaly": float(np.mean(coherent_values) if coherent_values else 0.0),
        "crosstalk_mean_odd_parity": float(odd_parity_hat),
    }


def simulate_workload_performance(
    state: HiddenState,
    workload: Mapping[str, object],
    sensitivity: Mapping[str, float],
    rng: np.random.Generator,
) -> float:
    """Synthetic workload performance proxy in [0, 1]."""
    qubits = [int(q) for q in workload["qubits"]]  # type: ignore[index]
    depth = int(workload["depth"])
    two_qubit_count = int(workload["two_qubit_count"])

    readout_sum = float(np.sum(state.readout_p10[qubits] + state.readout_p01[qubits]))
    coherent_sum = float(np.sum(np.abs(state.coherent_delta[qubits])))

    crosstalk_sum = 0.0
    qset = set(qubits)
    for (a, b), amp in state.crosstalk_amp.items():
        if a in qset and b in qset:
            crosstalk_sum += abs(float(amp))

    k_readout = float(sensitivity.get("readout", 45.0))
    k_coherent = float(sensitivity.get("coherent", 120.0))
    k_crosstalk = float(sensitivity.get("crosstalk", 220.0))

    exponent = -(
        k_readout * readout_sum * depth
        + k_coherent * coherent_sum * depth
        + k_crosstalk * crosstalk_sum * max(1, two_qubit_count)
    )
    exponent = max(-60.0, min(0.0, exponent))

    base = math.exp(exponent)
    noisy = base + rng.normal(0.0, 0.005)
    return float(min(1.0, max(0.0, noisy)))
