from __future__ import annotations

from dataclasses import dataclass, asdict
from math import pi
from typing import Any, Dict, List, Sequence, Tuple

try:
    from braket.circuits import Circuit
except ImportError:
    class Circuit:  # pragma: no cover - local fallback when Braket SDK is unavailable
        def __init__(self) -> None:
            self.instructions: List[Tuple[Any, ...]] = []

        def x(self, target: int) -> "Circuit":
            self.instructions.append(("x", target))
            return self

        def rx(self, target: int, angle: float) -> "Circuit":
            self.instructions.append(("rx", target, angle))
            return self

        def cnot(self, control: int, target: int) -> "Circuit":
            self.instructions.append(("cnot", control, target))
            return self

        def measure(self, *qubits: int) -> "Circuit":
            self.instructions.append(("measure", tuple(qubits)))
            return self


BIT_ORDER = "measured_qubits_left_to_right"


@dataclass
class CircuitDescriptor:
    circuit_index: int
    name: str
    circuit_type: str
    shots: int
    qubits: List[int]
    pair: List[int]
    repeats: int
    measured_qubits: List[int]
    bit_order: str = BIT_ORDER

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _validate_monitoring_inputs(monitor_qubits: Sequence[int], monitor_pair: Sequence[int]) -> None:
    if len(monitor_qubits) < 2:
        raise ValueError("monitor_qubits must include at least two qubits")
    if len(monitor_pair) != 2:
        raise ValueError("monitor_pair must contain exactly two qubit indices")


def build_default_sentinel_suite(
    monitor_qubits: Sequence[int],
    monitor_pair: Sequence[int],
    shots_per_circuit: int,
    coherent_repeats: Sequence[int],
) -> Tuple[List[Circuit], List[Dict[str, Any]]]:
    """Build the 5-circuit sentinel suite for drift logging.

    Suite order and descriptors are explicit and deterministic:
    1) RO0  : readout_zero on monitor_qubits
    2) RO1  : readout_one on monitor_qubits
    3) RX4  : coherent_rx on monitor_qubits[0], repeats=4
    4) RX8  : coherent_rx on monitor_qubits[0], repeats=8
    5) XT   : crosstalk parity probe on monitor_pair
    """
    _validate_monitoring_inputs(monitor_qubits, monitor_pair)

    if len(coherent_repeats) != 2:
        raise ValueError("coherent_repeats must contain exactly two repeat counts")

    q0 = int(monitor_qubits[0])
    qubits = [int(q) for q in monitor_qubits]
    pair = [int(monitor_pair[0]), int(monitor_pair[1])]

    circuits: List[Circuit] = []
    descriptors: List[Dict[str, Any]] = []

    # 1) readout_zero on monitored qubits
    ro0 = Circuit()
    for q in qubits:
        ro0.measure(q)
    circuits.append(ro0)
    descriptors.append(
        CircuitDescriptor(
            circuit_index=0,
            name="RO0",
            circuit_type="readout_zero",
            shots=int(shots_per_circuit),
            qubits=qubits,
            pair=[],
            repeats=0,
            measured_qubits=qubits,
        ).to_dict()
    )

    # 2) readout_one on monitored qubits
    ro1 = Circuit()
    for q in qubits:
        ro1.x(q)
    for q in qubits:
        ro1.measure(q)
    circuits.append(ro1)
    descriptors.append(
        CircuitDescriptor(
            circuit_index=1,
            name="RO1",
            circuit_type="readout_one",
            shots=int(shots_per_circuit),
            qubits=qubits,
            pair=[],
            repeats=0,
            measured_qubits=qubits,
        ).to_dict()
    )

    # 3) coherent Rx sentinel, repeats=coherent_repeats[0]
    r4 = int(coherent_repeats[0])
    rx4 = Circuit()
    for _ in range(r4):
        rx4.rx(q0, pi / 2.0)
    rx4.measure(q0)
    circuits.append(rx4)
    descriptors.append(
        CircuitDescriptor(
            circuit_index=2,
            name=f"RX{r4}_q{q0}",
            circuit_type="coherent_rx",
            shots=int(shots_per_circuit),
            qubits=[q0],
            pair=[],
            repeats=r4,
            measured_qubits=[q0],
        ).to_dict()
    )

    # 4) coherent Rx sentinel, repeats=coherent_repeats[1]
    r8 = int(coherent_repeats[1])
    rx8 = Circuit()
    for _ in range(r8):
        rx8.rx(q0, pi / 2.0)
    rx8.measure(q0)
    circuits.append(rx8)
    descriptors.append(
        CircuitDescriptor(
            circuit_index=3,
            name=f"RX{r8}_q{q0}",
            circuit_type="coherent_rx",
            shots=int(shots_per_circuit),
            qubits=[q0],
            pair=[],
            repeats=r8,
            measured_qubits=[q0],
        ).to_dict()
    )

    # 5) shallow non-trivial crosstalk parity probe
    xt = Circuit()
    xt.rx(pair[0], pi / 2.0)
    xt.rx(pair[1], pi / 2.0)
    xt.cnot(pair[0], pair[1])
    xt.rx(pair[0], -pi / 2.0)
    xt.rx(pair[1], -pi / 2.0)
    for p in pair:
        xt.measure(p)
    circuits.append(xt)
    descriptors.append(
        CircuitDescriptor(
            circuit_index=4,
            name=f"XT_{pair[0]}{pair[1]}",
            circuit_type="crosstalk_pair",
            shots=int(shots_per_circuit),
            qubits=[],
            pair=pair,
            repeats=0,
            measured_qubits=pair,
        ).to_dict()
    )

    return circuits, descriptors


def scale_descriptor_shots(descriptors: Sequence[Dict[str, Any]], shots_scale: float) -> List[Dict[str, Any]]:
    if shots_scale <= 0:
        raise ValueError("shots_scale must be > 0")
    scaled: List[Dict[str, Any]] = []
    for desc in descriptors:
        updated = dict(desc)
        original = int(updated["shots"])
        updated["shots"] = max(1, int(round(original * shots_scale)))
        scaled.append(updated)
    return scaled
