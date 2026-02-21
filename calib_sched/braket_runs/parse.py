from __future__ import annotations

import csv
import json
import math
import pathlib
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

try:
    from scipy.stats import beta as _scipy_beta
    from scipy.stats import norm as _scipy_norm

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    _scipy_beta = None
    _scipy_norm = None
    SCIPY_AVAILABLE = False

try:
    from braket.aws import AwsQuantumTask
except ImportError:
    AwsQuantumTask = None  # type: ignore


_COMMON_Z = {
    0.05: 1.9599639845,
    0.01: 2.5758293035,
}


def _z_value(alpha: float) -> float:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")

    if SCIPY_AVAILABLE:
        assert _scipy_norm is not None
        return float(_scipy_norm.ppf(1.0 - alpha / 2.0))

    for known_alpha, known_z in _COMMON_Z.items():
        if abs(alpha - known_alpha) < 1e-12:
            return known_z
    supported = ", ".join(str(a) for a in sorted(_COMMON_Z.keys()))
    raise ValueError(
        f"SciPy unavailable; unsupported alpha={alpha}. Supported fallback alphas: {supported}"
    )


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 1.0)
    if k < 0 or k > n:
        raise ValueError("k must satisfy 0 <= k <= n")

    z = _z_value(alpha)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p + z2 / (2.0 * n)
    half = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
    lo = max(0.0, (center - half) / denom)
    hi = min(1.0, (center + half) / denom)
    return lo, hi


def jeffreys_interval(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Jeffreys interval using Beta(k+1/2, n-k+1/2) posterior quantiles."""
    if n <= 0:
        return (0.0, 1.0)
    if k < 0 or k > n:
        raise ValueError("k must satisfy 0 <= k <= n")
    if not SCIPY_AVAILABLE:
        raise RuntimeError(
            "Jeffreys interval requires SciPy (scipy.stats.beta.ppf). Install scipy or use --interval wilson."
        )

    assert _scipy_beta is not None
    a = k + 0.5
    b = n - k + 0.5
    lo = float(_scipy_beta.ppf(alpha / 2.0, a, b))
    hi = float(_scipy_beta.ppf(1.0 - alpha / 2.0, a, b))
    return lo, hi


def _load_metadata(metadata_path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _extract_counts_from_task_result(task_result: Any, result_index: int) -> Dict[str, int]:
    # Single-result shape
    if hasattr(task_result, "measurement_counts"):
        counts = getattr(task_result, "measurement_counts")
        return {str(k): int(v) for k, v in dict(counts).items()}

    # Batched shape (best effort)
    if hasattr(task_result, "results"):
        results = getattr(task_result, "results")
        if isinstance(results, Sequence) and len(results) > result_index:
            sub = results[result_index]
            if hasattr(sub, "measurement_counts"):
                counts = getattr(sub, "measurement_counts")
                return {str(k): int(v) for k, v in dict(counts).items()}
            if isinstance(sub, Mapping) and "measurement_counts" in sub:
                counts = sub["measurement_counts"]
                return {str(k): int(v) for k, v in dict(counts).items()}

    if isinstance(task_result, Mapping) and "measurement_counts" in task_result:
        counts = task_result["measurement_counts"]
        return {str(k): int(v) for k, v in dict(counts).items()}

    raise RuntimeError("Could not extract measurement counts from task result")


def _task_result_measured_qubits(task_result: Any) -> List[int]:
    for attr in ("measured_qubits", "measurement_qubits"):
        value = getattr(task_result, attr, None)
        if isinstance(value, Sequence) and value and all(isinstance(v, int) for v in value):
            return [int(v) for v in value]
    return []


def _resolve_measured_qubits(
    desc: Mapping[str, Any],
    task_result: Any,
    counts: Mapping[str, int],
) -> List[int]:
    descriptor_measured = [int(q) for q in desc.get("measured_qubits", [])]
    result_measured = _task_result_measured_qubits(task_result)

    measured_qubits = result_measured or descriptor_measured
    if not measured_qubits:
        if desc.get("qubits"):
            measured_qubits = [int(q) for q in desc["qubits"]]
        elif desc.get("pair"):
            measured_qubits = [int(q) for q in desc["pair"]]

    if not measured_qubits:
        raise ValueError("Unable to determine measured qubits from result metadata or descriptor")

    sample_key = next(iter(counts), "")
    if sample_key and len(sample_key) != len(measured_qubits):
        # Fallback to descriptor order when result metadata shape is inconsistent.
        if descriptor_measured and len(sample_key) == len(descriptor_measured):
            measured_qubits = descriptor_measured
        else:
            raise ValueError(
                "Bitstring length does not match measured qubits. "
                f"bitstring='{sample_key}', measured_qubits={measured_qubits}"
            )

    return measured_qubits


def _bit_for_qubit(bitstring: str, measured_qubits: Sequence[int], qubit: int, bit_order: str) -> int:
    if qubit not in measured_qubits:
        raise ValueError(f"Qubit {qubit} not present in measured_qubits={measured_qubits}")
    idx = measured_qubits.index(qubit)

    if bit_order == "measured_qubits_left_to_right":
        pos = idx
    elif bit_order == "measured_qubits_right_to_left":
        pos = len(measured_qubits) - 1 - idx
    else:
        raise ValueError(f"Unsupported bit_order '{bit_order}'")

    if pos < 0 or pos >= len(bitstring):
        raise ValueError(
            f"Bitstring '{bitstring}' incompatible with measured_qubits={measured_qubits} and bit_order={bit_order}"
        )
    return 1 if bitstring[pos] == "1" else 0


def _odd_parity_for_pair(
    bitstring: str,
    measured_qubits: Sequence[int],
    pair: Sequence[int],
    bit_order: str,
) -> int:
    if len(pair) != 2:
        raise ValueError(f"Crosstalk descriptor must provide exactly two qubits; got {pair}")
    b0 = _bit_for_qubit(bitstring, measured_qubits, int(pair[0]), bit_order)
    b1 = _bit_for_qubit(bitstring, measured_qubits, int(pair[1]), bit_order)
    return (b0 + b1) % 2


def _aggregate_mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _append_row_dynamic(csv_path: pathlib.Path, row: Mapping[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_row = {k: row[k] for k in sorted(row.keys())}

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(new_row.keys()))
            writer.writeheader()
            writer.writerow(new_row)
        return

    first_line = csv_path.read_text(encoding="utf-8").splitlines()[0].strip()
    if first_line.startswith("#"):
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(new_row.keys()))
            writer.writeheader()
            writer.writerow(new_row)
        return

    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        existing_rows = list(reader)
        existing_fields = list(reader.fieldnames or [])

    all_fields = list(dict.fromkeys(existing_fields + list(new_row.keys())))
    existing_rows.append({k: str(new_row.get(k, "")) for k in all_fields})

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_fields)
        writer.writeheader()
        for existing in existing_rows:
            writer.writerow({k: existing.get(k, "") for k in all_fields})


def process_hardware_metadata(
    metadata_path: str,
    output_csv: str,
    interval_method: str = "wilson",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Parse completed Braket tasks and append one row to hardware_timeseries.csv."""
    if AwsQuantumTask is None:
        raise RuntimeError("Braket SDK not installed. Install amazon-braket-sdk to parse hardware results.")

    method = interval_method.lower()
    if method not in {"wilson", "jeffreys"}:
        raise ValueError("interval_method must be 'wilson' or 'jeffreys'")

    interval_fn = wilson_interval if method == "wilson" else jeffreys_interval

    meta = _load_metadata(pathlib.Path(metadata_path))
    if meta.get("dry_run", False):
        raise RuntimeError("Cannot parse a dry-run metadata file because no tasks were submitted.")

    descriptors: List[Dict[str, Any]] = [dict(d) for d in meta.get("circuit_descriptors", [])]
    if not descriptors:
        raise RuntimeError("Metadata contains no circuit_descriptors")

    cache: Dict[str, Any] = {}

    def get_task_result(task_arn: str) -> Any:
        if task_arn not in cache:
            cache[task_arn] = AwsQuantumTask(task_arn).result()
        return cache[task_arn]

    row: MutableMapping[str, Any] = {
        "created_at": meta.get("created_at", ""),
        "device_arn": meta.get("device_arn", ""),
        "region": meta.get("region", ""),
        "batched": bool(meta.get("batched", False)),
        "task_arns": "|".join(meta.get("task_arns", [])),
        "interval_method": method,
    }

    ro0_errors: List[float] = []
    ro1_errors: List[float] = []
    coherent_anomaly: List[float] = []
    crosstalk_odd: List[float] = []

    for desc in descriptors:
        task_arn = desc.get("task_arn")
        if not task_arn:
            raise RuntimeError("Descriptor missing task_arn; cannot parse deterministically")

        task_result = get_task_result(str(task_arn))
        result_index = int(desc.get("result_index", 0))
        counts = _extract_counts_from_task_result(task_result, result_index=result_index)

        shots = int(desc["shots"])
        circuit_type = str(desc["circuit_type"])
        measured_qubits = _resolve_measured_qubits(desc, task_result, counts)
        bit_order = str(desc.get("bit_order", "measured_qubits_left_to_right"))
        name = str(desc.get("name", f"c{desc.get('circuit_index', 0)}"))

        if circuit_type == "readout_zero":
            for q in [int(qv) for qv in desc.get("qubits", [])]:
                ones = 0
                for bitstring, c in counts.items():
                    ones += int(c) * _bit_for_qubit(bitstring, measured_qubits, q, bit_order)
                p = ones / shots if shots > 0 else 0.0
                lo, hi = interval_fn(ones, shots, alpha=alpha)
                row[f"{name}_q{q}_p10"] = p
                row[f"{name}_q{q}_ci_low"] = lo
                row[f"{name}_q{q}_ci_high"] = hi
                ro0_errors.append(p)

        elif circuit_type == "readout_one":
            for q in [int(qv) for qv in desc.get("qubits", [])]:
                zeros = 0
                for bitstring, c in counts.items():
                    zeros += int(c) * (1 - _bit_for_qubit(bitstring, measured_qubits, q, bit_order))
                p = zeros / shots if shots > 0 else 0.0
                lo, hi = interval_fn(zeros, shots, alpha=alpha)
                row[f"{name}_q{q}_p01"] = p
                row[f"{name}_q{q}_ci_low"] = lo
                row[f"{name}_q{q}_ci_high"] = hi
                ro1_errors.append(p)

        elif circuit_type == "coherent_rx":
            q = int(desc.get("qubits", [0])[0])
            repeats = int(desc.get("repeats", 0))
            ones = 0
            for bitstring, c in counts.items():
                ones += int(c) * _bit_for_qubit(bitstring, measured_qubits, q, bit_order)
            p_one = ones / shots if shots > 0 else 0.0
            expected = math.sin(repeats * math.pi / 4.0) ** 2
            anomaly = abs(p_one - expected)
            lo, hi = interval_fn(ones, shots, alpha=alpha)
            row[f"{name}_q{q}_p1"] = p_one
            row[f"{name}_q{q}_expected_p1"] = expected
            row[f"{name}_q{q}_anomaly"] = anomaly
            row[f"{name}_q{q}_ci_low"] = lo
            row[f"{name}_q{q}_ci_high"] = hi
            coherent_anomaly.append(anomaly)

        elif circuit_type in {"crosstalk", "crosstalk_pair"}:
            pair = [int(v) for v in desc.get("pair", [])]
            odd = 0
            for bitstring, c in counts.items():
                odd += int(c) * _odd_parity_for_pair(bitstring, measured_qubits, pair, bit_order)
            odd_rate = odd / shots if shots > 0 else 0.0
            anomaly = abs(odd_rate - 0.5)
            lo, hi = interval_fn(odd, shots, alpha=alpha)
            pair_label = f"{pair[0]}{pair[1]}" if len(pair) == 2 else "unknown"
            row[f"{name}_{pair_label}_odd_parity"] = odd_rate
            row[f"{name}_{pair_label}_anomaly"] = anomaly
            row[f"{name}_{pair_label}_ci_low"] = lo
            row[f"{name}_{pair_label}_ci_high"] = hi
            crosstalk_odd.append(odd_rate)

        else:
            raise ValueError(f"Unknown circuit_type '{circuit_type}'")

    row["readout_mean_error_zero"] = _aggregate_mean(ro0_errors)
    row["readout_mean_error_one"] = _aggregate_mean(ro1_errors)
    row["coherent_mean_anomaly"] = _aggregate_mean(coherent_anomaly)
    row["crosstalk_mean_odd_parity"] = _aggregate_mean(crosstalk_odd)
    row["crosstalk_mean_odd_parity"] = _aggregate_mean(crosstalk_odd)
    row["crosstalk_mean_anomaly"] = abs(row["crosstalk_mean_odd_parity"] - 0.5)


    output_path = pathlib.Path(output_csv)
    _append_row_dynamic(output_path, row)
    return dict(row)
