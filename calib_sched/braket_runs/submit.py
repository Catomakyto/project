from __future__ import annotations

import datetime as dt
import json
import pathlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from braket.aws import AwsDevice
except ImportError:
    AwsDevice = None  # type: ignore

from .circuits import Circuit


@dataclass
class CostEstimate:
    estimated_tasks: int
    estimated_shots: int
    per_task_usd: float
    per_shot_usd: float
    estimated_total_usd: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_s3_destination(s3_uri: str) -> Tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError("s3_destination must start with s3://")
    remainder = s3_uri[len("s3://") :]
    if "/" not in remainder:
        return remainder, ""
    bucket, key = remainder.split("/", 1)
    return bucket, key.rstrip("/")


def _format_s3_destination(bucket: str, key_prefix: str) -> str:
    return f"s3://{bucket}" if not key_prefix else f"s3://{bucket}/{key_prefix}"


def _resolve_submission_s3_destination(device: Any, requested_s3_destination: str) -> Tuple[str, Optional[str]]:
    """Return an S3 destination that Braket task creation will accept.

    Some accounts/device contexts reject custom bucket names and require an
    `amazon-braket-*` bucket. When that happens, we auto-fallback to the
    session default Braket bucket while preserving the configured key prefix.
    """
    bucket, key_prefix = _parse_s3_destination(requested_s3_destination)
    if bucket.startswith("amazon-braket-"):
        return requested_s3_destination, None

    default_bucket = str(device.aws_session.default_bucket())
    effective_s3_destination = _format_s3_destination(default_bucket, key_prefix)
    reason = (
        f"Configured bucket '{bucket}' is not Braket-compatible in this account/device context; "
        f"using default bucket '{default_bucket}' instead."
    )
    return effective_s3_destination, reason


def validate_preflight_plan(
    descriptors: Sequence[Mapping[str, Any]],
    s3_destination: str,
    min_shots_per_circuit: int = 1,
    max_shots_per_circuit: int = 100_000,
) -> Dict[str, Any]:
    """Validate a submission plan without submitting tasks."""
    bucket, key_prefix = _parse_s3_destination(s3_destination)
    if not descriptors:
        raise ValueError("No circuits/descriptors provided for submission.")

    shot_counts = [int(d["shots"]) for d in descriptors]
    for shots in shot_counts:
        if shots < min_shots_per_circuit or shots > max_shots_per_circuit:
            raise ValueError(
                f"Invalid shot count {shots}: must be in [{min_shots_per_circuit}, {max_shots_per_circuit}]"
            )

    return {
        "s3_bucket": bucket,
        "s3_key_prefix": key_prefix,
        "num_circuits": len(descriptors),
        "total_shots": int(sum(shot_counts)),
        "min_shots_per_circuit": min_shots_per_circuit,
        "max_shots_per_circuit": max_shots_per_circuit,
    }


def resolve_device_arn(device: str, aliases: Mapping[str, str]) -> Tuple[str, str]:
    """Resolve a device alias/ARN into (device_arn, device_label)."""
    if device in aliases:
        return aliases[device], device
    if device.startswith("arn:aws:braket"):
        return device, "custom"
    known = ", ".join(sorted(aliases))
    raise ValueError(f"Unknown device '{device}'. Use one of [{known}] or a full ARN.")


def pricing_for_device(device_label: str, device_arn: str, pricing_cfg: Mapping[str, Any]) -> Dict[str, float]:
    """Return pricing values from config only (no hard-coded prices)."""
    devices = pricing_cfg.get("devices", {})
    default = pricing_cfg.get("default", None)

    row = None
    if device_label in devices:
        row = devices[device_label]
    elif device_arn in devices:
        row = devices[device_arn]
    elif default is not None:
        row = default

    if row is None:
        raise ValueError(
            f"No pricing entry found for device '{device_label}' / '{device_arn}'. "
            "Add per_task_usd and per_shot_usd in configs/default.yaml"
        )

    if "per_task_usd" not in row or "per_shot_usd" not in row:
        raise ValueError("Pricing row must contain per_task_usd and per_shot_usd")

    return {
        "per_task_usd": float(row["per_task_usd"]),
        "per_shot_usd": float(row["per_shot_usd"]),
    }


def estimate_cost(
    descriptors: Sequence[Mapping[str, Any]],
    per_task_usd: float,
    per_shot_usd: float,
    assume_batched_single_task: bool,
) -> CostEstimate:
    estimated_tasks = 1 if assume_batched_single_task else len(descriptors)
    estimated_shots = int(sum(int(desc["shots"]) for desc in descriptors))
    estimated_total_usd = estimated_tasks * per_task_usd + estimated_shots * per_shot_usd
    return CostEstimate(
        estimated_tasks=estimated_tasks,
        estimated_shots=estimated_shots,
        per_task_usd=per_task_usd,
        per_shot_usd=per_shot_usd,
        estimated_total_usd=estimated_total_usd,
    )


def format_cost_estimate(cost: CostEstimate) -> str:
    return (
        "Estimated spend before submission:\n"
        f"  tasks: {cost.estimated_tasks}\n"
        f"  shots: {cost.estimated_shots}\n"
        f"  per_task_usd: ${cost.per_task_usd:.6f}\n"
        f"  per_shot_usd: ${cost.per_shot_usd:.6f}\n"
        f"  estimated_total_usd: ${cost.estimated_total_usd:.6f}"
    )


def _attempt_single_task_batch(
    device: Any,
    circuits: Sequence[Circuit],
    descriptors: Sequence[Mapping[str, Any]],
    s3_destination: str,
    tags: Mapping[str, str],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Attempt to submit circuits in a single batched Braket API call.

    This path is best-effort because device SDK capabilities differ by backend.
    """
    bucket, key = _parse_s3_destination(s3_destination)
    shots = [int(d["shots"]) for d in descriptors]
    unique_shots = sorted(set(shots))
    if len(unique_shots) != 1:
        raise ValueError(
            "Batched submission requires identical shots across circuits; "
            f"got {unique_shots}."
        )

    task = device.run_batch(
        task_specifications=list(circuits),
        shots=unique_shots[0],
        s3_destination_folder=(bucket, key),
        disable_qubit_rewiring=True,
        tags=dict(tags),
    )

    # Different SDK versions may expose different return shapes.
    if hasattr(task, "id"):
        task_arns = [str(task.id)]
    elif hasattr(task, "tasks"):
        task_arns = [str(t.id) for t in task.tasks if hasattr(t, "id")]
    elif isinstance(task, list):
        task_arns = [str(t.id) for t in task if hasattr(t, "id")]
    else:
        raise RuntimeError("Unsupported return value from batched submission")

    desc_out: List[Dict[str, Any]] = []
    if len(task_arns) == 1:
        for i, desc in enumerate(descriptors):
            d = dict(desc)
            d["task_arn"] = task_arns[0]
            d["result_index"] = i
            desc_out.append(d)
    elif len(task_arns) == len(descriptors):
        for i, (desc, arn) in enumerate(zip(descriptors, task_arns)):
            d = dict(desc)
            d["task_arn"] = arn
            d["result_index"] = 0
            d["circuit_index"] = i
            desc_out.append(d)
    else:
        raise RuntimeError(
            "Batched submission returned an unexpected number of task IDs "
            f"({len(task_arns)} for {len(descriptors)} circuits)"
        )
    return task_arns, desc_out


def _submit_unbatched(
    device: Any,
    circuits: Sequence[Circuit],
    descriptors: Sequence[Mapping[str, Any]],
    s3_destination: str,
    tags: Mapping[str, str],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    bucket, key = _parse_s3_destination(s3_destination)
    task_arns: List[str] = []
    desc_out: List[Dict[str, Any]] = []

    for circuit, desc in zip(circuits, descriptors):
        task = device.run(
            task_specification=circuit,
            shots=int(desc["shots"]),
            s3_destination_folder=(bucket, key),
            disable_qubit_rewiring=True,
            tags=dict(tags),
        )
        arn = str(task.id)
        task_arns.append(arn)
        d = dict(desc)
        d["task_arn"] = arn
        d["result_index"] = 0
        desc_out.append(d)

    return task_arns, desc_out


def submit_sentinel_suite(
    circuits: Sequence[Circuit],
    descriptors: Sequence[Mapping[str, Any]],
    device_arn: str,
    region: str,
    s3_destination: str,
    metadata_dir: str,
    cost_estimate: CostEstimate,
    dry_run: bool,
    confirm_spend: bool,
    yes_i_understand: bool,
    prefer_single_task_batch: bool,
    tags: Optional[Mapping[str, str]] = None,
) -> pathlib.Path:
    """Submit sentinel circuits with strict spend gating and full metadata logging."""
    if len(circuits) != len(descriptors):
        raise ValueError("circuits and descriptors length mismatch")

    if not dry_run and not (confirm_spend and yes_i_understand):
        raise PermissionError(
            "Submission blocked: both --confirm-spend and --yes-i-understand are required."
        )

    metadata_root = pathlib.Path(metadata_dir)
    _ensure_dir(metadata_root)
    created_at = _utc_timestamp()

    metadata: Dict[str, Any] = {
        "created_at": created_at,
        "device_arn": device_arn,
        "region": region,
        "s3_destination": s3_destination,
        "requested_s3_destination": s3_destination,
        "effective_s3_destination": s3_destination,
        "s3_destination_override_reason": None,
        "dry_run": bool(dry_run),
        "confirm_spend": bool(confirm_spend),
        "yes_i_understand": bool(yes_i_understand),
        "prefer_single_task_batch": bool(prefer_single_task_batch),
        "estimated_cost": cost_estimate.to_dict(),
        "shots": [int(d["shots"]) for d in descriptors],
        "shot_counts": [int(d["shots"]) for d in descriptors],
        "circuit_descriptors": [dict(d) for d in descriptors],
        "batched": False,
        "task_arns": [],
        "task_ids": [],
        "batch_error": None,
    }

    if dry_run:
        filename = f"sentinel_run_dryrun_{created_at.replace(':', '_')}.json"
        path = metadata_root / filename
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return path

    if AwsDevice is None:
        raise RuntimeError(
            "Braket SDK not installed. Install amazon-braket-sdk before submission."
        )

    device = AwsDevice(device_arn)
    effective_s3_destination, s3_override_reason = _resolve_submission_s3_destination(
        device=device,
        requested_s3_destination=s3_destination,
    )
    metadata["s3_destination"] = effective_s3_destination
    metadata["effective_s3_destination"] = effective_s3_destination
    metadata["s3_destination_override_reason"] = s3_override_reason

    task_arns: List[str] = []
    submitted_descriptors: List[Dict[str, Any]] = [dict(d) for d in descriptors]
    batched = False
    batch_error: Optional[str] = None

    if prefer_single_task_batch:
        try:
            task_arns, submitted_descriptors = _attempt_single_task_batch(
                device=device,
                circuits=circuits,
                descriptors=descriptors,
                s3_destination=effective_s3_destination,
                tags=tags or {},
            )
            batched = True
        except Exception as exc:  # pragma: no cover - depends on SDK/device support
            batch_error = str(exc)

    if not batched:
        task_arns, submitted_descriptors = _submit_unbatched(
            device=device,
            circuits=circuits,
            descriptors=descriptors,
            s3_destination=effective_s3_destination,
            tags=tags or {},
        )

    metadata["batched"] = batched
    metadata["batch_error"] = batch_error
    metadata["task_arns"] = task_arns
    metadata["task_ids"] = list(task_arns)
    metadata["actual_tasks_submitted"] = len(task_arns)
    metadata["circuit_descriptors"] = submitted_descriptors

    filename = f"sentinel_run_{created_at.replace(':', '_')}.json"
    path = metadata_root / filename
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path
