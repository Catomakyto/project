from __future__ import annotations

import argparse
import pathlib
from typing import Any, Dict

import yaml

from .braket_runs.circuits import build_default_sentinel_suite, scale_descriptor_shots
from .braket_runs.submit import (
    estimate_cost,
    format_cost_estimate,
    pricing_for_device,
    resolve_device_arn,
    submit_sentinel_suite,
    validate_preflight_plan,
)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual hardware sentinel submission with strict spend gating.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--device",
        default=None,
        help="Explicit device alias or ARN. Required for non-dry-run submission.",
    )
    parser.add_argument("--shots-scale", type=float, default=1.0)
    parser.add_argument("--confirm-spend", action="store_true")
    parser.add_argument("--yes-i-understand", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate config + estimate spend, then exit without writing/submitting tasks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Always dry-run even if confirmations are present.",
    )
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    hw = cfg["hardware"]

    default_device = str(hw["default_device"])
    device_input = str(args.device) if args.device is not None else default_device
    explicit_device = args.device is not None

    device_arn, device_label = resolve_device_arn(device_input, hw["aliases"])

    circuits, descriptors = build_default_sentinel_suite(
        monitor_qubits=[int(q) for q in hw["monitor_qubits"]],
        monitor_pair=[int(hw["monitor_pair"][0]), int(hw["monitor_pair"][1])],
        shots_per_circuit=int(hw["sentinel_suite"]["shots_per_circuit"]),
        coherent_repeats=[int(v) for v in hw["sentinel_suite"]["coherent_repeats"]],
    )
    descriptors = scale_descriptor_shots(descriptors, shots_scale=float(args.shots_scale))

    per_device_price = pricing_for_device(device_label, device_arn, hw["pricing"])
    prefer_single_task_batch = bool(hw["batching"]["prefer_single_task"])
    cost = estimate_cost(
        descriptors=descriptors,
        per_task_usd=float(per_device_price["per_task_usd"]),
        per_shot_usd=float(per_device_price["per_shot_usd"]),
        assume_batched_single_task=prefer_single_task_batch,
    )
    preflight = validate_preflight_plan(
        descriptors=descriptors,
        s3_destination=str(hw["s3_destination"]),
        min_shots_per_circuit=1,
        max_shots_per_circuit=100_000,
    )

    print(f"Device label: {device_label}")
    print(f"Device ARN:   {device_arn}")
    print(
        f"Preflight: circuits={preflight['num_circuits']}, total_shots={preflight['total_shots']}, "
        f"s3_bucket={preflight['s3_bucket']}, s3_key_prefix={preflight['s3_key_prefix']}"
    )
    print(format_cost_estimate(cost))

    if args.preflight_only:
        print("Preflight-only mode: exiting without creating metadata or submitting tasks.")
        return

    if args.confirm_spend != args.yes_i_understand:
        parser.error("Both --confirm-spend and --yes-i-understand must be provided together.")

    allow_submit = bool(args.confirm_spend and args.yes_i_understand and not args.dry_run)
    if allow_submit and not explicit_device:
        parser.error(
            "Non-dry-run submission requires an explicit --device alias/ARN "
            "(default device from config is not sufficient)."
        )

    dry_run = not allow_submit
    if dry_run:
        print("DRY-RUN enabled: no Braket tasks will be submitted.")

    metadata_path = submit_sentinel_suite(
        circuits=circuits,
        descriptors=descriptors,
        device_arn=device_arn,
        region=str(hw["default_region"]),
        s3_destination=str(hw["s3_destination"]),
        metadata_dir=str(hw["metadata_dir"]),
        cost_estimate=cost,
        dry_run=dry_run,
        confirm_spend=bool(args.confirm_spend),
        yes_i_understand=bool(args.yes_i_understand),
        prefer_single_task_batch=prefer_single_task_batch,
        tags={
            "project": "calib_sched",
            "mode": "sentinel_logging",
        },
    )

    print(f"Wrote metadata: {pathlib.Path(metadata_path)}")


if __name__ == "__main__":
    main()
