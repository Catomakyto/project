from __future__ import annotations

import argparse
from typing import Any, Dict

import yaml

from .braket_runs.parse import process_hardware_metadata


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse hardware sentinel task metadata and append CSV.")
    parser.add_argument("--metadata", required=True, help="Path to metadata JSON from run_hardware")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out-csv", default="", help="Override output hardware timeseries CSV path")
    parser.add_argument("--interval", default="wilson", choices=["wilson", "jeffreys"])
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    output_csv = args.out_csv or cfg["hardware"]["timeseries_csv"]

    row = process_hardware_metadata(
        metadata_path=args.metadata,
        output_csv=output_csv,
        interval_method=args.interval,
    )

    print(f"Appended row to: {output_csv}")
    print(f"Summary: readout_zero={row['readout_mean_error_zero']:.6f}, "
          f"readout_one={row['readout_mean_error_one']:.6f}, "
          f"coherent={row['coherent_mean_anomaly']:.6f}, "
          f"crosstalk_odd={row['crosstalk_mean_odd_parity']:.6f}")


if __name__ == "__main__":
    main()
