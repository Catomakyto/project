from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import pandas as pd

CHANNEL_COLUMNS = {
    "z": "readout_mean_error_zero",
    "o": "readout_mean_error_one",
    "c": "coherent_mean_anomaly",
}
DEFAULT_BURST_PROB = 0.1
EPS = 1e-9


@dataclass(frozen=True)
class ChannelFit:
    """Fitted drift parameters for one channel."""

    mu: float
    sigma: float
    p_burst: float
    sigma_burst: float
    initial_mean: float
    initial_sigma: float
    min_value: float
    max_value: float
    n_points: int
    n_deltas: int

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "mu": float(self.mu),
            "sigma": float(self.sigma),
            "p_burst": float(self.p_burst),
            "sigma_burst": float(self.sigma_burst),
            "initial_mean": float(self.initial_mean),
            "initial_sigma": float(self.initial_sigma),
            "min_value": float(self.min_value),
            "max_value": float(self.max_value),
            "n_points": int(self.n_points),
            "n_deltas": int(self.n_deltas),
        }


def _mad(values: np.ndarray, center: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.median(np.abs(values - center)))


def _robust_scale(values: np.ndarray, center: float) -> float:
    return 1.4826 * _mad(values, center)


def _clip_probability(value: float) -> float:
    return float(min(0.95, max(0.0, value)))


def _to_numeric(series: pd.Series, column_name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().all():
        raise ValueError(f"Column '{column_name}' contains no numeric values.")
    return numeric


def _prepare_dataframe(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")

    df = pd.read_csv(path)
    if "created_at" not in df.columns:
        raise ValueError("CSV must contain a 'created_at' column.")

    for col in CHANNEL_COLUMNS.values():
        if col not in df.columns:
            raise ValueError(f"CSV must contain column '{col}'.")

    created_at = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    if created_at.isna().all():
        raise ValueError("Column 'created_at' has no parseable ISO timestamps.")

    x: pd.Series
    if "crosstalk_mean_anomaly" in df.columns:
        x = _to_numeric(df["crosstalk_mean_anomaly"], "crosstalk_mean_anomaly")
    else:
        x = pd.Series(np.nan, index=df.index, dtype=float)

    if "crosstalk_mean_odd_parity" in df.columns:
        odd = _to_numeric(df["crosstalk_mean_odd_parity"], "crosstalk_mean_odd_parity")
        derived = (odd - 0.5).abs()
        x = x.where(~x.isna(), derived)

    if x.isna().all():
        raise ValueError(
            "CSV must contain either 'crosstalk_mean_anomaly' or "
            "'crosstalk_mean_odd_parity' with numeric data."
        )

    prepared = pd.DataFrame(
        {
            "created_at": created_at,
            "z": _to_numeric(df[CHANNEL_COLUMNS["z"]], CHANNEL_COLUMNS["z"]),
            "o": _to_numeric(df[CHANNEL_COLUMNS["o"]], CHANNEL_COLUMNS["o"]),
            "c": _to_numeric(df[CHANNEL_COLUMNS["c"]], CHANNEL_COLUMNS["c"]),
            "x": pd.to_numeric(x, errors="coerce"),
        }
    )
    prepared = prepared.dropna(subset=["created_at", "z", "o", "c", "x"]).sort_values("created_at")
    if prepared.empty:
        raise ValueError("No valid rows after parsing and cleaning required columns.")
    return prepared


def _fit_channel(values: np.ndarray) -> ChannelFit:
    if values.size < 1:
        raise ValueError("At least one value is required to fit a channel.")

    deltas = np.diff(values)
    n_deltas = int(deltas.size)

    if n_deltas > 0:
        mu = float(np.median(deltas))
        sigma = _robust_scale(deltas, mu)
        if sigma < EPS:
            sigma = float(np.std(deltas, ddof=1 if n_deltas > 1 else 0))
        if sigma < EPS:
            sigma = max(1e-4, float(np.median(np.abs(deltas - mu)) + 1e-4))
    else:
        mu = 0.0
        sigma = max(1e-4, _robust_scale(values, float(np.median(values))))

    if n_deltas >= 3:
        residual = deltas - mu
        burst_threshold = 2.5 * max(sigma, 1e-6)
        burst_mask = np.abs(residual) > burst_threshold
        p_burst = float(np.mean(burst_mask))

        if int(np.sum(burst_mask)) > 0:
            burst_residual = residual[burst_mask]
            sigma_burst = _robust_scale(burst_residual, float(np.median(burst_residual)))
            if sigma_burst < EPS:
                sigma_burst = float(np.std(burst_residual, ddof=0))
        else:
            sigma_burst = 0.0

        if sigma_burst < 1.5 * sigma:
            sigma_burst = 3.0 * sigma
        if p_burst <= 0.0:
            p_burst = DEFAULT_BURST_PROB
    else:
        p_burst = DEFAULT_BURST_PROB
        sigma_burst = 3.0 * sigma

    initial_mean = float(np.quantile(values, 0.2))
    initial_sigma = _robust_scale(values, float(np.median(values)))
    if initial_sigma < EPS:
        initial_sigma = float(np.std(values, ddof=0))
    if initial_sigma < EPS:
        initial_sigma = max(1e-4, 0.1 * max(initial_mean, 1e-3))

    return ChannelFit(
        mu=float(mu),
        sigma=float(max(1e-6, sigma)),
        p_burst=_clip_probability(max(0.01, p_burst)),
        sigma_burst=float(max(1e-6, sigma_burst)),
        initial_mean=float(max(0.0, initial_mean)),
        initial_sigma=float(max(1e-6, initial_sigma)),
        min_value=0.0,
        max_value=1.0,
        n_points=int(values.size),
        n_deltas=n_deltas,
    )


def fit_drift_params(csv_path: str) -> Dict[str, object]:
    """Fit grounded drift parameters from a hardware time-series CSV."""

    prepared = _prepare_dataframe(csv_path)
    fitted = {channel: _fit_channel(prepared[channel].to_numpy(dtype=float)) for channel in ["z", "o", "c", "x"]}

    if len(prepared) > 1:
        dt_seconds = (
            prepared["created_at"].diff().dropna().dt.total_seconds().to_numpy(dtype=float)
        )
        median_dt_seconds = float(np.median(dt_seconds)) if dt_seconds.size > 0 else None
    else:
        median_dt_seconds = None

    result: Dict[str, object] = {
        "source_csv": str(Path(csv_path)),
        "num_rows_used": int(len(prepared)),
        "created_at_min": str(prepared["created_at"].min()),
        "created_at_max": str(prepared["created_at"].max()),
        "median_step_seconds": median_dt_seconds,
        "channels": {name: fit.to_dict() for name, fit in fitted.items()},
    }
    return result


def save_fitted_params(params: Mapping[str, object], out_path: str) -> None:
    """Write fitted parameter JSON to disk."""

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(params), indent=2), encoding="utf-8")


def fit_and_save(csv_path: str, out_path: str) -> Dict[str, object]:
    """Fit parameters from CSV and save them to JSON."""

    params = fit_drift_params(csv_path)
    save_fitted_params(params, out_path)
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit simple hardware-grounded drift parameters.")
    parser.add_argument(
        "--csv",
        default="data/hardware/hardware_timeseries.csv",
        help="Path to hardware time-series CSV.",
    )
    parser.add_argument(
        "--out",
        default="results/fitted_drift_params.json",
        help="Output JSON path for fitted parameters.",
    )
    args = parser.parse_args()

    print(f"[fit] Loading hardware data from: {args.csv}")
    params = fit_and_save(args.csv, args.out)
    channels = params.get("channels", {})
    print(f"[fit] Rows used: {params.get('num_rows_used')}")
    for name in ["z", "o", "c", "x"]:
        item = channels.get(name, {})
        print(
            "[fit] "
            f"{name}: mu={float(item.get('mu', 0.0)):.6g}, "
            f"sigma={float(item.get('sigma', 0.0)):.6g}, "
            f"p_burst={float(item.get('p_burst', 0.0)):.3f}, "
            f"sigma_burst={float(item.get('sigma_burst', 0.0)):.6g}"
        )
    print(f"[fit] Saved parameters: {args.out}")


if __name__ == "__main__":
    main()
