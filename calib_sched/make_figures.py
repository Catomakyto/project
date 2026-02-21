from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PLOT_POLICIES: Sequence[str] = (
    "periodic_full",
    "periodic_partial",
    "conservative_bandit",
)


def _load_hardware_timeseries(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "created_at" not in df.columns:
        raise ValueError("CSV must contain 'created_at'.")

    required_cols = [
        "readout_mean_error_zero",
        "readout_mean_error_one",
        "coherent_mean_anomaly",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must contain '{col}'.")

    created_at = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    if created_at.isna().all():
        raise ValueError("Column 'created_at' has no parseable timestamps.")

    if "crosstalk_mean_anomaly" in df.columns:
        crosstalk = pd.to_numeric(df["crosstalk_mean_anomaly"], errors="coerce")
    else:
        crosstalk = pd.Series(np.nan, index=df.index, dtype=float)

    if "crosstalk_mean_odd_parity" in df.columns:
        odd = pd.to_numeric(df["crosstalk_mean_odd_parity"], errors="coerce")
        derived = (odd - 0.5).abs()
        crosstalk = crosstalk.where(~crosstalk.isna(), derived)

    prepared = pd.DataFrame(
        {
            "created_at": created_at,
            "readout_mean_error_zero": pd.to_numeric(df["readout_mean_error_zero"], errors="coerce"),
            "readout_mean_error_one": pd.to_numeric(df["readout_mean_error_one"], errors="coerce"),
            "coherent_mean_anomaly": pd.to_numeric(df["coherent_mean_anomaly"], errors="coerce"),
            "crosstalk_mean_anomaly": pd.to_numeric(crosstalk, errors="coerce"),
        }
    )
    prepared = prepared.dropna().sort_values("created_at")
    if prepared.empty:
        raise ValueError("No valid rows found for plotting hardware time series.")
    return prepared


def _plot_hardware_drift_timeseries(csv_path: str, outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    df = _load_hardware_timeseries(csv_path)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["created_at"], df["readout_mean_error_zero"], linewidth=1.8, label="readout_mean_error_zero")
    ax.plot(df["created_at"], df["readout_mean_error_one"], linewidth=1.8, label="readout_mean_error_one")
    ax.plot(df["created_at"], df["coherent_mean_anomaly"], linewidth=1.8, label="coherent_mean_anomaly")
    ax.plot(df["created_at"], df["crosstalk_mean_anomaly"], linewidth=1.8, label="crosstalk_mean_anomaly")

    ax.set_title("Hardware Drift Time Series", fontsize=13)
    ax.set_xlabel("created_at", fontsize=11)
    ax.set_ylabel("Anomaly / Error", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="best")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()

    out_path = outdir / "hardware_drift_timeseries.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _trace_key(policy: str, suffix: str) -> str:
    return f"{policy}__{suffix}"


def _load_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(str(npz_path))
    with np.load(npz_path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _plot_performance_from_traces(
    traces: Dict[str, np.ndarray],
    out_path: Path,
    policies: Sequence[str],
    switch_step: int | None = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    for policy in policies:
        key = _trace_key(policy, "performance_traces")
        if key not in traces:
            continue
        arr = np.asarray(traces[key], dtype=float)
        if arr.ndim != 2:
            continue
        mean = np.mean(arr, axis=0)
        if arr.shape[0] > 1:
            se = np.std(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
        else:
            se = np.zeros_like(mean)
        ci = 1.96 * se
        x = np.arange(arr.shape[1])
        ax.plot(x, mean, linewidth=2.1, label=policy)
        ax.fill_between(x, mean - ci, mean + ci, alpha=0.16)

    if switch_step is not None and switch_step >= 0:
        ax.axvline(float(switch_step), linestyle="--", linewidth=1.5, color="black", alpha=0.7)

    ax.set_title("Performance vs Time (Mean +/- 95% CI)", fontsize=13)
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Performance", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_ttf_boxplot_from_traces(
    traces: Dict[str, np.ndarray],
    out_path: Path,
    policies: Sequence[str],
) -> None:
    import matplotlib.pyplot as plt

    data: List[np.ndarray] = []
    labels: List[str] = []
    for policy in policies:
        key = _trace_key(policy, "time_to_failure")
        if key not in traces:
            continue
        arr = np.asarray(traces[key], dtype=float)
        if arr.ndim != 1:
            continue
        data.append(arr)
        labels.append(policy)

    if not data:
        return

    y_max = max(float(np.max(arr)) for arr in data)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylim(0.0, y_max + 1.0)
    ax.set_title("Time-to-Failure Distribution by Policy", fontsize=13)
    ax.set_xlabel("Policy", fontsize=11)
    ax.set_ylabel("Time to Failure (steps)", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def generate_figures(csv_path: str, outdir: str) -> List[str]:
    """Generate all required figures from saved evaluation artifacts."""

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    saved: List[str] = []

    hardware_plot = _plot_hardware_drift_timeseries(csv_path, outdir_path)
    saved.append(str(hardware_plot.resolve()))

    eval_npz = outdir_path / "eval_traces.npz"
    if eval_npz.exists():
        eval_traces = _load_npz(eval_npz)
        perf_plot = outdir_path / "performance_vs_time.png"
        _plot_performance_from_traces(eval_traces, perf_plot, policies=PLOT_POLICIES)
        saved.append(str(perf_plot.resolve()))

        ttf_plot = outdir_path / "time_to_failure_boxplot.png"
        _plot_ttf_boxplot_from_traces(eval_traces, ttf_plot, policies=PLOT_POLICIES)
        saved.append(str(ttf_plot.resolve()))

    shock_npz = outdir_path / "shock_traces.npz"
    if shock_npz.exists():
        shock_traces = _load_npz(shock_npz)
        shock_start = int(shock_traces.get("shock_start", np.asarray([-1], dtype=int))[0])
        shock_plot = outdir_path / "shock_performance_vs_time.png"
        _plot_performance_from_traces(
            shock_traces,
            shock_plot,
            policies=PLOT_POLICIES,
            switch_step=shock_start,
        )
        saved.append(str(shock_plot.resolve()))

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clean figures for grounded calibration experiments.")
    parser.add_argument("--csv", default="data/hardware/hardware_timeseries.csv")
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    files = generate_figures(csv_path=args.csv, outdir=args.outdir)
    print("[figures] Saved:")
    for path in files:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
