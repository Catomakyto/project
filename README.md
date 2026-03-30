# Drift-Aware Calibration Scheduling

## Commands

1. Simulation evaluation (no AWS spend)
```bash
python -m calib_sched.run_sim --config configs/default.yaml --out results/sim
```

2. Generate synthetic logged data with propensities
```bash
python -m calib_sched.run_sim \
  --config configs/default.yaml \
  --out results/sim \
  --log-out data/sim/logs.parquet
```

3. Off-policy evaluation (IPS + DR + LCB)
```bash
python -m calib_sched.ope_eval \
  --config configs/default.yaml \
  --log data/sim/logs.parquet \
  --policy conservative_bandit \
  --out results/ope/ope_report.json
```

4. Hardware sentinel submit (manual + confirmation gate)

Dry-run preview (safe default):
```bash
python -m calib_sched.run_hardware \
  --config configs/default.yaml \
  --device rigetti_ankaa3 \
  --shots-scale 1.0
```

Explicit preflight validation only (no metadata write, no submission):
```bash
python -m calib_sched.run_hardware \
  --config configs/default.yaml \
  --device rigetti_ankaa3 \
  --shots-scale 1.0 \
  --preflight-only
```

Actual submit (requires both confirmation flags):
```bash
python -m calib_sched.run_hardware \
  --config configs/default.yaml \
  --device rigetti_ankaa3 \
  --confirm-spend \
  --yes-i-understand \
  --shots-scale 1.0
```

5. Parse completed hardware tasks + append time-series CSV
```bash
python -m calib_sched.parse_hardware \
  --config configs/default.yaml \
  --metadata data/hardware/raw_tasks/<metadata_file>.json
```

## Sentinel Plan in `configs/default.yaml`
Default hardware sentinel suite per timepoint (5 circuits):
1. `readout_zero` on `[0,1]`
2. `readout_one` on `[0,1]`
3. `coherent_rx` repeats=4 on `[0]`
4. `coherent_rx` repeats=8 on `[0]`
5. `crosstalk` parity probe on pair `[0,1]`

Default shots per circuit: `1500` (scaled by `--shots-scale`).

Default device: `sv1` (safe simulator default in config). Use `--device rigetti_ankaa3` explicitly when you intentionally want QPU submissions.

## Tests
```bash
python -m unittest discover -s tests -p 'test_*.py'
```
