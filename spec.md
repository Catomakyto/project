# Calibration Scheduling Project Specification

## 1. Overview

This document defines the decision‑making problem, data structures and tasks for the drift‑aware calibration scheduling project.  It serves as a common reference for the implementation and experiments.

## 2. State (Context)

At each decision point a context vector is constructed from three categories of information:

### 2.1 Sentinel summaries

Sentinels are quick diagnostic circuits executed on the quantum processor to detect drift.  They provide per‑qubit or per‑pair statistics along with confidence intervals.  The following summaries are extracted and form part of the state vector:

* **Readout confusion**: estimated bit‑flip probabilities \(\hat{p}(1\mid 0)\) and \(\hat{p}(0\mid 1)\) for each qubit with Wilson or Jeffreys interval.  These are measured using preparation of \(|0\rangle\) and \(|1\rangle\) states followed by measurement.
* **Coherent error proxy**: a scalar anomaly score per qubit derived from repeated single‑qubit rotations (e.g. repeated \(R_x(\pi/2)\) gates).  The score captures coherent over‑rotation drift; confidence intervals are estimated from binomial statistics.
* **Crosstalk anomaly score**: an anomaly score for selected qubit pairs, measured using simple two‑qubit patterns (e.g. simultaneous CNOTs or parity checks).  Confidence intervals are estimated as above.

### 2.2 Circuit metadata

Upcoming workload properties are encoded in a feature vector:

* Number of qubits involved
* Circuit depth and two‑qubit gate count
* Entangling topology information (adjacency features)
* A sensitivity prior learned from training (quantifies how performance degrades as each error channel drifts)

### 2.3 Temporal features

* Time since the last full calibration
* Time since the last partial calibration on each channel family
* Recent drift velocities (changes in sentinel summaries per unit time)

## 3. Action Space

The agent may choose one of four discrete actions at each decision point:

1. **Do nothing** – proceed with the workload without probing or recalibration.
2. **Probe** – run a short sentinel suite, analyse the results, then make a new decision immediately afterward.  The probe itself incurs a small shot/time cost but does not reset the device.
3. **Partial recalibration** – reset one error channel family (readout, coherent single‑qubit rotations or two‑qubit crosstalk).  Partial recalibrations incur a moderate cost and reduce drift for the selected channel to near‑zero residual error.
4. **Full recalibration** – perform a complete calibration across all channels.  This yields the highest cost but resets all hidden drift states to near zero residual error.

Additional action variants (e.g. recalibrating multiple channels) may be added later but are avoided here for clarity and tractability.

## 4. Cost Model and Utility

Each action incurs a cost measured in shots (or equivalently in wall‑clock time since calibrations and probes require running circuits).  Let \(C_{\text{probe}}\), \(C_{\text{part}}\) and \(C_{\text{full}}\) denote the shot cost of probing, partial and full recalibration.  The utility for decision time \(t\) is defined as

\[
u_t = P_t \, - \, \lambda\;\text{cost}(a_t) \, - \, \mu\;\mathbf{1}[\text{silent failure}],
\]

where \(P_t \in [0,1]\) is the measured circuit performance (e.g. fidelity proxy) after the action, \(\lambda > 0\) trades off performance with overhead, \(\mu > 0\) penalises silent failures and \(\mathbf{1}[\cdot]\) is the indicator function.  A **silent failure** occurs when performance falls below a predefined threshold \(\tau\) without the policy taking a probe or calibration beforehand.

## 5. Sentinel Suite

Three classes of small circuits are used as sentinels to estimate drift:

### 5.1 Readout sentinel

Prepare all qubits in \(|0\rangle\) and measure; prepare all qubits in \(|1\rangle\) and measure.  The outcomes estimate bit‑flip probabilities.  These circuits are cheap and form the core of drift detection.

### 5.2 Coherent over‑rotation sentinel

Apply a repeated single‑qubit rotation sequence on each qubit, such as repeating \(R_x(\pi/2)\) a fixed number of times, followed by measurement in the computational basis.  Deviations from the expected oscillation envelope act as a proxy for coherent over‑rotation drift.

### 5.3 Crosstalk sentinel

For selected pairs of qubits, run simple two‑qubit patterns (e.g. prepare \(|00\rangle\), perform a CNOT, undo with an inverse, and measure) to expose correlated errors and bursty crosstalk episodes.  Only a small set of sensitive pairs is monitored to control cost.

The sentinel suite definition lives in `braket_runs/circuits.py`.  Each sentinel function returns a list of Braket `Circuit` objects and an associated shot count.

## 6. Implementation Artifacts for Day 1

The following skeleton modules are included in this repository:

* **`braket_runs/circuits.py`** – defines the sentinel circuits using the `braket.circuits.Circuit` API.  These functions are parameterised by the number of qubits and selected pairs.  They return a list of circuits and shot counts.
* **`braket_runs/submit.py`** – provides a function `run_sentinel_suite` that submits the sentinel circuits to an AWS Braket device.  It requires an AWS region, device ARN, shot counts and an S3 bucket name.  The script logs metadata such as timestamps and task IDs to `data/hardware/raw_tasks/`.
* **`braket_runs/parse.py`** – reads Braket task results (once completed) and computes sentinel summary statistics with Wilson/Jeffreys confidence intervals.  It appends a row to `data/hardware/hardware_timeseries.csv` with the computed statistics and run metadata.

Additionally, a configuration file `configs/default.yaml` describes hyperparameters such as shot counts, risk thresholds, cost weights \(\lambda\), \(\mu\) and the silent failure threshold \(\tau\).

## 7. Next Steps

Day 1 tasks include finalising this specification, creating the repository structure and configuration, implementing the sentinel circuits and submission script, and launching the first sentinel run.  Because AWS authentication requires user credentials, the actual submission of tasks will need a user takeover to log in to the AWS console.  The skeleton code can be executed once credentials are configured.