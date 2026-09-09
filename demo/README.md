# Demo Guide (uv)

This folder contains runnable demos for cluster mapping and circuit-level QPC evolution.

## Prerequisites

From the repository root:

```bash
uv sync
```

Optional (for tests):

```bash
uv sync --extra test
```

The demos are included in the installed package and can also be run outside a repository checkout:

```bash
python -m demo --list
python -m demo.map_demo
```

## Run Demos

You can list demos through the menu entrypoint:

```bash
uv run -m demo --list
```

Run the default demo:

```bash
uv run -m demo
```

Demo launcher options (`uv run -m demo ...`):
- `--list` / `-l`: print available demos and exit.
- `<index>`: run by 1-based index from the list, e.g. `uv run -m demo 2`.
- `<module_name>`: run by module key, e.g. `uv run -m demo circuit_demo_grover_oracle`.

Or run each module directly:

### 1) `circuit_demo.py`
- Command:

```bash
uv run -m demo.circuit_demo
```

- What it tests:
  - End-to-end circuit EA flow: device data loading, cluster mapping, anchor crossover, mutation, fitness breakdown, and final evolved circuit summary.

- CLI options:
  - None.

### 2) `circuit_demo_entangling.py`
- Command:

```bash
uv run -m demo.circuit_demo_entangling
```

- What it tests:
  - Entangling-target evolution (includes `cx`) and comparison against a Qiskit transpile baseline (depth/size/2q and semantic fidelity metrics).

- CLI options:
  - None.

### 3) `circuit_demo_grover_oracle.py`
- Command:

```bash
uv run -m demo.circuit_demo_grover_oracle
```

- What it tests:
  - Grover oracle (CZ) synthesis from samples under asymmetric qubit noise.
  - Shows ideal fitness can tie equivalent decompositions while QPC distinguishes the more noise-robust one.

- CLI options:
  - None.

### 4) `circuit_demo_qpc_approximation.py`
- Command:

```bash
uv run -m demo.circuit_demo_qpc_approximation
```

- What it tests:
  - EA convergence using QPC `exact` vs first-order `approx` mode.
  - Demonstrates speed/cost benefits of approximation with small error rates.

- CLI options:
  - None.

### 5) `circuit_demo_qpc_correction_priority.py`
- Command:

```bash
uv run -m demo.circuit_demo_qpc_correction_priority
```

- What it tests:
  - Resource-priority behavior under heterogeneous qubit error rates.
  - Shows fitness gains when correction targets the highest-error qubits first.

- CLI options:
  - None.

### 6) `circuit_demo_qpc_error_correction.py`
- Command:

```bash
uv run -m demo.circuit_demo_qpc_error_correction
```

- What it tests:
  - Evolution of correction wrappers around a given logical workload (with ancilla support).
  - Evaluates whether evolved structure improves QPC fitness on working qubits.

- CLI options:
  - None.

### 7) `circuit_demo_qpc_identities.py`
- Command:

```bash
uv run -m demo.circuit_demo_qpc_identities
```

- What it tests:
  - Numerical verification of QPC probabilistic-combinator algebraic identities (residual checks, pass/fail summary).

- CLI options:
  - None.

### 8) `circuit_demo_qpc_protective_scaffold.py`
- Command:

```bash
uv run -m demo.circuit_demo_qpc_protective_scaffold
```

- What it tests:
  - Protective scaffold search for a logical X workload on qubit 0, with ancilla-assisted structure.
  - Starts from either `surface-mini` or `repetition` scaffold, then optionally refines with EA.
  - Re-ranks candidates using final noisy-channel correction impact (plus topology/syndrome signals), not only raw EA score.
  - Supports optional recovery-map experiments (`off`, `ideal`, `noisy`) and noise-profile sweeps.

- CLI options:
  - `--preset {micro,smoke,fast,balanced,thorough}`
    - Default: `fast`
    - Controls EA budget/depth and gate catalog.
  - `--scaffold {surface-mini,repetition}`
    - Default: `surface-mini`
    - Selects initial hand-crafted protection scaffold.
  - `--skip-evolution`
    - Default: disabled
    - Skips EA loop and only computes workload/scaffold scores.
  - `--noise-profile {uniform,x-only}`
    - Default: `uniform`
    - `uniform`: applies noise to all gates.
    - `x-only`: applies noise only on X gates.
  - `--recovery {off,ideal,noisy}`
    - Default: `off`
    - Applies post-wrapper CPTP recovery map.
  - `--syndrome-error-rate <float>`
    - Default: `0.02`
    - Syndrome processing error rate used in `--recovery noisy`.
  - `--readout-error-rate <float>`
    - Default: `0.02`
    - Syndrome readout bit-flip error rate used in `--recovery noisy`.
  - `--recovery-confidence-threshold <float>`
    - Default: `0.75`
    - Recovery is only applied when syndrome confidence exceeds this threshold.

- Example runs:

```bash
uv run -m demo.circuit_demo_qpc_protective_scaffold --preset balanced --scaffold repetition
uv run -m demo.circuit_demo_qpc_protective_scaffold --recovery noisy --syndrome-error-rate 0.01 --readout-error-rate 0.03
uv run -m demo.circuit_demo_qpc_protective_scaffold --skip-evolution --noise-profile x-only
```

### 9) `circuit_demo_qpc_x_vs_hzh.py`
- Command:

```bash
uv run -m demo.circuit_demo_qpc_x_vs_hzh
```

- What it tests:
  - Standard vs QPC-guided EA search for an `X` target.
  - Illustrates how QPC tends to prefer shorter, more noise-robust solutions.

- CLI options:
  - None.

### 10) `circuit_demo_quantamorphism_protection.py`
- Command:

```bash
uv run -m demo.circuit_demo_quantamorphism_protection
```

- What it tests:
  - Baseline "quantamorphism" circuit plus ancilla-based protective wrappers.
  - Compares composed designs under multiple noise scenarios to estimate protection impact by qubit.

- CLI options:
  - None.

### 11) `map_demo.py`
- Command:

```bash
uv run -m demo.map_demo
```

- What it tests:
  - Evolutionary qubit-cluster mapping on a backend-like noise profile and coupling graph.
  - Reports best cluster, fitness, and final population scores.

- CLI options:
  - None.

## Notes

- All commands should be run from the repository root.
- The demos are included in the installed package.
- The test suite is available from the repository and source distribution, but is not included in the wheel.
- If a demo uses optional Qiskit functionality, ensure dependencies are installed by `uv sync`.
