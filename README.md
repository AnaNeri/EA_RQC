# Evolutionary Algorithm for Robust Quantum Circuits (EA-RQC)

EA-RQC is a Python package that searches for robust qubit clusters on noisy quantum hardware using an evolutionary algorithm.

Features
- Core evolutionary mapping logic for cluster search
- Circuit-level evolutionary routines and fitness evaluation
- Qiskit-like adapters and parsers
- Demo scripts and a test suite

Requirements
- Python 3.10 or newer

Installation
1. Clone the repository.
2. Install dependencies:

```bash
uv sync
```

3. (Optional) Install test dependencies:

```bash
uv sync --extra test
```

Quick start
- Run the mapping demo:

```bash
uv run -m demo.map_demo
```

- Run a circuit demo (examples available under `demo/`):

```bash
uv run -m demo.circuit_demo
uv run -m demo.circuit_demo_grover_oracle
```

- Full demo catalog and per-demo purpose:
	- See `demo/README.md`

- Run the full test suite:

```bash
uv run pytest
```

Main API
- Cluster-level orchestration: `evolution.cluster.evolutionary.evolutionary_best_clusters`
- Circuit-level orchestration: `evolution.circuit.evolutionary.evolutionary_best_circuit`

Key parameters (examples)
- `device_qubits`: total device qubits
- `target_qubits`: cluster size to search
- `max_generation`: number of generations
- `population`: initial population size
- `noise_model`: qubit error information
- `type_ranking`: ranking strategy (`linear`, `exponential`, `deterministic`)
- `crossover_rate`, `mutation_rate`, `lambda_ratio`
- `coupling_graph`: optional device topology

Project layout
- `evolution/` — evolutionary algorithm implementation
- `circuit/` — circuit and noise entities
- `adapters/` — integration layer (Qiskit adapters)
- `demo/` — runnable demonstrations
- `tests/` — test suite

Contributing
- Open issues and pull requests are welcome. Follow repository coding style and add tests for new features.

License
- This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

