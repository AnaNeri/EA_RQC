# Evolutionary Algorithm for Robust Quantum Circuits (EA-RQC)

EA-RQC is a Python package that searches for robust qubit clusters on noisy quantum hardware using an evolutionary algorithm.

EA-RQC accompanies research on applying the quantum probabilistic combinator to noisy quantum-circuit analysis and evolutionary circuit synthesis. The repository first validates noise and correction behaviour, then uses evolutionary search to synthesise circuits under hardware gate-set, connectivity, complexity, and noise constraints. QPC-aware fitness evaluates not only ideal circuit behaviour but also the effect of faulty channels.

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

The demos are included when the package is installed, so they can also be run with `python -m demo...` in the installation environment.

Demo summary:
- `map_demo`: selects robust connected qubit clusters from a hardware noise profile.
- `circuit_demo`: demonstrates the complete circuit-evolution workflow.
- `circuit_demo_entangling`: evolves an entangling circuit under native-gate and connectivity constraints.
- `circuit_demo_grover_oracle`: compares equivalent Grover-oracle decompositions under asymmetric noise.
- `circuit_demo_qpc_approximation`: compares exact and approximate QPC evaluation.
- `circuit_demo_qpc_correction_priority`: studies correction-resource prioritisation.
- `circuit_demo_qpc_error_correction`: evolves protective structures around a workload.
- `circuit_demo_qpc_identities`: checks algebraic QPC identities numerically.
- `circuit_demo_qpc_protective_scaffold`: refines protection scaffolds with evolutionary search.
- `circuit_demo_qpc_x_vs_hzh`: compares standard and QPC-guided search for an X circuit.
- `circuit_demo_quantamorphism_protection`: evaluates protective wrappers for a quantamorphism workload.

- Full demo catalog and per-demo purpose:
	- See `demo/README.md`

- Run the full test suite:

```bash
uv run pytest
```

Tests are retained in the repository and source distribution for reproducibility; they are not included in the installed wheel.

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

License
- This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Citation
- Cite version `0.1.0` using the metadata in [CITATION.cff](CITATION.cff). Create a tagged release and archive it with Zenodo before assigning a DOI.

