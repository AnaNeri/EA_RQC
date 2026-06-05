# Evolutionary Algorithm for Robust Quantum Circuits (EA-RQC)

EA-RQC is a Python package for searching robust qubit clusters on noisy quantum hardware using an evolutionary algorithm.

The project includes:

- Core evolutionary mapping logic.
- Data models and operators for selection, crossover, and mutation.
- Parsers to construct internal models from Qiskit-like sources.
- Tests and a runnable demo.

## Requirements

- Python 3.10+

## Installation

Clone the repository and install it in editable mode:

```bash
pip install -e .
```

Install test dependencies:

```bash
pip install -e .[test]
```

## Run the demo

Run the example in `demo/map_demo.py`:

```bash
python -m demo.map_demo
```

The demo prints:

- The best cluster found.
- The best fitness score.
- The final ranked population.
- The number of generations executed.

Run the circuit-level intermediate demo in `demo/circuit_demo.py`:

```bash
python -m demo.circuit_demo
```

This demo prints intermediate checkpoints:

- Shared structure anchors detected between parent circuits.
- Children produced by probabilistic anchor crossover.
- Mutation result.
- Graph projection from list-based circuit representation.
- Multi-objective fitness breakdown.
- Generation-by-generation diversity for circuit EA.

## Run tests

```bash
pytest
```

## Main API

The main orchestration function is `evolution.cluster.evolutionary.evolutionary_best_clusters`.

Circuit-level orchestration is available in `evolution.circuit.evolutionary.evolutionary_best_circuit`.

Important parameters:

- `device_qubits`: total number of qubits in the device.
- `target_qubits`: cluster size to search for.
- `max_generation`: number of generations to evolve.
- `population`: initial population size.
- `noise_model`: qubit error information.
- `type_ranking`: ranking strategy (`linear`, `exponential`, or `deterministic`).
- `lambda_ratio`: number of parent-pair rounds per generation.
- `crossover_rate`: probability of applying crossover.
- `mutation_rate`: probability of mutating a child.
- `coupling_graph`: optional topology for coupling-aware fitness.
- `prioritize`: objective weighting mode (`qubits`, `coupling`, or `both`).

Return value includes the final population and fitness values, plus the best cluster, best fitness, and generations executed.

## Project structure

- `evolution/`: canonical evolutionary algorithm implementation.
- `circuit/`: circuit and noise entities.
- `adapters/`: canonical integration layer (including Qiskit adapters).
- `demo/`: runnable demonstration script.
- `tests/`: test suite.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

