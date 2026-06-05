"""Demo for circuit EA using an entangling target circuit (includes cx)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from circuit.entities.circuit_list import CircuitGate, CircuitList
from evolution.circuit.evolutionary import CircuitEvolutionConfig, evolutionary_best_circuit
from evolution.circuit.fitness import fitness_circuit
from evolution.circuit.matrix_eval import circuit_to_matrix, fidelity_similarity, frobenius_similarity

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Operator
except ImportError:  # pragma: no cover - optional runtime dependency for the demo
    QuantumCircuit = None
    transpile = None
    Operator = None


@dataclass(frozen=True)
class DemoNoiseModel:
    qubit_error_rates: dict[int, float]


@dataclass(frozen=True)
class DemoConfig:
    basis_gates: list[str]


class DemoBackend:
    """Small backend-like object to demonstrate base-gate extraction."""

    def configuration(self):
        return DemoConfig(basis_gates=["id", "rz", "sx", "x", "h", "ecr", "measure", "barrier"])


def _gate(name: str, qubits: tuple[int, ...], depth: int, parameter: float | None = None) -> CircuitGate:
    params = (parameter,) if parameter is not None else ()
    return CircuitGate.from_values(name=name, qubits=qubits, parameters=params, depth=depth)


def print_circuit(label: str, circuit: CircuitList) -> None:
    print(f"{label}: cluster={circuit.cluster}, gates={len(circuit.gates)}")
    for index, gate in enumerate(circuit.gates, start=1):
        print(
            f"  {index:02d}. name={gate.name} qubits={list(gate.qubits)} "
            f"params={list(gate.parameters)} depth={gate.depth} immutable={gate.immutable}"
        )


def print_matrix(label: str, matrix: np.ndarray) -> None:
    print(f"{label} ({matrix.shape[0]}x{matrix.shape[1]}):")
    print(np.array2string(matrix, precision=4, suppress_small=True))


def _extract_catalog_from_backend(backend: DemoBackend) -> list[str]:
    basis = backend.configuration().basis_gates
    excluded = {"measure", "barrier", "delay", "reset", "snapshot"}
    return [name.lower().strip() for name in basis if name.lower().strip() not in excluded]


def _localize_circuit(circuit: CircuitList) -> CircuitList:
    if circuit.cluster:
        logical_qubits = list(circuit.cluster)
    else:
        logical_qubits = sorted({qubit for gate in circuit.gates for qubit in gate.qubits})

    if not logical_qubits:
        return circuit.clone()

    qubit_map = {qubit: index for index, qubit in enumerate(logical_qubits)}
    localized_gates = []
    for gate in circuit.gates:
        localized_gates.append(
            CircuitGate.from_values(
                name=gate.name,
                qubits=tuple(qubit_map.get(qubit, qubit) for qubit in gate.qubits),
                parameters=gate.parameters,
                depth=gate.depth,
                immutable=gate.immutable,
            )
        )

    return CircuitList(
        gates=localized_gates,
        cluster=tuple(range(len(logical_qubits))),
        max_depth=circuit.max_depth,
        max_gates=circuit.max_gates,
    )


def _to_qiskit_circuit(circuit: CircuitList) -> Any:
    if QuantumCircuit is None:
        raise RuntimeError("Qiskit is not available in this environment")

    max_qubit = max((qubit for gate in circuit.gates for qubit in gate.qubits), default=0)
    qc = QuantumCircuit(max_qubit + 1)

    for gate in circuit.gates:
        if not hasattr(qc, gate.name):
            continue
        op = getattr(qc, gate.name)
        op(*gate.parameters, *gate.qubits)

    return qc


def _two_qubit_gate_count(qc: Any) -> int:
    return sum(1 for instruction in qc.data if len(instruction.qubits) == 2)


def _print_transpile_comparison(best_circuit: CircuitList, target: CircuitList, gate_catalog: list[str], target_matrix: np.ndarray) -> None:
    print("\n=== Qiskit Baseline vs Evolved Comparison ===")
    if QuantumCircuit is None or transpile is None or Operator is None:
        print("Qiskit is not installed. Install qiskit to run this comparison.")
        return

    target_qc = _to_qiskit_circuit(target)
    localized_best = _localize_circuit(best_circuit)
    best_qc = _to_qiskit_circuit(localized_best)

    # Baseline: what Qiskit gives when it optimizes the intended target circuit.
    qiskit_baseline = transpile(target_qc, basis_gates=gate_catalog, optimization_level=3)
    # Candidate: your evolved circuit, normalized through the same transpiler settings.
    evolved_transpiled = transpile(best_qc, basis_gates=gate_catalog, optimization_level=3)

    baseline_depth = qiskit_baseline.depth()
    baseline_size = qiskit_baseline.size()
    baseline_2q = _two_qubit_gate_count(qiskit_baseline)

    evolved_depth = evolved_transpiled.depth()
    evolved_size = evolved_transpiled.size()
    evolved_2q = _two_qubit_gate_count(evolved_transpiled)

    depth_delta = evolved_depth - baseline_depth
    size_delta = evolved_size - baseline_size
    two_qubit_delta = evolved_2q - baseline_2q

    print(
        "Qiskit baseline (transpiled target): "
        f"depth={baseline_depth}, size={baseline_size}, 2q={baseline_2q}"
    )
    print(
        "Your evolved (transpiled evolved): "
        f"depth={evolved_depth}, size={evolved_size}, 2q={evolved_2q}"
    )
    print(
        "Delta (evolved - qiskit baseline): "
        f"depth={depth_delta:+d}, size={size_delta:+d}, 2q={two_qubit_delta:+d}"
    )

    baseline_unitary = Operator(qiskit_baseline).data
    evolved_unitary = Operator(evolved_transpiled).data
    baseline_fidelity = fidelity_similarity(baseline_unitary, target_matrix)
    evolved_fidelity = fidelity_similarity(evolved_unitary, target_matrix)
    baseline_frobenius = frobenius_similarity(baseline_unitary, target_matrix)
    evolved_frobenius = frobenius_similarity(evolved_unitary, target_matrix)

    print(
        "Semantic (vs target matrix): "
        f"baseline fidelity={baseline_fidelity:.4f}, evolved fidelity={evolved_fidelity:.4f}; "
        f"baseline frobenius_sim={baseline_frobenius:.4f}, evolved frobenius_sim={evolved_frobenius:.4f}"
    )


def main() -> None:
    noise_model = DemoNoiseModel(qubit_error_rates={0: 0.01, 1: 0.012, 2: 0.02, 3: 0.03})
    gate_catalog = _extract_catalog_from_backend(DemoBackend())
    print(f"Device-derived base gate catalog: {gate_catalog}")

    target = CircuitList(
        gates=[
            _gate("h", (0,), 1),
            _gate("cx", (0, 1), 2),
            _gate("x", (1,), 3),
        ],
        cluster=(0, 1),
        max_depth=4,
        max_gates=6,
    )

    print("=== Entangling Target Circuit ===")
    print_circuit("Target", target)
    target_matrix = circuit_to_matrix(target, target_kind="unitary")
    print_matrix("Target matrix", target_matrix)

    config = CircuitEvolutionConfig(
        device_qubits=4,
        cluster_size=2,
        population_size=25,
        max_generations=20,
        lambda_ratio=2,
        crossover_rate=0.9,
        mutation_rate=0.45,
        tournament_k=3,
        max_depth=6,
        max_gates=10,
        diversity_floor=0.05,
        stagnation_generations=6,
        stagnation_min_delta=1e-6,
        target_fitness_threshold=0.9999,
        behavior_weight=0.9,
        robustness_weight=0.05,
        complexity_weight=0.05,
    )

    print(
        "Run controls: "
        f"max_generations={config.max_generations}, "
        f"diversity_floor={config.diversity_floor}, "
        f"stagnation_generations={config.stagnation_generations}, "
        f"target_fitness_threshold={config.target_fitness_threshold}"
    )
    print(
        "Fitness weights: "
        f"behavior={config.behavior_weight}, "
        f"robustness={config.robustness_weight}, "
        f"complexity={config.complexity_weight}, "
        f"fidelity={config.fidelity_weight}, "
        f"frobenius={config.frobenius_weight}"
    )

    result = evolutionary_best_circuit(
        config=config,
        gate_catalog=gate_catalog,
        target_circuit=None,
        target_matrix=target_matrix,
        noise_model=noise_model,
        seed=321,
    )

    print("\n=== Circuit EA (Per generation diversity) ===")
    for generation, diversity in enumerate(result.diversity_history, start=1):
        print(f"  generation {generation:02d}: diversity={diversity:.3f}")

    print("\n=== Best Evolved Circuit ===")
    print_circuit("Best", result.best_circuit)
    localized_best = _localize_circuit(result.best_circuit)
    evolved_matrix = circuit_to_matrix(localized_best, target_kind="unitary")
    print_matrix("Evolved matrix (localized)", evolved_matrix)

    score = fitness_circuit(
        result.best_circuit,
        target=None,
        target_matrix=target_matrix,
        target_kind="unitary",
        noise_model=noise_model,
    )
    print("\n=== Best Fitness Breakdown (vs entangling target) ===")
    print(f"behavior={score.behavior_score:.4f}")
    print(f"robustness={score.robustness_score:.4f}")
    print(f"complexity={score.complexity_score:.4f}")
    print(f"total={score.total_fitness:.4f}")
    print(f"best fitness={result.best_fitness:.4f}, generations={result.generations_run}")
    print(f"stop reason={result.stop_reason}")
    print(f"cache entries={result.cache_entries}")

    _print_transpile_comparison(result.best_circuit, target, gate_catalog, target_matrix)


if __name__ == "__main__":
    main()
