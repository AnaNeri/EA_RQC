"""Demo for circuit EA using an entangling target circuit (includes cx)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from circuit.entities.circuit_list import CircuitGate, CircuitList
from demo.utils import load_fake_kyiv_device_data
from evolution.cluster.evolutionary import evolutionary_best_clusters
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
class DemoCouplingGraph:
    edges: dict[tuple[int, int], float]

    def get_edge_data(self, source: int, target: int, default=None):
        return self.edges.get((source, target), self.edges.get((target, source), default))


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


def print_top_clusters(clusters: list[list[int]], scores: list[float]) -> None:
    for index, cluster in enumerate(clusters, start=1):
        score = scores[index - 1] if index - 1 < len(scores) else 0.0
        print(f"  top-{index}: cluster={cluster}, fitness={score:.6f}")


# demo utilities (backend loader, gate sanitization) moved to demo/utils.py


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
    # Baseline: what Qiskit gives when it optimizes the intended target circuit.
    qiskit_baseline = transpile(target_qc, basis_gates=gate_catalog, optimization_level=3)
    # Note: do NOT transpile the evolved EA circuit. Keep the EA circuit as-is
    # and use its declared metadata for depth/size/2q counts.
    baseline_depth = qiskit_baseline.depth()
    baseline_size = qiskit_baseline.size()
    baseline_2q = _two_qubit_gate_count(qiskit_baseline)

    # Use localized_best (CircuitList) metadata instead of a transpiled qc
    evolved_depth = max((gate.depth for gate in localized_best.gates), default=0)
    evolved_size = len(localized_best.gates)
    evolved_2q = sum(1 for gate in localized_best.gates if len(gate.qubits) == 2)

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

    # IMPORTANT: do NOT transpile the evolved circuit here — compare the
    # Qiskit-transpiled baseline to the EA-produced unitary as-is.
    evolved_unitary = circuit_to_matrix(localized_best, target_kind="unitary")

    # A = Qiskit-transpiled target (what Qiskit would run)
    A = baseline_unitary
    # B = EA-evolved unitary (as produced by the evolution, not transpiled)
    B = evolved_unitary

    # Compare A vs target and B vs target
    a_fid = fidelity_similarity(A, target_matrix)
    b_fid = fidelity_similarity(B, target_matrix)
    a_frob = frobenius_similarity(A, target_matrix)
    b_frob = frobenius_similarity(B, target_matrix)

    print("\nSemantic comparisons to target:")
    print(f" A (transpiled target)    -> fidelity={a_fid:.4f}, frobenius_sim={a_frob:.4f}")
    print(f" B (EA-evolved unitary) -> fidelity={b_fid:.4f}, frobenius_sim={b_frob:.4f}")

    # Indicate which is better for each metric
    if a_fid > b_fid:
        fid_winner = "A (transpiled target)"
    elif b_fid > a_fid:
        fid_winner = "B (EA-evolved unitary)"
    else:
        fid_winner = "tie"

    if a_frob > b_frob:
        frob_winner = "A (transpiled target)"
    elif b_frob > a_frob:
        frob_winner = "B (EA-evolved unitary)"
    else:
        frob_winner = "tie"

    print(f" Better (fidelity): {fid_winner}")
    print(f" Better (frobenius): {frob_winner}")


def run_experiment(
    device_qubits: int = 1000,
    mapping_generations: int = 8,
    mapping_population: int = 12,
    circuit_generations: int = 50,
    circuit_population: int = 30,
):
    """Run the entangling demo and return structured results for CLI/UI."""
    noise_model, gate_catalog, backend_name = load_fake_kyiv_device_data(device_qubits=device_qubits)

    coupling_graph = DemoCouplingGraph(
        edges={(0, 1): 0.002, (1, 2): 0.003, (2, 3): 0.004, (0, 2): 0.007, (1, 3): 0.009}
    )
    mapping = evolutionary_best_clusters(
        device_qubits=device_qubits,
        target_qubits=2,
        max_generation=mapping_generations,
        population=mapping_population,
        noise_model=noise_model,
        type_ranking="linear",
        lambda_ratio=2,
        crossover_rate=0.85,
        mutation_rate=0.2,
        coupling_graph=coupling_graph,
        prioritize="both",
        top_k=3,
        seed=321,
    )

    candidate_clusters = tuple(tuple(cluster) for cluster in mapping.top_clusters)
    candidate_cluster_weights = tuple(mapping.top_fitness_scores)
    best_cluster = tuple(mapping.best_cluster) if mapping.best_cluster else (0, 1)

    target = CircuitList(
        gates=[_gate("h", (best_cluster[0],), 1), _gate("cx", (best_cluster[0], best_cluster[1]), 2), _gate("x", (best_cluster[1],), 3)],
        cluster=best_cluster,
        max_depth=4,
        max_gates=6,
    )

    localized_target = _localize_circuit(target)
    target_matrix = circuit_to_matrix(localized_target, target_kind="unitary")

    config = CircuitEvolutionConfig(
        device_qubits=device_qubits,
        cluster_size=2,
        population_size=circuit_population,
        max_generations=circuit_generations,
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
        candidate_clusters=candidate_clusters,
        candidate_cluster_weights=candidate_cluster_weights,
        coupling_graph=coupling_graph,
        enforce_cluster_connectivity=True,
    )

    ea_result = evolutionary_best_circuit(
        config=config,
        gate_catalog=gate_catalog,
        target_circuit=None,
        target_matrix=target_matrix,
        noise_model=noise_model,
        seed=321,
    )

    fitness = fitness_circuit(
        ea_result.best_circuit,
        target=None,
        target_matrix=target_matrix,
        target_kind="unitary",
        noise_model=noise_model,
    )

    return {
        "noise_model": noise_model,
        "gate_catalog": gate_catalog,
        "backend_name": backend_name,
        "mapping": mapping,
        "target": target,
        "localized_target": localized_target,
        "target_matrix": target_matrix,
        "ea_result": ea_result,
        "fitness": fitness,
    }


def main() -> None:
    # For CLI/main, run through the structured run_experiment and print results
    results = run_experiment()
    print(f"Noise model source: {results.get('backend_name')}")
    print(f"Device-derived base gate catalog: {results.get('gate_catalog')}")

    mapping = results.get("mapping")
    print("\n=== Cluster Mapping (Top-3) ===")
    print_top_clusters(mapping.top_clusters, mapping.top_fitness_scores)

    target = results.get("target")
    print("=== Entangling Target Circuit ===")
    print_circuit("Target", target)
    localized_target = results.get("localized_target")
    target_matrix = results.get("target_matrix")
    print_matrix("Target matrix", target_matrix)

    result = results.get("ea_result")
    print("\n=== Circuit EA (Per generation diversity) ===")
    for generation, diversity in enumerate(result.diversity_history, start=1):
        print(f"  generation {generation:02d}: diversity={diversity:.3f}")

    print("\n=== Best Evolved Circuit ===")
    print_circuit("Best", result.best_circuit)
    localized_best = _localize_circuit(result.best_circuit)
    evolved_matrix = circuit_to_matrix(localized_best, target_kind="unitary")
    print_matrix("Evolved matrix (localized)", evolved_matrix)

    score = results.get("fitness")
    print("\n=== Best Fitness Breakdown (vs entangling target) ===")
    print(f"behavior={score.behavior_score:.4f}")
    print(f"robustness={score.robustness_score:.4f}")
    print(f"complexity={score.complexity_score:.4f}")
    print(f"total={score.total_fitness:.4f}")
    print(f"best fitness={result.best_fitness:.4f}, generations={result.generations_run}")
    print(f"stop reason={result.stop_reason}")
    print(f"cache entries={result.cache_entries}")

    # Use the localized target for transpiler comparisons (keeps qubit indices small)
    _print_transpile_comparison(result.best_circuit, localized_target, results.get("gate_catalog"), target_matrix)


if __name__ == "__main__":
    main()
