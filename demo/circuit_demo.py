"""Demo for circuit-level evolutionary search with intermediate outputs."""

from __future__ import annotations

from dataclasses import dataclass

from circuit.entities.circuit_graph import CircuitGraph
from circuit.entities.circuit_list import CircuitGate, CircuitList
from adapters.qiskit import parse_qiskit_backend_gate_catalog
from evolution.circuit.evolutionary import CircuitEvolutionConfig, evolutionary_best_circuit
from evolution.circuit.fitness import fitness_circuit
from evolution.circuit.operators import find_anchor_blocks, mutate_circuit, probabilistic_anchor_crossover


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


def build_parent_circuits() -> tuple[CircuitList, CircuitList]:
    parent1 = CircuitList(
        gates=[
            _gate("rz", (0,), 1, 3.14),
            _gate("sx", (1,), 1),
            _gate("ecr", (0, 1), 2),
            _gate("x", (0,), 3),
        ],
        cluster=(0, 1),
        max_depth=6,
        max_gates=12,
    )
    parent2 = CircuitList(
        gates=[
            _gate("h", (0,), 1),
            _gate("sx", (1,), 1),
            _gate("ecr", (0, 1), 2),
            _gate("z", (1,), 3),
        ],
        cluster=(0, 1),
        max_depth=6,
        max_gates=12,
    )
    return parent1, parent2


def print_circuit(label: str, circuit: CircuitList) -> None:
    print(f"{label}: cluster={circuit.cluster}, gates={len(circuit.gates)}")
    for index, gate in enumerate(circuit.gates, start=1):
        print(
            f"  {index:02d}. name={gate.name} qubits={list(gate.qubits)} "
            f"params={list(gate.parameters)} depth={gate.depth} immutable={gate.immutable}"
        )


def main() -> None:
    noise_model = DemoNoiseModel(qubit_error_rates={0: 0.01, 1: 0.015, 2: 0.02, 3: 0.03})
    gate_catalog = parse_qiskit_backend_gate_catalog(DemoBackend())
    print(f"Device-derived base gate catalog: {gate_catalog}")

    parent1, parent2 = build_parent_circuits()
    print("=== Parents ===")
    print_circuit("Parent A", parent1)
    print_circuit("Parent B", parent2)

    print("\n=== Shared Structure (Anchors) ===")
    anchors = find_anchor_blocks(parent1, parent2)
    for anchor in anchors:
        print(
            f"  block p1[{anchor.start1}:{anchor.end1}] <-> "
            f"p2[{anchor.start2}:{anchor.end2}] weight={anchor.weight:.3f}"
        )

    print("\n=== Crossover Result ===")
    child1, child2 = probabilistic_anchor_crossover(parent1, parent2, crossover_rate=1.0, seed=12)
    print_circuit("Child 1", child1)
    print_circuit("Child 2", child2)

    print("\n=== Mutation Result ===")
    mutated = mutate_circuit(
        child1,
        device_qubits=4,
        mutation_rate=1.0,
        gate_catalog=gate_catalog,
        seed=5,
    )
    print_circuit("Mutated Child 1", mutated)

    print("\n=== Graph Projection ===")
    graph = CircuitGraph.from_circuit_list(mutated)
    print(f"nodes={graph.nodes}")
    for edge in graph.edges:
        print(f"  edge {edge.source} -> {edge.target}, qubit={edge.qubit}, type={edge.edge_type}")

    print("\n=== Fitness Breakdown ===")
    target = CircuitList(gates=[_gate("x", (0,), 1)], cluster=(0, 1), max_depth=3, max_gates=4)
    score = fitness_circuit(mutated, target=target, noise_model=noise_model)
    print(f"behavior={score.behavior_score:.4f}")
    print(f"robustness={score.robustness_score:.4f}")
    print(f"complexity={score.complexity_score:.4f}")
    print(f"total={score.total_fitness:.4f}")

    print("\n=== Circuit EA (Intermediate per generation) ===")
    config = CircuitEvolutionConfig(
        device_qubits=4,
        cluster_size=2,
        population_size=8,
        max_generations=6,
        lambda_ratio=2,
        crossover_rate=0.9,
        mutation_rate=0.4,
        tournament_k=3,
        max_depth=6,
        max_gates=10,
        diversity_floor=0.1,
    )

    result = evolutionary_best_circuit(
        config=config,
        gate_catalog=gate_catalog,
        target_circuit=target,
        noise_model=noise_model,
        seed=123,
    )

    for generation, diversity in enumerate(result.diversity_history, start=1):
        print(f"  generation {generation:02d}: diversity={diversity:.3f}")

    print("\nBest evolved circuit:")
    print_circuit("Best", result.best_circuit)
    print(f"best fitness={result.best_fitness:.4f}, generations={result.generations_run}")


if __name__ == "__main__":
    main()
