"""Demo for circuit-level evolutionary search with intermediate outputs."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from circuit.entities.circuit_graph import CircuitGraph
from circuit.entities.circuit_list import CircuitGate, CircuitList
from demo.utils import DemoNoiseModel, load_fake_kyiv_device_data
from evolution.cluster.evolutionary import evolutionary_best_clusters
from evolution.circuit.evolutionary import CircuitEvolutionConfig, evolutionary_best_circuit
from evolution.circuit.fitness import fitness_circuit
from evolution.circuit.operators import find_anchor_blocks, mutate_circuit, probabilistic_anchor_crossover


@dataclass(frozen=True)
class DemoCouplingGraph:
    edges: dict[tuple[int, int], float]

    def get_edge_data(self, source: int, target: int, default=None):
        return self.edges.get((source, target), self.edges.get((target, source), default))


# demo utilities (backend loader, gate sanitization) moved to demo/utils.py


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


def print_top_clusters(clusters: list[list[int]], scores: list[float]) -> None:
    for index, cluster in enumerate(clusters, start=1):
        score = scores[index - 1] if index - 1 < len(scores) else 0.0
        print(f"  top-{index}: cluster={cluster}, fitness={score:.6f}")


def main() -> None:
    # Delegate to run_experiment() for structured results and keep CLI behavior
    results = run_experiment()
    # Print same human-readable outputs for CLI
    noise_model = results.get("noise_model")
    gate_catalog = results.get("gate_catalog")
    mapping = results.get("mapping")
    print(f"Noise model source: {results.get('backend_name')}")
    print(f"Device-derived base gate catalog: {gate_catalog}")
    print("\n=== Cluster Mapping (Top-3) ===")
    print_top_clusters(mapping.top_clusters, mapping.top_fitness_scores)

    parent1 = results.get("parent1")
    parent2 = results.get("parent2")
    print("=== Parents ===")
    print_circuit("Parent A", parent1)
    print_circuit("Parent B", parent2)

    print("\n=== Shared Structure (Anchors) ===")
    anchors = results.get("anchors") or []
    for anchor in anchors:
        print(
            f"  block p1[{anchor.start1}:{anchor.end1}] <-> "
            f"p2[{anchor.start2}:{anchor.end2}] weight={anchor.weight:.3f}"
        )

    print("\n=== Crossover Result ===")
    child1 = results.get("child1")
    child2 = results.get("child2")
    print_circuit("Child 1", child1)
    print_circuit("Child 2", child2)

    print("\n=== Mutation Result ===")
    mutated = results.get("mutated")
    print_circuit("Mutated Child 1", mutated)

    print("\n=== Graph Projection ===")
    graph = CircuitGraph.from_circuit_list(mutated)
    print(f"nodes={graph.nodes}")
    for edge in graph.edges:
        print(f"  edge {edge.source} -> {edge.target}, qubit={edge.qubit}, type={edge.edge_type}")

    print("\n=== Fitness Breakdown ===")
    score = results.get("fitness")
    print(f"behavior={score.behavior_score:.4f}")
    print(f"robustness={score.robustness_score:.4f}")
    print(f"complexity={score.complexity_score:.4f}")
    print(f"total={score.total_fitness:.4f}")

    print("\n=== Circuit EA (Intermediate per generation) ===")
    result = results.get("ea_result")
    for generation, diversity in enumerate(result.diversity_history, start=1):
        print(f"  generation {generation:02d}: diversity={diversity:.3f}")

    print("\nBest evolved circuit:")
    print_circuit("Best", result.best_circuit)
    print(f"best fitness={result.best_fitness:.4f}, generations={result.generations_run}")


def run_experiment(
    device_qubits: int = 8,
    mapping_population: int = 8,
    mapping_generations: int = 4,
    circuit_population: int = 10,
    circuit_generations: int = 6,
):
    """Run the demo experiment and return structured results for CLI or UI.

    The returned dict contains: noise_model, gate_catalog, backend_name,
    mapping, parent1, parent2, anchors, child1, child2, mutated, fitness, ea_result
    """
    noise_model, gate_catalog, backend_name = load_fake_kyiv_device_data(device_qubits=device_qubits)

    coupling_graph = DemoCouplingGraph(edges={(0, 1): 0.002, (1, 2): 0.004, (0, 2): 0.008})
    mapping = evolutionary_best_clusters(
        device_qubits=device_qubits,
        target_qubits=2,
        max_generation=mapping_generations,
        population=mapping_population,
        noise_model=noise_model,
        type_ranking="linear",
        lambda_ratio=2,
        crossover_rate=0.8,
        mutation_rate=0.2,
        coupling_graph=coupling_graph,
        prioritize="both",
        top_k=3,
        seed=42,
    )

    candidate_clusters = tuple(tuple(cluster) for cluster in mapping.top_clusters)
    candidate_cluster_weights = tuple(mapping.top_fitness_scores)

    parent1, parent2 = build_parent_circuits()

    anchors = find_anchor_blocks(parent1, parent2)

    child1, child2 = probabilistic_anchor_crossover(parent1, parent2, crossover_rate=1.0, seed=12)

    mutated = mutate_circuit(
        child1,
        device_qubits=device_qubits,
        mutation_rate=1.0,
        gate_catalog=gate_catalog,
        candidate_clusters=candidate_clusters,
        candidate_cluster_weights=candidate_cluster_weights,
        coupling_graph=coupling_graph,
        seed=5,
    )

    # Use a localized target (logical qubit 0 within a two-qubit cluster)
    target = CircuitList(gates=[_gate("x", (0,), 1)], cluster=(0, 1), max_depth=3, max_gates=4)
    score = fitness_circuit(mutated, target=target, noise_model=noise_model)

    config = CircuitEvolutionConfig(
        device_qubits=device_qubits,
        cluster_size=2,
        population_size=circuit_population,
        max_generations=circuit_generations,
        lambda_ratio=2,
        crossover_rate=0.9,
        mutation_rate=0.4,
        tournament_k=3,
        max_depth=6,
        max_gates=10,
        diversity_floor=0.1,
        candidate_clusters=candidate_clusters,
        candidate_cluster_weights=candidate_cluster_weights,
        coupling_graph=coupling_graph,
        enforce_cluster_connectivity=False,
    )

    ea_result = evolutionary_best_circuit(
        config=config,
        gate_catalog=gate_catalog,
        target_circuit=target,
        noise_model=noise_model,
        seed=123,
    )

    return {
        "noise_model": noise_model,
        "gate_catalog": gate_catalog,
        "backend_name": backend_name,
        "mapping": mapping,
        "parent1": parent1,
        "parent2": parent2,
        "anchors": anchors,
        "child1": child1,
        "child2": child2,
        "mutated": mutated,
        "fitness": score,
        "ea_result": ea_result,
    }





if __name__ == "__main__":
    main()
