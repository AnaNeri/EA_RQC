from __future__ import annotations

from dataclasses import dataclass
from random import Random

from circuit.entities.circuit_list import CircuitGate, CircuitList
from evolution.circuit.evolutionary import (
    CircuitEvolutionConfig,
    _select_mu_lambda_survivors,
    _tournament_pick,
    evolutionary_best_circuit,
)
from evolution.circuit.matrix_eval import circuit_to_matrix
from evolution.circuit.operators import mutate_circuit


@dataclass(frozen=True)
class FakeNoiseModel:
    qubit_error_rates: dict[int, float]


def _gate(name: str, qubits: tuple[int, ...], depth: int) -> CircuitGate:
    return CircuitGate.from_values(name=name, qubits=qubits, depth=depth)


def test_mutation_preserves_size_constraints():
    circuit = CircuitList(
        gates=[_gate("x", (0,), 1), _gate("sx", (1,), 2)],
        cluster=(0, 1),
        max_depth=4,
        max_gates=3,
    )

    mutated = mutate_circuit(
        circuit,
        device_qubits=4,
        mutation_rate=1.0,
        gate_catalog=["x", "sx", "rz", "ecr"],
        seed=42,
    )

    assert len(mutated.gates) <= 3
    assert all(gate.depth <= 4 for gate in mutated.gates)


def test_tournament_pick_uses_fitness_weights():
    population = [
        CircuitList(gates=[_gate("x", (0,), 1)], cluster=(0, 1), max_depth=4, max_gates=4),
        CircuitList(gates=[_gate("h", (0,), 1)], cluster=(0, 1), max_depth=4, max_gates=4),
    ]
    fitness_scores = [0.1, 0.9]

    selections = []
    for seed in range(50):
        chosen = _tournament_pick(population, fitness_scores, k=2, rng=Random(seed))
        selections.append(chosen)

    assert selections.count(population[1]) > selections.count(population[0])


def test_mu_lambda_survivors_keep_only_top_offspring():
    children = [
        CircuitList(gates=[_gate("x", (0,), 1)], cluster=(0, 1), max_depth=4, max_gates=4),
        CircuitList(gates=[_gate("h", (0,), 1)], cluster=(0, 1), max_depth=4, max_gates=4),
        CircuitList(gates=[_gate("sx", (0,), 1)], cluster=(0, 1), max_depth=4, max_gates=4),
    ]
    child_breakdowns = [
        type("Breakdown", (), {"total_fitness": 0.2})(),
        type("Breakdown", (), {"total_fitness": 0.9})(),
        type("Breakdown", (), {"total_fitness": 0.5})(),
    ]

    survivors = _select_mu_lambda_survivors(children, child_breakdowns, mu=2)

    assert survivors == [children[1], children[2]]


def test_evolutionary_best_circuit_runs_and_returns_ranked_population():
    config = CircuitEvolutionConfig(
        device_qubits=5,
        cluster_size=2,
        population_size=6,
        max_generations=4,
        lambda_ratio=2,
        crossover_rate=0.9,
        mutation_rate=0.3,
        tournament_k=3,
        max_depth=5,
        max_gates=8,
        diversity_floor=0.0,
    )

    target = CircuitList(
        gates=[_gate("x", (0,), 1)],
        cluster=(0, 1),
        max_depth=5,
        max_gates=8,
    )

    result = evolutionary_best_circuit(
        config=config,
        gate_catalog=["x", "y", "z", "h", "sx", "rz", "cx", "ecr"],
        target_circuit=target,
        noise_model=FakeNoiseModel(qubit_error_rates={0: 0.01, 1: 0.02, 2: 0.03, 3: 0.04, 4: 0.02}),
        seed=99,
    )

    assert result.generations_run <= 4
    assert len(result.population) == 6
    assert len(result.fitness_scores) == 6
    assert 0.0 <= result.best_fitness <= 1.0
    assert len(result.diversity_history) >= 1


def test_evolutionary_best_circuit_accepts_matrix_target():
    config = CircuitEvolutionConfig(
        device_qubits=4,
        cluster_size=2,
        population_size=6,
        max_generations=3,
        lambda_ratio=2,
        crossover_rate=0.9,
        mutation_rate=0.4,
        tournament_k=3,
        max_depth=5,
        max_gates=8,
        diversity_floor=0.0,
        target_kind="unitary",
    )

    target = CircuitList(
        gates=[_gate("h", (0,), 1), _gate("cx", (0, 1), 2)],
        cluster=(0, 1),
        max_depth=5,
        max_gates=8,
    )
    target_matrix = circuit_to_matrix(target, target_kind="unitary")

    result = evolutionary_best_circuit(
        config=config,
        gate_catalog=["x", "h", "sx", "rz", "cx", "ecr"],
        target_circuit=None,
        target_matrix=target_matrix,
        noise_model=FakeNoiseModel(qubit_error_rates={0: 0.01, 1: 0.02, 2: 0.02, 3: 0.03}),
        seed=17,
    )

    assert len(result.population) == 6
    assert 0.0 <= result.best_fitness <= 1.0


def test_evolutionary_best_circuit_reports_stagnation_stop_reason():
    config = CircuitEvolutionConfig(
        device_qubits=2,
        cluster_size=1,
        population_size=1,
        max_generations=10,
        lambda_ratio=1,
        crossover_rate=0.0,
        mutation_rate=0.0,
        tournament_k=1,
        max_depth=1,
        max_gates=1,
        diversity_floor=0.0,
        stagnation_generations=1,
        target_fitness_threshold=1.1,
    )

    target = CircuitList(
        gates=[_gate("x", (0,), 1)],
        cluster=(0,),
        max_depth=1,
        max_gates=1,
    )

    result = evolutionary_best_circuit(
        config=config,
        gate_catalog=["x"],
        target_circuit=target,
        noise_model=FakeNoiseModel(qubit_error_rates={0: 0.01, 1: 0.02}),
        seed=5,
    )

    assert result.stop_reason == "stagnation"
    assert result.generations_run < config.max_generations


def test_evolutionary_best_circuit_respects_candidate_clusters():
    config = CircuitEvolutionConfig(
        device_qubits=4,
        cluster_size=2,
        population_size=8,
        max_generations=4,
        lambda_ratio=2,
        crossover_rate=0.8,
        mutation_rate=0.5,
        tournament_k=3,
        max_depth=5,
        max_gates=8,
        diversity_floor=0.0,
        candidate_clusters=((0, 1), (1, 2), (2, 3)),
        candidate_cluster_weights=(0.9, 0.5, 0.3),
    )

    target = CircuitList(
        gates=[_gate("h", (0,), 1), _gate("cx", (0, 1), 2)],
        cluster=(0, 1),
        max_depth=5,
        max_gates=8,
    )

    result = evolutionary_best_circuit(
        config=config,
        gate_catalog=["h", "x", "rz", "cx", "ecr"],
        target_circuit=target,
        noise_model=FakeNoiseModel(qubit_error_rates={0: 0.01, 1: 0.02, 2: 0.03, 3: 0.04}),
        seed=11,
    )

    allowed = {tuple(cluster) for cluster in config.candidate_clusters}
    assert all(tuple(circuit.cluster) in allowed for circuit in result.population)
