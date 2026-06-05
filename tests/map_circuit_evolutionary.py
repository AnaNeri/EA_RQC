from __future__ import annotations

from dataclasses import dataclass

from circuit.entities.circuit_list import CircuitGate, CircuitList
from evolution.circuit.evolutionary import CircuitEvolutionConfig, evolutionary_best_circuit
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
