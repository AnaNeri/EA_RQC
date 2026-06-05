from __future__ import annotations

from dataclasses import dataclass

from circuit.entities.circuit_list import CircuitGate, CircuitList
from evolution.circuit.fitness import fitness_circuit


@dataclass(frozen=True)
class FakeNoiseModel:
    qubit_error_rates: dict[int, float]


def _gate(name: str, qubits: tuple[int, ...], depth: int, param: float | None = None) -> CircuitGate:
    params = (param,) if param is not None else ()
    return CircuitGate.from_values(name=name, qubits=qubits, parameters=params, depth=depth)


def test_fitness_circuit_returns_bounded_values():
    circuit = CircuitList(
        gates=[_gate("x", (0,), 1), _gate("ecr", (0, 1), 2)],
        cluster=(0, 1),
        max_depth=6,
        max_gates=8,
    )
    target = CircuitList(gates=[_gate("x", (0,), 1)], cluster=(0, 1), max_depth=3, max_gates=4)
    noise = FakeNoiseModel(qubit_error_rates={0: 0.01, 1: 0.03})

    breakdown = fitness_circuit(circuit, target=target, noise_model=noise)

    assert 0.0 <= breakdown.behavior_score <= 1.0
    assert 0.0 <= breakdown.robustness_score <= 1.0
    assert 0.0 <= breakdown.complexity_score <= 1.0
    assert 0.0 <= breakdown.total_fitness <= 1.0


def test_fitness_circuit_prefers_target_match():
    target = CircuitList(gates=[_gate("x", (0,), 1)], cluster=(0, 1), max_depth=3, max_gates=4)
    perfect = CircuitList(gates=[_gate("x", (0,), 1)], cluster=(0, 1), max_depth=3, max_gates=4)
    mismatch = CircuitList(gates=[_gate("h", (0,), 1), _gate("z", (1,), 2)], cluster=(0, 1), max_depth=3, max_gates=4)
    noise = FakeNoiseModel(qubit_error_rates={0: 0.02, 1: 0.02})

    fit_perfect = fitness_circuit(perfect, target=target, noise_model=noise)
    fit_mismatch = fitness_circuit(mismatch, target=target, noise_model=noise)

    assert fit_perfect.total_fitness > fit_mismatch.total_fitness
