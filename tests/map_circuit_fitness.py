from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from circuit.entities.circuit_list import CircuitGate, CircuitList
from evolution.circuit.fitness import fitness_circuit
from evolution.circuit.matrix_eval import circuit_to_matrix
from evolution.circuit.targets import SampleTarget, StateVectorSample


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
    target_matrix = circuit_to_matrix(target, target_kind="unitary")
    noise = FakeNoiseModel(qubit_error_rates={0: 0.01, 1: 0.03})

    breakdown = fitness_circuit(circuit, target=None, target_matrix=target_matrix, target_kind="unitary", noise_model=noise)

    assert 0.0 <= breakdown.behavior_score <= 1.0
    assert 0.0 <= breakdown.robustness_score <= 1.0
    assert 0.0 <= breakdown.complexity_score <= 1.0
    assert 0.0 <= breakdown.total_fitness <= 1.0


def test_fitness_circuit_prefers_target_match():
    target = CircuitList(gates=[_gate("x", (0,), 1)], cluster=(0, 1), max_depth=3, max_gates=4)
    perfect = CircuitList(gates=[_gate("x", (0,), 1)], cluster=(0, 1), max_depth=3, max_gates=4)
    mismatch = CircuitList(gates=[_gate("h", (0,), 1), _gate("z", (1,), 2)], cluster=(0, 1), max_depth=3, max_gates=4)
    target_matrix = circuit_to_matrix(target, target_kind="unitary")
    noise = FakeNoiseModel(qubit_error_rates={0: 0.02, 1: 0.02})

    fit_perfect = fitness_circuit(perfect, target=None, target_matrix=target_matrix, target_kind="unitary", noise_model=noise)
    fit_mismatch = fitness_circuit(mismatch, target=None, target_matrix=target_matrix, target_kind="unitary", noise_model=noise)

    assert fit_perfect.total_fitness > fit_mismatch.total_fitness


def test_fitness_circuit_ignores_ancilla_with_working_qubits():
    target = CircuitList(
        gates=[_gate("x", (0,), 1)],
        cluster=(0, 1),
        max_depth=4,
        max_gates=6,
    )
    candidate_clean = CircuitList(
        gates=[_gate("x", (0,), 1)],
        cluster=(0, 1),
        max_depth=4,
        max_gates=6,
    )
    candidate_with_ancilla_change = CircuitList(
        gates=[_gate("x", (0,), 1), _gate("z", (1,), 2)],
        cluster=(0, 1),
        max_depth=4,
        max_gates=6,
    )
    target_matrix = circuit_to_matrix(target, target_kind="unitary")
    noise = FakeNoiseModel(qubit_error_rates={0: 0.0, 1: 0.0})

    fit_clean = fitness_circuit(
        candidate_clean,
        target=None,
        target_matrix=target_matrix,
        target_kind="unitary",
        noise_model=noise,
        behavior_weight=1.0,
        robustness_weight=0.0,
        complexity_weight=0.0,
        working_qubits=(0,),
    )
    fit_with_ancilla = fitness_circuit(
        candidate_with_ancilla_change,
        target=None,
        target_matrix=target_matrix,
        target_kind="unitary",
        noise_model=noise,
        behavior_weight=1.0,
        robustness_weight=0.0,
        complexity_weight=0.0,
        working_qubits=(0,),
    )

    assert abs(fit_clean.behavior_score - fit_with_ancilla.behavior_score) < 1e-9


def test_sample_score_ignores_ancilla_with_working_qubits():
    # Target operation on working qubit 0 is identity on one qubit.
    sample_target = SampleTarget(
        samples=(
            StateVectorSample(
                input_state=np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128),
                output_state=np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128),
                label="|0>",
            ),
            StateVectorSample(
                input_state=np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128),
                output_state=np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128),
                label="|1>",
            ),
        ),
        num_qubits=1,
    )

    clean = CircuitList(
        gates=[_gate("id", (0,), 1)],
        cluster=(0, 1),
        max_depth=4,
        max_gates=6,
    )
    ancilla_perturbed = CircuitList(
        gates=[_gate("id", (0,), 1), _gate("z", (1,), 2)],
        cluster=(0, 1),
        max_depth=4,
        max_gates=6,
    )
    noise = FakeNoiseModel(qubit_error_rates={0: 0.0, 1: 0.0})

    fit_clean = fitness_circuit(
        clean,
        target=None,
        noise_model=noise,
        behavior_weight=0.0,
        robustness_weight=0.0,
        complexity_weight=0.0,
        samples=sample_target,
        sample_weight=1.0,
        working_qubits=(0,),
    )
    fit_ancilla = fitness_circuit(
        ancilla_perturbed,
        target=None,
        noise_model=noise,
        behavior_weight=0.0,
        robustness_weight=0.0,
        complexity_weight=0.0,
        samples=sample_target,
        sample_weight=1.0,
        working_qubits=(0,),
    )

    assert abs(fit_clean.sample_score - fit_ancilla.sample_score) < 1e-9
