from __future__ import annotations

import numpy as np

from circuit.entities.circuit_list import CircuitGate, CircuitList
from evolution.circuit.matrix_eval import circuit_to_matrix, fidelity_similarity, frobenius_similarity


def _gate(name: str, qubits: tuple[int, ...], depth: int) -> CircuitGate:
    return CircuitGate.from_values(name=name, qubits=qubits, depth=depth)


def test_circuit_to_matrix_unitary_shape():
    circuit = CircuitList(gates=[_gate("h", (0,), 1), _gate("cx", (0, 1), 2)], cluster=(0, 1), max_depth=4, max_gates=8)

    matrix = circuit_to_matrix(circuit, target_kind="unitary")

    assert matrix.shape == (4, 4)


def test_similarity_metrics_prefer_identical_matrix():
    matrix_a = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    matrix_b = np.array([[1, 0], [0, 1]], dtype=np.complex128)

    assert fidelity_similarity(matrix_a, matrix_a) > fidelity_similarity(matrix_a, matrix_b)
    assert frobenius_similarity(matrix_a, matrix_a) > frobenius_similarity(matrix_a, matrix_b)
