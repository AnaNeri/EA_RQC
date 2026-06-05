from __future__ import annotations

from circuit.entities.circuit_list import CircuitGate, CircuitList
from evolution.circuit.operators import probabilistic_anchor_crossover, weighted_lcs_indices


def _gate(name: str, qubits: tuple[int, ...]) -> CircuitGate:
    return CircuitGate.from_values(name=name, qubits=qubits)


def _demo_parent_a() -> CircuitList:
    return CircuitList(
        gates=[
            _gate("rz", (0,)),
            _gate("sx", (1,)),
            _gate("ecr", (0, 1)),
            _gate("x", (0,)),
        ],
        cluster=(0, 1),
        max_depth=10,
        max_gates=20,
    )


def _demo_parent_b() -> CircuitList:
    return CircuitList(
        gates=[
            _gate("h", (0,)),
            _gate("sx", (1,)),
            _gate("ecr", (0, 1)),
            _gate("z", (1,)),
        ],
        cluster=(0, 1),
        max_depth=10,
        max_gates=20,
    )


def test_weighted_lcs_detects_shared_structure():
    parent1 = _demo_parent_a()
    parent2 = _demo_parent_b()

    matches = weighted_lcs_indices(parent1, parent2)

    assert len(matches) == 2
    assert matches == [(1, 1), (2, 2)]


def test_probabilistic_anchor_crossover_keeps_common_block_with_high_keep_probability():
    parent1 = _demo_parent_a()
    parent2 = _demo_parent_b()

    child1, child2 = probabilistic_anchor_crossover(
        parent1,
        parent2,
        crossover_rate=1.0,
        base_keep_probability=1.0,
        seed=7,
    )

    child1_signatures = child1.signatures()
    child2_signatures = child2.signatures()
    shared_block = [_gate("sx", (1,)).signature(), _gate("ecr", (0, 1)).signature()]

    assert any(child1_signatures[index : index + 2] == shared_block for index in range(max(0, len(child1_signatures) - 1)))
    assert any(child2_signatures[index : index + 2] == shared_block for index in range(max(0, len(child2_signatures) - 1)))


def test_probabilistic_anchor_crossover_can_skip_when_rate_is_zero():
    parent1 = _demo_parent_a()
    parent2 = _demo_parent_b()

    child1, child2 = probabilistic_anchor_crossover(
        parent1,
        parent2,
        crossover_rate=0.0,
        base_keep_probability=1.0,
        seed=7,
    )

    assert child1.signatures() == parent1.signatures()
    assert child2.signatures() == parent2.signatures()


def test_probabilistic_anchor_crossover_keeps_gate_qubits_inside_child_cluster():
    parent1 = CircuitList(
        gates=[_gate("h", (0,)), _gate("cx", (0, 1)), _gate("x", (1,))],
        cluster=(0, 1),
        max_depth=10,
        max_gates=20,
    )
    parent2 = CircuitList(
        gates=[_gate("h", (2,)), _gate("cx", (2, 3)), _gate("x", (3,))],
        cluster=(2, 3),
        max_depth=10,
        max_gates=20,
    )

    child1, child2 = probabilistic_anchor_crossover(
        parent1,
        parent2,
        crossover_rate=1.0,
        base_keep_probability=1.0,
        seed=11,
    )

    assert child1.cluster == (0, 1)
    assert child2.cluster == (2, 3)
    assert all(qubit in child1.cluster for gate in child1.gates for qubit in gate.qubits)
    assert all(qubit in child2.cluster for gate in child2.gates for qubit in gate.qubits)
