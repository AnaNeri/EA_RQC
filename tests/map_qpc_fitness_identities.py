from __future__ import annotations

from dataclasses import dataclass

from circuit.entities.circuit_list import CircuitGate, CircuitList
from evolution.circuit.fitness import fitness_circuit
from evolution.circuit.matrix_eval import circuit_to_matrix, mix_channels


@dataclass(frozen=True)
class FakeNoiseModel:
    qubit_error_rates: dict[int, float]


def _gate(name: str, qubits: tuple[int, ...], depth: int) -> CircuitGate:
    return CircuitGate.from_values(name=name, qubits=qubits, depth=depth)


def _behavior_only_fitness(
    circuit: CircuitList,
    *,
    target_matrix,
    target_kind: str,
    noise: FakeNoiseModel,
    qpc_enabled: bool,
) -> float:
    breakdown = fitness_circuit(
        circuit,
        target=None,
        target_matrix=target_matrix,
        target_kind=target_kind,
        noise_model=noise,
        behavior_weight=1.0,
        robustness_weight=0.0,
        complexity_weight=0.0,
        qpc_enabled=qpc_enabled,
        qpc_mode="exact",
    )
    return breakdown.behavior_score


def test_qpc_exact_zero_noise_matches_non_qpc_behavior() -> None:
    circuit = CircuitList(gates=[_gate("x", (0,), 1)], cluster=(0,), max_depth=2, max_gates=2)
    target_unitary = circuit_to_matrix(circuit, target_kind="unitary")
    noise = FakeNoiseModel(qubit_error_rates={0: 0.0})

    score_non_qpc = _behavior_only_fitness(
        circuit,
        target_matrix=target_unitary,
        target_kind="unitary",
        noise=noise,
        qpc_enabled=False,
    )
    score_qpc_exact = _behavior_only_fitness(
        circuit,
        target_matrix=target_unitary,
        target_kind="unitary",
        noise=noise,
        qpc_enabled=True,
    )

    assert abs(score_qpc_exact - score_non_qpc) < 1e-12
    assert score_qpc_exact > 1.0 - 1e-12


def test_qpc_exact_unique_behavior_identity_with_channel_target() -> None:
    circuit = CircuitList(gates=[_gate("h", (0,), 1)], cluster=(0,), max_depth=2, max_gates=2)
    ideal_channel = circuit_to_matrix(circuit, target_kind="channel")
    mixed_same_channel = mix_channels(ideal_channel, ideal_channel, p=0.37)
    noise = FakeNoiseModel(qubit_error_rates={0: 0.0})

    score_vs_ideal = _behavior_only_fitness(
        circuit,
        target_matrix=ideal_channel,
        target_kind="channel",
        noise=noise,
        qpc_enabled=True,
    )
    score_vs_mixed = _behavior_only_fitness(
        circuit,
        target_matrix=mixed_same_channel,
        target_kind="channel",
        noise=noise,
        qpc_enabled=True,
    )

    assert abs(score_vs_mixed - score_vs_ideal) < 1e-12
    assert score_vs_mixed > 1.0 - 1e-12


def test_qpc_exact_rearrangement_identity_with_channel_target() -> None:
    candidate = CircuitList(gates=[_gate("x", (0,), 1)], cluster=(0,), max_depth=2, max_gates=2)
    other = CircuitList(gates=[_gate("z", (0,), 1)], cluster=(0,), max_depth=2, max_gates=2)

    u = circuit_to_matrix(candidate, target_kind="channel")
    v = circuit_to_matrix(other, target_kind="channel")
    p = 0.23
    target_a = mix_channels(u, v, p)
    target_b = mix_channels(v, u, 1.0 - p)
    noise = FakeNoiseModel(qubit_error_rates={0: 0.0})

    score_a = _behavior_only_fitness(
        candidate,
        target_matrix=target_a,
        target_kind="channel",
        noise=noise,
        qpc_enabled=True,
    )
    score_b = _behavior_only_fitness(
        candidate,
        target_matrix=target_b,
        target_kind="channel",
        noise=noise,
        qpc_enabled=True,
    )

    assert abs(score_a - score_b) < 1e-12