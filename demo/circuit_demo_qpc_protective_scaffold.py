"""Demo: repetition-code-inspired protective wrapper for a logical single-qubit workload.

This is a standalone second demo that does not modify core library code.
It seeds evolution with a simple coherent repetition-style scaffold and lets
EA refine around it under QPC-aware fitness.

Note:
- This is not a full measurement-based QEC cycle.
- It is a unitary, syndrome-like approximation suitable for the current
  simulator/evolution pipeline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from circuit.entities.circuit_list import CircuitGate, CircuitList
from evolution.circuit.evolutionary import CircuitEvolutionConfig, evolutionary_best_circuit
from evolution.circuit.fitness import fitness_circuit
from evolution.circuit.matrix_eval import (
    apply_channel_to_state,
    circuit_to_faulty_channel_model,
    circuit_to_matrix,
    partial_trace_density_matrix,
    state_fidelity,
    subsystem_similarity,
    unitary_to_superoperator,
)


@dataclass(frozen=True)
class DemoNoiseModel:
    qubit_error_rates: dict[int, float]
    gate_error_rates: dict[str, float] | None = None


@dataclass(frozen=True)
class RecoveryConfig:
    mode: str
    syndrome_error_rate: float
    readout_error_rate: float
    confidence_threshold: float = 0.75
    syndrome_qubits: tuple[int, ...] = (1, 2, 3, 4)


@dataclass(frozen=True)
class DemoPreset:
    population_size: int
    max_generations: int
    max_depth: int
    max_gates: int
    qpc_mode: str
    gate_catalog: tuple[str, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repetition-code-inspired QPC demo with speed/quality presets."
    )
    parser.add_argument(
        "--preset",
        choices=("micro", "smoke", "fast", "balanced", "thorough"),
        default="fast",
        help="Execution preset. 'fast' is recommended for quick iteration.",
    )
    parser.add_argument(
        "--scaffold",
        choices=("surface-mini", "repetition"),
        default="surface-mini",
        help="Initial hand-crafted structure used to seed EA.",
    )
    parser.add_argument(
        "--skip-evolution",
        action="store_true",
        help="Only compute workload/scaffold scores and skip EA loop.",
    )
    parser.add_argument(
        "--noise-profile",
        choices=("uniform", "x-only"),
        default="uniform",
        help="Noise profile: uniform applies to all gates, x-only applies only on X gates.",
    )
    parser.add_argument(
        "--recovery",
        choices=("off", "ideal", "noisy"),
        default="off",
        help="Apply an explicit post-wrapper CPTP recovery map.",
    )
    parser.add_argument(
        "--syndrome-error-rate",
        type=float,
        default=0.02,
        help="Syndrome processing error rate for recovery mode 'noisy'.",
    )
    parser.add_argument(
        "--readout-error-rate",
        type=float,
        default=0.02,
        help="Syndrome readout bit-flip rate for recovery mode 'noisy'.",
    )
    parser.add_argument(
        "--recovery-confidence-threshold",
        type=float,
        default=0.75,
        help="Only apply recovery when syndrome confidence exceeds this threshold.",
    )
    return parser.parse_args()


def _resolve_preset(name: str) -> DemoPreset:
    if name == "micro":
        return DemoPreset(
            population_size=8,
            max_generations=4,
            max_depth=4,
            max_gates=10,
            qpc_mode="approx",
            gate_catalog=("id", "x", "h", "cx"),
        )
    if name == "smoke":
        return DemoPreset(
            population_size=6,
            max_generations=2,
            max_depth=6,
            max_gates=8,
            qpc_mode="approx",
            gate_catalog=("id", "x", "h", "cx", "cz"),
        )
    if name == "fast":
        return DemoPreset(
            population_size=12,
            max_generations=6,
            max_depth=8,
            max_gates=12,
            qpc_mode="approx",
            gate_catalog=("id", "x", "h", "sx", "rz", "cx", "cz"),
        )
    if name == "thorough":
        return DemoPreset(
            population_size=28,
            max_generations=24,
            max_depth=12,
            max_gates=20,
            qpc_mode="exact",
            gate_catalog=("id", "x", "h", "sx", "rz", "cx", "cz", "ecr"),
        )
    return DemoPreset(
        population_size=18,
        max_generations=12,
        max_depth=10,
        max_gates=16,
        qpc_mode="approx",
        gate_catalog=("id", "x", "h", "sx", "rz", "cx", "cz"),
    )


def _gate(name: str, qubits: tuple[int, ...], depth: int, param: float | None = None) -> CircuitGate:
    params = (param,) if param is not None else ()
    return CircuitGate.from_values(name=name, qubits=qubits, parameters=params, depth=depth)


def _print_circuit(label: str, circuit: CircuitList) -> None:
    print(f"{label}: cluster={circuit.cluster}, gates={len(circuit.gates)}")
    for gate in circuit.gates:
        params = f" params={tuple(round(value, 4) for value in gate.parameters)}" if gate.parameters else ""
        print(f"  d={gate.depth:02d} {gate.name.upper():>3} q={gate.qubits}{params}")


def _logical_x_full_unitary(*, num_qubits: int, logical_qubit: int = 0) -> np.ndarray:
    x_only = CircuitList(
        gates=[_gate("x", (logical_qubit,), 1)],
        cluster=tuple(range(num_qubits)),
        max_depth=2,
        max_gates=2,
    )
    return circuit_to_matrix(x_only, target_kind="unitary", num_qubits=num_qubits)


def _apply_recovery_cptp_map(
    *,
    rho_in: np.ndarray,
    num_qubits: int,
    recovery: RecoveryConfig,
    logical_qubit: int = 0,
) -> np.ndarray:
    if recovery.mode == "off":
        return rho_in

    if recovery.mode == "ideal":
        syndrome_error_rate = 0.0
        readout_error_rate = 0.0
    else:
        syndrome_error_rate = max(0.0, min(1.0, float(recovery.syndrome_error_rate)))
        readout_error_rate = max(0.0, min(1.0, float(recovery.readout_error_rate)))

    syndrome_qubits = tuple(sorted(int(q) for q in recovery.syndrome_qubits))
    if not syndrome_qubits:
        return rho_in

    x_full = _logical_x_full_unitary(num_qubits=num_qubits, logical_qubit=logical_qubit)
    x_dag = x_full.conjugate().T

    dim = 2**num_qubits
    num_syndrome = len(syndrome_qubits)
    threshold = (num_syndrome // 2) + 1
    confidence_threshold = max(0.0, min(1.0, float(recovery.confidence_threshold)))

    out = np.zeros_like(rho_in, dtype=np.complex128)
    for syndrome in range(2**num_syndrome):
        mask = np.zeros(dim, dtype=np.float64)
        for basis in range(dim):
            matches = True
            for bit_pos, qubit in enumerate(syndrome_qubits):
                basis_bit = (basis >> qubit) & 1
                syndrome_bit = (syndrome >> bit_pos) & 1
                if basis_bit != syndrome_bit:
                    matches = False
                    break
            if matches:
                mask[basis] = 1.0

        if float(mask.sum()) <= 0.0:
            continue

        projector = np.diag(mask.astype(np.complex128))
        branch = projector @ rho_in @ projector

        ones = int(bin(syndrome).count("1"))
        confidence = abs((2.0 * ones / max(num_syndrome, 1)) - 1.0)
        if ones < threshold or confidence < confidence_threshold:
            p_apply_x = 0.0
        else:
            after_readout = 1.0 - readout_error_rate
            p_apply_x = after_readout * (1.0 - syndrome_error_rate) + (1.0 - after_readout) * syndrome_error_rate
        p_apply_x = max(0.0, min(1.0, p_apply_x))

        out += (1.0 - p_apply_x) * branch
        out += p_apply_x * (x_full @ branch @ x_dag)

    return out


def _logical_channel_report(
    *,
    label: str,
    circuit: CircuitList,
    target_unitary: np.ndarray,
    noise_model: DemoNoiseModel,
    recovery_config: RecoveryConfig,
    num_qubits: int,
    working_qubits: tuple[int, ...],
) -> None:
    """Print matrix-level logical impact metrics on working qubits only.

    This evaluates both ideal unitary action and noisy channel action reduced to
    the logical subsystem, so encode/decode wrappers are judged by what they do
    to the protected state (not only by gate structure).
    """
    target_channel = unitary_to_superoperator(target_unitary)
    candidate_unitary = circuit_to_matrix(circuit, target_kind="unitary", num_qubits=num_qubits)
    candidate_channel = circuit_to_faulty_channel_model(
        circuit,
        error_rates=noise_model.qubit_error_rates,
        gate_error_rates=noise_model.gate_error_rates,
        num_qubits=num_qubits,
        noise_type="depolarizing",
    )

    ideal_fid, ideal_frob = subsystem_similarity(
        candidate_unitary,
        target_unitary,
        target_kind="unitary",
        num_qubits=num_qubits,
        working_qubits=working_qubits,
    )
    noisy_fid, noisy_frob = subsystem_similarity(
        candidate_channel,
        target_channel,
        target_kind="channel",
        num_qubits=num_qubits,
        working_qubits=working_qubits,
    )

    # Probe logical action on |0> against the provided target unitary.
    dim = 2**num_qubits
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0

    target_out = target_unitary @ psi
    target_rho = np.outer(target_out, target_out.conjugate())
    target_logical_rho = partial_trace_density_matrix(target_rho, num_qubits=num_qubits, keep_qubits=working_qubits)

    rho_out = apply_channel_to_state(candidate_channel, psi)
    rho_out_recovered = _apply_recovery_cptp_map(
        rho_in=rho_out,
        num_qubits=num_qubits,
        recovery=recovery_config,
        logical_qubit=working_qubits[0],
    )
    rho_logical = partial_trace_density_matrix(rho_out, num_qubits=num_qubits, keep_qubits=working_qubits)
    rho_logical_recovered = partial_trace_density_matrix(
        rho_out_recovered,
        num_qubits=num_qubits,
        keep_qubits=working_qubits,
    )
    logical_probe_overlap = float(np.real(np.trace(rho_logical @ target_logical_rho)))
    logical_probe_overlap_recovered = float(np.real(np.trace(rho_logical_recovered @ target_logical_rho)))
    purity = float(np.real(np.trace(rho_logical @ rho_logical)))
    purity_recovered = float(np.real(np.trace(rho_logical_recovered @ rho_logical_recovered)))

    print(f"{label} matrix impact (working subsystem):")
    print(f"  ideal subsystem fidelity-like   : {ideal_fid:.6f}")
    print(f"  ideal subsystem frob-like       : {ideal_frob:.6f}")
    print(f"  noisy subsystem fidelity-like   : {noisy_fid:.6f}")
    print(f"  noisy subsystem frob-like       : {noisy_frob:.6f}")
    print(f"  noisy logical probe overlap     : {logical_probe_overlap:.6f}")
    print(f"  noisy logical purity            : {purity:.6f}")
    if recovery_config.mode != "off":
        print(f"  recovered probe overlap         : {logical_probe_overlap_recovered:.6f}")
        print(f"  recovered logical purity        : {purity_recovered:.6f}")


def _full_register_basis_state(*, num_qubits: int, qubit0_bit: int) -> np.ndarray:
    state = np.zeros(2**num_qubits, dtype=np.complex128)
    full_index = 1 if int(qubit0_bit) == 1 else 0
    state[full_index] = 1.0
    return state


def _correction_score_from_final_matrix(
    *,
    circuit: CircuitList,
    noise_model: DemoNoiseModel,
    recovery_config: RecoveryConfig,
    num_qubits: int,
) -> tuple[float, float]:
    """Return (logical_correction_score, syndrome_separation_score).

    logical_correction_score:
    - Average fidelity of reduced logical output against ideal X action on qubit 0
      for probes |0> and |1> (ancillas initialised at |0>). Uses final noisy channel.

    syndrome_separation_score:
    - Distance between ancilla reduced states for the two probes, normalized to [0,1].
      This is a proxy for how distinguishable syndrome information becomes.
    """
    noisy_channel = circuit_to_faulty_channel_model(
        circuit,
        error_rates=noise_model.qubit_error_rates,
        gate_error_rates=noise_model.gate_error_rates,
        num_qubits=num_qubits,
        noise_type="depolarizing",
    )

    expected = {
        0: np.array([0.0, 1.0], dtype=np.complex128),
        1: np.array([1.0, 0.0], dtype=np.complex128),
    }

    fidelities: list[float] = []
    ancilla_rhos: list[np.ndarray] = []
    for bit in (0, 1):
        psi = _full_register_basis_state(num_qubits=num_qubits, qubit0_bit=bit)
        rho_out = apply_channel_to_state(noisy_channel, psi)
        rho_out = _apply_recovery_cptp_map(
            rho_in=rho_out,
            num_qubits=num_qubits,
            recovery=recovery_config,
            logical_qubit=0,
        )

        rho_logical = partial_trace_density_matrix(rho_out, num_qubits=num_qubits, keep_qubits=(0,))
        fidelities.append(state_fidelity(rho_logical, expected[bit]))

        rho_ancilla = partial_trace_density_matrix(rho_out, num_qubits=num_qubits, keep_qubits=(1, 2, 3, 4))
        ancilla_rhos.append(rho_ancilla)

    logical_correction_score = sum(fidelities) / len(fidelities)

    # Frobenius distance upper-bounded here by 2 for density matrices in this setting.
    syndrome_distance = float(np.linalg.norm(ancilla_rhos[0] - ancilla_rhos[1], "fro"))
    syndrome_separation_score = max(0.0, min(1.0, syndrome_distance / 2.0))
    return logical_correction_score, syndrome_separation_score


def _hub_connectivity_score(circuit: CircuitList, *, hub_qubit: int, aux_qubits: Iterable[int]) -> float:
    """Score how much two-qubit structure is directly connected to hub_qubit."""
    aux_set = {int(q) for q in aux_qubits}
    two_qubit_gates = [gate for gate in circuit.gates if len(gate.qubits) == 2]
    if not two_qubit_gates:
        return 0.0

    direct = [gate for gate in two_qubit_gates if hub_qubit in gate.qubits]
    direct_ratio = len(direct) / len(two_qubit_gates)

    connected_aux = set()
    for gate in direct:
        q0, q1 = gate.qubits
        other = q1 if q0 == hub_qubit else q0
        if other in aux_set:
            connected_aux.add(other)
    aux_coverage = len(connected_aux) / max(1, len(aux_set))

    return 0.5 * direct_ratio + 0.5 * aux_coverage


def _demo_updated_fitness(
    *,
    circuit: CircuitList,
    base_total_fitness: float,
    noise_model: DemoNoiseModel,
    recovery_config: RecoveryConfig,
    num_qubits: int,
    baseline_correction: float,
) -> tuple[float, float, float, float]:
    """Demo-local updated fitness using final matrix correction behavior.

    Returns:
      (updated_total, correction_score, syndrome_score, topology_score)
    """
    correction_score, syndrome_score = _correction_score_from_final_matrix(
        circuit=circuit,
        noise_model=noise_model,
        recovery_config=recovery_config,
        num_qubits=num_qubits,
    )
    topology_score = _hub_connectivity_score(circuit, hub_qubit=0, aux_qubits=(1, 2, 3, 4))

    # Focus on protection gain over workload baseline, not absolute correctness.
    protection_gain = max(0.0, correction_score - baseline_correction)

    updated_total = (
        0.15 * max(0.0, min(1.0, base_total_fitness))
        + 0.70 * protection_gain
        + 0.10 * topology_score
        + 0.05 * syndrome_score
    )

    # Hard gate: if no direct hub-to-aux structure exists, do not consider it protective.
    if topology_score <= 0.0:
        updated_total = 0.0

    return updated_total, correction_score, syndrome_score, topology_score


def _compose_circuits(prefix: CircuitList, suffix: CircuitList) -> CircuitList:
    """Compose two circuits as prefix || suffix and recompute depths."""
    composed = CircuitList(
        gates=list(prefix.gates) + list(suffix.gates),
        cluster=tuple(sorted(set(prefix.cluster) | set(suffix.cluster))),
        max_depth=max(prefix.max_depth, suffix.max_depth) + len(prefix.gates) + len(suffix.gates),
        max_gates=(prefix.max_gates or len(prefix.gates)) + (suffix.max_gates or len(suffix.gates)) + 4,
    )
    composed.crop_to_limits()
    return composed


def _identity_reference(num_qubits: int) -> CircuitList:
    return CircuitList(gates=[], cluster=tuple(range(num_qubits)), max_depth=1, max_gates=1)


def build_logical_workload() -> CircuitList:
    """Logical workload: X on logical qubit 0 in a 5-qubit register."""
    return CircuitList(
        gates=[_gate("x", (0,), 1)],
        cluster=(0, 1, 2, 3, 4),
        max_depth=12,
        max_gates=20,
    )


def build_repetition_scaffold() -> CircuitList:
    """Coherent repetition-style scaffold around logical qubit 0.

    Data-like qubits: 0,1,2
    Ancilla-like qubits: 3,4

    Pattern:
    - Encode parity from q0 to q1,q2
    - Apply a syndrome-like coherent parity round using ancillas
    - Uncompute the parity fan-out
    """
    gates = [
        # Encode-like fan-out (repetition-inspired)
        _gate("cx", (0, 1), 1),
        _gate("cx", (0, 2), 2),
        # Coherent syndrome-like checks
        _gate("cx", (0, 3), 3),
        _gate("cx", (1, 3), 4),
        _gate("cx", (0, 4), 5),
        _gate("cx", (2, 4), 6),
        # Ancilla basis mixing
        _gate("h", (3,), 7),
        _gate("h", (4,), 8),
        # Uncompute encoder fan-out
        _gate("cx", (0, 2), 9),
        _gate("cx", (0, 1), 10),
    ]
    circuit = CircuitList(
        gates=gates,
        cluster=(0, 1, 2, 3, 4),
        max_depth=12,
        max_gates=20,
    )
    circuit.crop_to_limits()
    return circuit


def build_surface_mini_scaffold() -> CircuitList:
    """Tiny surface-like coherent parity round for 1 data + 4 ancillas.

    Working/data-like qubit: 0
    Ancilla-like qubits: 1,2,3,4

    The check-uncheck pattern is unitary and lightweight, making it suitable for
    quick EA iterations while still encouraging ancilla-coupled structure.
    """
    gates = [
        _gate("cx", (0, 1), 1),
        _gate("cx", (0, 2), 2),
        _gate("cx", (0, 3), 3),
        _gate("cx", (0, 4), 4),
        _gate("cx", (0, 4), 5),
        _gate("cx", (0, 3), 6),
        _gate("cx", (0, 2), 7),
        _gate("cx", (0, 1), 8),
    ]
    circuit = CircuitList(
        gates=gates,
        cluster=(0, 1, 2, 3, 4),
        max_depth=10,
        max_gates=16,
    )
    circuit.crop_to_limits()
    return circuit


def build_bitflip3_known_good_seed() -> CircuitList:
    """Bit-flip-code-inspired coherent seed used by EA (no measurement feedback).

    Data-like qubits: 0,1,2
    Syndrome ancillas: 3,4
    """
    gates = [
        # Encode repetition parity from logical qubit 0.
        _gate("cx", (0, 1), 1),
        _gate("cx", (0, 2), 2),
        # Workload representative on logical core.
        _gate("x", (0,), 3),
        # Coherent syndrome-style checks (Z2 checks via CNOT parity copy).
        _gate("cx", (0, 3), 4),
        _gate("cx", (1, 3), 5),
        _gate("cx", (0, 4), 6),
        _gate("cx", (2, 4), 7),
        # Uncompute checks.
        _gate("cx", (2, 4), 8),
        _gate("cx", (0, 4), 9),
        _gate("cx", (1, 3), 10),
        _gate("cx", (0, 3), 11),
        # Decode back to logical qubit 0.
        _gate("cx", (0, 2), 12),
        _gate("cx", (0, 1), 13),
    ]
    circuit = CircuitList(
        gates=gates,
        cluster=(0, 1, 2, 3, 4),
        max_depth=16,
        max_gates=24,
    )
    circuit.crop_to_limits()
    return circuit


def _cx_sequence(circuit: CircuitList) -> list[tuple[int, int]]:
    return [
        tuple(gate.qubits)
        for gate in circuit.gates
        if gate.name == "cx" and len(gate.qubits) == 2
    ]


def _ordered_match_ratio(candidate: list[tuple[int, int]], template: list[tuple[int, int]]) -> float:
    if not template:
        return 0.0
    template_index = 0
    for pair in candidate:
        if template_index < len(template) and pair == template[template_index]:
            template_index += 1
    return template_index / len(template)


def _recognize_known_scheme(circuit: CircuitList) -> tuple[str, float, dict[str, float]]:
    """Classify evolved circuit against known code templates (post-hoc analysis)."""
    templates = {
        "surface_mini": build_surface_mini_scaffold(),
        "repetition_scaffold": build_repetition_scaffold(),
        "bitflip3_seed": build_bitflip3_known_good_seed(),
    }
    candidate_seq = _cx_sequence(circuit)
    candidate_set = set(candidate_seq)

    scores: dict[str, float] = {}
    for name, template in templates.items():
        template_seq = _cx_sequence(template)
        template_set = set(template_seq)
        coverage = len(candidate_set.intersection(template_set)) / max(1, len(template_set))
        order = _ordered_match_ratio(candidate_seq, template_seq)
        scores[name] = (0.6 * coverage) + (0.4 * order)

    best_name = max(scores, key=scores.get)
    best_score = scores[best_name]
    if best_score < 0.55:
        return "unrecognized", best_score, scores
    return best_name, best_score, scores


def main() -> None:
    args = _parse_args()
    preset = _resolve_preset(args.preset)

    print("=" * 118)
    print("QPC Demo: Repetition-Code-Inspired Protective Scaffold + EA Refinement")
    print("=" * 118)
    print(
        f"Preset: {args.preset} (qpc_mode={preset.qpc_mode}, pop={preset.population_size}, "
        f"gens={preset.max_generations}, scaffold={args.scaffold})"
    )

    workload = build_logical_workload()
    scaffold = build_surface_mini_scaffold() if args.scaffold == "surface-mini" else build_repetition_scaffold()
    scaffold_with_workload = _compose_circuits(scaffold, workload)

    target_matrix = circuit_to_matrix(workload, target_kind="unitary", num_qubits=5)
    identity_target_matrix = circuit_to_matrix(_identity_reference(5), target_kind="unitary", num_qubits=5)

    # Logical qubit is noisy; helper qubits are cleaner.
    if args.noise_profile == "x-only":
        noise_model = DemoNoiseModel(
            qubit_error_rates={
                0: 0.035,
                1: 0.0,
                2: 0.0,
                3: 0.0,
                4: 0.0,
            },
            gate_error_rates={"x": 1.0},
        )
    else:
        noise_model = DemoNoiseModel(
            qubit_error_rates={
                0: 0.035,
                1: 0.0,
                2: 0.0,
                3: 0.0,
                4: 0.0,
            },
            gate_error_rates=None,
        )

    recovery_config = RecoveryConfig(
        mode=args.recovery,
        syndrome_error_rate=max(0.0, min(1.0, float(args.syndrome_error_rate))),
        readout_error_rate=max(0.0, min(1.0, float(args.readout_error_rate))),
        confidence_threshold=max(0.0, min(1.0, float(args.recovery_confidence_threshold))),
    )

    baseline = fitness_circuit(
        workload,
        target=None,
        target_matrix=target_matrix,
        target_kind="unitary",
        noise_model=noise_model,
        qpc_enabled=True,
        qpc_mode=preset.qpc_mode,
        behavior_weight=0.6,
        robustness_weight=0.35,
        complexity_weight=0.05,
        working_qubits=(0,),
    )

    seeded = fitness_circuit(
        scaffold_with_workload,
        target=None,
        target_matrix=target_matrix,
        target_kind="unitary",
        noise_model=noise_model,
        qpc_enabled=True,
        qpc_mode=preset.qpc_mode,
        behavior_weight=0.6,
        robustness_weight=0.35,
        complexity_weight=0.05,
        working_qubits=(0,),
    )

    config = CircuitEvolutionConfig(
        device_qubits=5,
        cluster_size=5,
        population_size=preset.population_size,
        max_generations=preset.max_generations,
        lambda_ratio=2,
        crossover_rate=0.9,
        mutation_rate=0.35,
        tournament_k=3,
        max_depth=preset.max_depth,
        max_gates=preset.max_gates,
        diversity_floor=0.02,
        stagnation_generations=7,
        target_fitness_threshold=0.999,
        target_kind="unitary",
        behavior_weight=0.6,
        robustness_weight=0.35,
        complexity_weight=0.05,
        fidelity_weight=0.7,
        frobenius_weight=0.3,
        qpc_enabled=True,
        qpc_mode=preset.qpc_mode,
        working_qubits=(0,),
        candidate_clusters=((0, 1, 2, 3, 4),),
        candidate_cluster_weights=(1.0,),
    )

    gate_catalog = list(preset.gate_catalog)

    print("Architecture objective: prefer 2-qubit structure with direct links to hub qubit 0.")
    print("EA template seeding: enabled (surface_mini, repetition_scaffold, bitflip3_seed)")
    print(f"Noise profile: {args.noise_profile}")
    print(
        f"Recovery map: {recovery_config.mode} "
        f"(syndrome_err={recovery_config.syndrome_error_rate:.3f}, "
        f"readout_err={recovery_config.readout_error_rate:.3f}, "
        f"confidence={recovery_config.confidence_threshold:.3f})"
    )

    baseline_correction, _ = _correction_score_from_final_matrix(
        circuit=workload,
        noise_model=noise_model,
        recovery_config=recovery_config,
        num_qubits=5,
    )

    if args.skip_evolution:
        result = None
        evolved = None
        selected_circuit = None
        selected_meta = None
    else:
        # Seed all known templates (keeps population size fixed; EA fills the rest).
        initial_population = [
            build_surface_mini_scaffold(),
            build_repetition_scaffold(),
            build_bitflip3_known_good_seed(),
        ]

        result = evolutionary_best_circuit(
            config=config,
            gate_catalog=gate_catalog,
            target_circuit=None,
            target_matrix=target_matrix,
            noise_model=noise_model,
            seed=2026,
            initial_population=initial_population,
        )

        evolved = fitness_circuit(
            result.best_circuit,
            target=None,
            target_matrix=target_matrix,
            target_kind="unitary",
            noise_model=noise_model,
            qpc_enabled=True,
            qpc_mode=preset.qpc_mode,
            behavior_weight=0.6,
            robustness_weight=0.35,
            complexity_weight=0.05,
            working_qubits=(0,),
        )

        # Re-rank final candidates with demo-local updated fitness that uses the
        # final noisy matrix correction score on qubit 0.
        scored_candidates: list[tuple[float, CircuitList, float, float, float]] = []
        for circuit, base_breakdown in zip(result.population, result.fitness_scores):
            updated_total, correction, syndrome, topology = _demo_updated_fitness(
                circuit=circuit,
                base_total_fitness=base_breakdown.total_fitness,
                noise_model=noise_model,
                recovery_config=recovery_config,
                num_qubits=5,
                baseline_correction=baseline_correction,
            )
            scored_candidates.append((updated_total, circuit, correction, syndrome, topology))

        # Also score the explicit extra candidates (best, scaffold||workload, workload).
        extras = [result.best_circuit, scaffold_with_workload, workload]
        for circuit in extras:
            base = fitness_circuit(
                circuit,
                target=None,
                target_matrix=target_matrix,
                target_kind="unitary",
                noise_model=noise_model,
                qpc_enabled=True,
                qpc_mode=preset.qpc_mode,
                behavior_weight=0.6,
                robustness_weight=0.35,
                complexity_weight=0.05,
                working_qubits=(0,),
            )
            updated_total, correction, syndrome, topology = _demo_updated_fitness(
                circuit=circuit,
                base_total_fitness=base.total_fitness,
                noise_model=noise_model,
                recovery_config=recovery_config,
                num_qubits=5,
                baseline_correction=baseline_correction,
            )
            scored_candidates.append((updated_total, circuit, correction, syndrome, topology))

        best_updated = max(scored_candidates, key=lambda entry: entry[0])
        selected_circuit = best_updated[1]
        selected_meta = {
            "updated_total": best_updated[0],
            "correction": best_updated[2],
            "syndrome": best_updated[3],
            "topology": best_updated[4],
            "pool_size": len(scored_candidates),
        }

        recognized_name, recognized_score, recognized_scores = _recognize_known_scheme(selected_circuit)
        selected_meta["recognized_name"] = recognized_name
        selected_meta["recognized_score"] = recognized_score
        selected_meta["recognized_surface"] = recognized_scores.get("surface_mini", 0.0)
        selected_meta["recognized_repetition"] = recognized_scores.get("repetition_scaffold", 0.0)
        selected_meta["recognized_bitflip"] = recognized_scores.get("bitflip3_seed", 0.0)

        evolved = fitness_circuit(
            selected_circuit,
            target=None,
            target_matrix=target_matrix,
            target_kind="unitary",
            noise_model=noise_model,
            qpc_enabled=True,
            qpc_mode=preset.qpc_mode,
            behavior_weight=0.6,
            robustness_weight=0.35,
            complexity_weight=0.05,
            working_qubits=(0,),
        )

    print()
    print("Logical workload:")
    _print_circuit("Workload", workload)

    print()
    print("Seed scaffold (repetition-inspired):")
    _print_circuit("Scaffold", scaffold)

    print()
    print("Baseline (no wrapper):")
    print(f"  total fitness      : {baseline.total_fitness:.6f}")
    print(f"  behavior score     : {baseline.behavior_score:.6f}")
    print(f"  robustness score   : {baseline.robustness_score:.6f}")
    print(f"  complexity score   : {baseline.complexity_score:.6f}")

    print()
    print("Seed scaffold under same model:")
    print("  (evaluated as scaffold || workload-X)")
    print(f"  total fitness      : {seeded.total_fitness:.6f}")
    print(f"  behavior score     : {seeded.behavior_score:.6f}")
    print(f"  robustness score   : {seeded.robustness_score:.6f}")
    print(f"  complexity score   : {seeded.complexity_score:.6f}")

    print()
    print("Improvement summary:")
    print(f"  scaffold vs baseline total  : {seeded.total_fitness - baseline.total_fitness:+.6f}")

    print()
    print("Matrix-level logical impact analysis:")
    _logical_channel_report(
        label="Workload",
        circuit=workload,
        target_unitary=target_matrix,
        noise_model=noise_model,
        recovery_config=recovery_config,
        num_qubits=5,
        working_qubits=(0,),
    )
    _logical_channel_report(
        label="Scaffold+Workload",
        circuit=scaffold_with_workload,
        target_unitary=target_matrix,
        noise_model=noise_model,
        recovery_config=recovery_config,
        num_qubits=5,
        working_qubits=(0,),
    )
    _logical_channel_report(
        label="Scaffold-only (identity reference)",
        circuit=scaffold,
        target_unitary=identity_target_matrix,
        noise_model=noise_model,
        recovery_config=recovery_config,
        num_qubits=5,
        working_qubits=(0,),
    )

    if args.skip_evolution:
        print("  evolution skipped           : yes (--skip-evolution)")
    else:
        print()
        print("Best evolved candidate under updated demo fitness:")
        _print_circuit("Evolved", selected_circuit)
        print(f"  stop reason        : {result.stop_reason}")
        print(f"  generations run    : {result.generations_run}")
        print(f"  cache entries      : {result.cache_entries}")
        print(f"  candidate pool     : {selected_meta['pool_size']}")
        print(f"  updated fitness    : {selected_meta['updated_total']:.6f}")
        print(f"  correction score   : {selected_meta['correction']:.6f}")
        print(f"  syndrome score     : {selected_meta['syndrome']:.6f}")
        print(f"  topology score     : {selected_meta['topology']:.6f}")
        print(f"  recognized scheme  : {selected_meta['recognized_name']}")
        print(f"  recognition score  : {selected_meta['recognized_score']:.6f}")
        print(f"  match surface_mini : {selected_meta['recognized_surface']:.6f}")
        print(f"  match repetition   : {selected_meta['recognized_repetition']:.6f}")
        print(f"  match bitflip3     : {selected_meta['recognized_bitflip']:.6f}")

        print()
        print("Evolved under same noisy QPC model:")
        print(f"  total fitness      : {evolved.total_fitness:.6f}")
        print(f"  behavior score     : {evolved.behavior_score:.6f}")
        print(f"  robustness score   : {evolved.robustness_score:.6f}")
        print(f"  complexity score   : {evolved.complexity_score:.6f}")
        print(f"  evolved vs baseline total   : {evolved.total_fitness - baseline.total_fitness:+.6f}")
        print(f"  evolved vs scaffold total   : {evolved.total_fitness - seeded.total_fitness:+.6f}")
        print()
        _logical_channel_report(
            label="Evolved",
            circuit=selected_circuit,
            target_unitary=target_matrix,
            noise_model=noise_model,
            recovery_config=recovery_config,
            num_qubits=5,
            working_qubits=(0,),
        )
    print()
    print("Note: This is a repetition-code-inspired coherent wrapper, not full")
    print("measurement-and-feedback QEC. It is compatible with current unitary EA flow.")


if __name__ == "__main__":
    main()
