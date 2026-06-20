"""Demo: quantamorphism baseline + protective wrappers analysis.

Goals implemented:
 - start with a baseline circuit named "quantamorphism"
 - generate wrapper circuits (random + hand-crafted) that use ancillas to protect
   qubits 0,1,2
 - compose wrappers with the baseline and evaluate fitness under varying noise
 - analyze which qubit protections have more impact

This demo uses the existing random_circuit and fitness_circuit helpers so it
integrates with the project's main code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from circuit.entities.circuit_list import CircuitGate, CircuitList
from evolution.circuit.operators import random_circuit
from evolution.circuit.fitness import fitness_circuit
from evolution.circuit.matrix_eval import circuit_to_matrix


@dataclass(frozen=True)
class DemoNoiseModel:
    qubit_error_rates: dict[int, float]


def _gate(name: str, qubits: tuple[int, ...], depth: int, param: float | None = None) -> CircuitGate:
    params = (param,) if param is not None else ()
    return CircuitGate.from_values(name=name, qubits=qubits, parameters=params, depth=depth)


def build_quantamorphism() -> CircuitList:
    # Use the exact quantamorphism baseline specified by the user.
    return CircuitList(
        gates=[
            _gate("id", (0,), 1),
            _gate("x", (1,), 1),
            _gate("x", (2,), 1),
            _gate("cx", (1, 3), 2),
            _gate("cx", (2, 3), 2),
            _gate("rz", (0,), 3, math.pi),
            _gate("cx", (3, 0), 4),
            _gate("cx", (2, 3), 5),
            _gate("cx", (1, 3), 5),
        ],
        cluster=(0, 1, 2, 3),
        max_depth=8,
        max_gates=20,
    )


def protective_wrapper_for(qubits: Iterable[int], ancilla_map: dict[int, int]) -> CircuitList:
    """Create a small wrapper that entangles each target qubit with its ancilla.

    The wrapper will be prepended to the baseline when composed. This is a
    simple entangling protection: CX(target -> ancilla) then H ancilla, which
    can act as a crude parity/backup entanglement.
    """
    gates = []
    depth = 1
    for q in qubits:
        a = ancilla_map.get(q)
        if a is None:
            continue
        # Copy parity to ancilla
        gates.append(_gate("cx", (q, a), depth))
        depth += 1
        # Rotate ancilla to spread coherence
        gates.append(_gate("h", (a,), depth))
        depth += 1
    return CircuitList(gates=gates, cluster=tuple(sorted(set(list(qubits) + list(ancilla_map.values())))), max_depth=8, max_gates=16)


def pseudo_syndrome_round(qubits: Iterable[int], ancilla_map: dict[int, int]) -> CircuitList:
    """Add a measurement-free syndrome-like check round.

    This simulator does not support mid-circuit measurement/conditional gates.
    We therefore model a check-uncheck parity round, which is unitary and returns
    ancillas to |0> in the noiseless case while still contributing realistic gate
    overhead under noise.
    """
    gates = []
    depth = 1
    for q in qubits:
        a = ancilla_map.get(q)
        if a is None:
            continue
        gates.append(_gate("cx", (q, a), depth))
        depth += 1
        gates.append(_gate("h", (a,), depth))
        depth += 1
        gates.append(_gate("h", (a,), depth))
        depth += 1
        gates.append(_gate("cx", (q, a), depth))
        depth += 1
    return CircuitList(gates=gates, cluster=tuple(sorted(set(list(qubits) + list(ancilla_map.values())))), max_depth=16, max_gates=32)


# Gates that are their own inverse (no parameter negation needed)
_SELF_INVERSE_GATES = {"id", "x", "y", "z", "h", "cx", "cy", "cz", "ccx", "swap"}


def invert_wrapper(wrapper: CircuitList) -> CircuitList:
    """Return the inverse of a wrapper by reversing gate order and negating rotation parameters."""
    inv_gates = []
    for g in reversed(wrapper.gates):
        if g.name in _SELF_INVERSE_GATES:
            inv_params = g.parameters
        else:
            # Negate all rotation parameters (covers rz, rx, ry, p, u1, ...)
            inv_params = tuple(-p for p in g.parameters)
        inv_gates.append(
            CircuitGate.from_values(
                name=g.name,
                qubits=g.qubits,
                parameters=inv_params,
                depth=0,  # will be relayered
            )
        )
    result = CircuitList(
        gates=inv_gates,
        cluster=wrapper.cluster,
        max_depth=wrapper.max_depth,
        max_gates=wrapper.max_gates,
    )
    result.relayer_depths()
    return result


def compose_encode_decode(
    wrapper: CircuitList,
    baseline: CircuitList,
    *,
    working_qubits: tuple[int, ...],
    ancilla_map: dict[int, int],
    include_syndrome_round: bool,
) -> CircuitList:
    """Build wrapper || baseline || wrapper_inverse || (optional syndrome round)."""
    inv = invert_wrapper(wrapper)
    all_gates = list(wrapper.gates) + list(baseline.gates) + list(inv.gates)
    if include_syndrome_round:
        syndrome = pseudo_syndrome_round(working_qubits, ancilla_map)
        all_gates.extend(syndrome.gates)
    combined_cluster = tuple(sorted(set(wrapper.cluster) | set(baseline.cluster)))
    # Allow enough room for all concatenated sections
    comp = CircuitList(
        gates=[],
        cluster=combined_cluster,
        max_depth=(3 * wrapper.max_depth + baseline.max_depth + 12) or 0,
        max_gates=(len(all_gates) + 1),
    )
    comp.gates = all_gates
    comp.relayer_depths()
    return comp


def wrapper_protects_qubit(wrapper: CircuitList, qubit: int, ancillas: set[int]) -> bool:
    # Heuristic: wrapper that contains two-qubit gates involving qubit and any
    # ancilla is considered a protection for that qubit.
    for g in wrapper.gates:
        if len(g.qubits) >= 2 and qubit in g.qubits and any(a in g.qubits for a in ancillas):
            return True
    return False


def analyze_population(population: list[CircuitList], baseline: CircuitList, working_qubits: tuple[int, ...]) -> None:
    baseline_matrix = circuit_to_matrix(baseline, target_kind="unitary", num_qubits=max(4, max(baseline.cluster) + 1))

    # Noise scenarios
    base_noise = DemoNoiseModel(qubit_error_rates={q: 0.02 for q in range(6)})
    # High-noise scenarios where one of 0,1,2 is made worse
    high_noise_by_qubit = {
        q: DemoNoiseModel(qubit_error_rates={**{i: 0.02 for i in range(6)}, q: 0.2}) for q in working_qubits
    }

    ancillas = {3, 4, 5}
    ancilla_map = {0: 3, 1: 4, 2: 5}
    include_syndrome_round = True

    # Evaluate each wrapper composed with baseline under base and high-noise
    records = []
    for idx, wrapper in enumerate(population):
        comp = compose_encode_decode(
            wrapper,
            baseline,
            working_qubits=working_qubits,
            ancilla_map=ancilla_map,
            include_syndrome_round=include_syndrome_round,
        )
        base_fb = fitness_circuit(
            comp,
            target=None,
            target_matrix=baseline_matrix,
            target_kind="unitary",
            noise_model=base_noise,
            behavior_weight=0.8,
            robustness_weight=0.2,
            complexity_weight=0.0,
            qpc_enabled=True,
            qpc_mode="approx",
            working_qubits=working_qubits,
        )

        high_scores = {}
        for q, noise in high_noise_by_qubit.items():
            fb = fitness_circuit(
                comp,
                target=None,
                target_matrix=baseline_matrix,
                target_kind="unitary",
                noise_model=noise,
                behavior_weight=0.8,
                robustness_weight=0.2,
                complexity_weight=0.0,
                qpc_enabled=True,
                qpc_mode="approx",
                working_qubits=working_qubits,
            )
            high_scores[q] = fb.total_fitness

        protections = {q: wrapper_protects_qubit(wrapper, q, ancillas) for q in working_qubits}

        records.append({
            "idx": idx,
            "wrapper": wrapper,
            "base_fitness": base_fb.total_fitness,
            "high_scores": high_scores,
            "protections": protections,
        })

    # Print per-candidate summary
    mode = "encode-baseline-decode-syndrome" if include_syndrome_round else "encode-baseline-decode"
    print(f"Analysis for composed population (baseline=quantamorphism, mode={mode})")
    print("Idx  base     high0    high1    high2   protects(0,1,2)")
    for r in records:
        print(f"{r['idx']:3d}  {r['base_fitness']:.4f}  {r['high_scores'][0]:.4f}  {r['high_scores'][1]:.4f}  {r['high_scores'][2]:.4f}   {tuple(int(r['protections'][q]) for q in working_qubits)}")

    # Aggregate impact: average delta when qubit q is noisy, grouped by whether
    # wrapper protects that qubit.
    print()
    print("Aggregate impact by protection:")
    for q in working_qubits:
        protected_deltas = []
        unprotected_deltas = []
        for r in records:
            delta = r["base_fitness"] - r["high_scores"][q]
            if r["protections"][q]:
                protected_deltas.append(delta)
            else:
                unprotected_deltas.append(delta)
        def avg(xs):
            return sum(xs) / len(xs) if xs else float("nan")
        print(f"Qubit {q}: avg delta when protected = {avg(protected_deltas):.4f}, when unprotected = {avg(unprotected_deltas):.4f}")


def main() -> None:
    print("Demo: quantamorphism protection analysis")
    working_qubits = (0, 1, 2)

    baseline = build_quantamorphism()
    print("Baseline (quantamorphism) cluster:")
    for g in baseline.gates:
        print(f"  d={g.depth:02d} {g.name} q={g.qubits}")

    # Build a small population of wrappers: some hand-crafted protections and
    # some random wrappers that may also use ancillas.
    ancilla_map = {0: 3, 1: 4, 2: 5}
    population: list[CircuitList] = []

    # Hand-crafted wrappers protecting individual qubits and combos
    population.append(protective_wrapper_for((0,), ancilla_map))
    population.append(protective_wrapper_for((1,), ancilla_map))
    population.append(protective_wrapper_for((2,), ancilla_map))
    population.append(protective_wrapper_for((0, 1), ancilla_map))
    population.append(protective_wrapper_for((0, 2), ancilla_map))
    population.append(protective_wrapper_for((1, 2), ancilla_map))
    population.append(protective_wrapper_for((0, 1, 2), ancilla_map))

    # Random wrappers (use clusters that include ancillas)
    gate_catalog = ["id", "x", "h", "rz", "cx", "cz"]
    for i in range(20):
        pop = random_circuit(cluster=(0, 1, 2, 3, 4, 5), max_depth=6, max_gates=6, gate_catalog=gate_catalog, seed=1000 + i)
        population.append(pop)

    analyze_population(population, baseline, working_qubits)


if __name__ == "__main__":
    main()
