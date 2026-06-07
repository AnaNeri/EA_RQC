"""Demo: QPC-guided evolution with focus on critical qubits.

This demo is inspired by a quantamorphism-style circuit structure with
separate target, control, and ancilla roles. It evolves a robust circuit that
preserves target behavior while operating under heterogeneous qubit noise,
where specific control qubits are intentionally more error-prone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from circuit.entities.circuit_list import CircuitGate, CircuitList
from evolution.circuit.evolutionary import CircuitEvolutionConfig, evolutionary_best_circuit
from evolution.circuit.fitness import fitness_circuit
from evolution.circuit.matrix_eval import circuit_to_matrix


@dataclass(frozen=True)
class DemoNoiseModel:
	qubit_error_rates: dict[int, float]


def _gate(name: str, qubits: tuple[int, ...], depth: int, param: float | None = None) -> CircuitGate:
	params = (param,) if param is not None else ()
	return CircuitGate.from_values(name=name, qubits=qubits, parameters=params, depth=depth)


def _print_circuit(label: str, circuit: CircuitList) -> None:
	print(f"{label}: cluster={circuit.cluster}, gates={len(circuit.gates)}")
	for gate in circuit.gates:
		params = f" params={tuple(round(value, 4) for value in gate.parameters)}" if gate.parameters else ""
		print(f"  d={gate.depth:02d} {gate.name.upper():>3} q={gate.qubits}{params}")


def _critical_usage(circuit: CircuitList, critical_qubits: tuple[int, ...]) -> tuple[int, int, float]:
	total = len(circuit.gates)
	critical = 0
	critical_set = set(critical_qubits)
	for gate in circuit.gates:
		if any(qubit in critical_set for qubit in gate.qubits):
			critical += 1
	ratio = (critical / total) if total > 0 else 0.0
	return critical, total, ratio


def _rank_qubits_for_protection(circuit: CircuitList, noise_model: DemoNoiseModel) -> list[tuple[int, float, int, float]]:
	"""Return ranked qubits as (qubit, score, gate_hits, noise_rate).

	Score heuristic: noise_rate * gate_hits, where gate_hits counts how often each
	qubit appears in circuit gates.
	"""
	noise_rates = noise_model.qubit_error_rates
	gate_hits: dict[int, int] = {qubit: 0 for qubit in circuit.cluster}
	for gate in circuit.gates:
		for qubit in gate.qubits:
			gate_hits[int(qubit)] = gate_hits.get(int(qubit), 0) + 1

	ranked: list[tuple[int, float, int, float]] = []
	for qubit in sorted(set(circuit.cluster) | set(noise_rates.keys()) | set(gate_hits.keys())):
		hits = int(gate_hits.get(int(qubit), 0))
		noise = float(noise_rates.get(int(qubit), 0.0))
		score = noise * hits
		ranked.append((int(qubit), score, hits, noise))

	ranked.sort(key=lambda item: (item[1], item[2], item[3]), reverse=True)
	return ranked


def build_quantamorphism_like_circuit() -> CircuitList:
	"""Build a compact circuit with roles similar to target/control/ancilla.

	Qubit roles:
	- q0: target
	- q1, q2: controls (critical)
	- q3: ancilla helper
	"""
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


def main() -> None:
	print("=" * 120)
	print("QPC Demo: Evolve Robust Circuit with Critical-Qubit Priority")
	print("=" * 120)
	print()
	print("Roles: q0 target, q1-q2 critical controls, q3 ancilla helper")

	critical_qubits = (1, 2)
	working_qubits = (0, 1, 2)

	given_circuit = build_quantamorphism_like_circuit()
	target_matrix = circuit_to_matrix(given_circuit, target_kind="unitary", num_qubits=4)

	# Critical controls are intentionally much noisier than target/ancilla.
	noise_model = DemoNoiseModel(
		qubit_error_rates={
			0: 0.010,  # target
			1: 0.040,  # critical control
			2: 0.035,  # critical control
			3: 0.006,  # ancilla
		}
	)

	gate_catalog = ["id", "x", "h", "sx", "rz", "cx", "cz", "ecr", "swap"]

	baseline = fitness_circuit(
		given_circuit,
		target=None,
		target_matrix=target_matrix,
		target_kind="unitary",
		noise_model=noise_model,
		behavior_weight=0.55,
		robustness_weight=0.40,
		complexity_weight=0.05,
		qpc_enabled=True,
		qpc_mode="exact",
		working_qubits=working_qubits,
	)

	config = CircuitEvolutionConfig(
		device_qubits=4,
		cluster_size=4,
		population_size=14,
		max_generations=8,
		lambda_ratio=2,
		crossover_rate=0.9,
		mutation_rate=0.35,
		tournament_k=3,
		max_depth=8,
		max_gates=18,
		diversity_floor=0.02,
		stagnation_generations=4,
		target_fitness_threshold=0.9975,
		target_kind="unitary",
		behavior_weight=0.55,
		robustness_weight=0.40,
		complexity_weight=0.05,
		fidelity_weight=0.7,
		frobenius_weight=0.3,
		qpc_enabled=True,
		qpc_mode="mixed",
		working_qubits=working_qubits,
		candidate_clusters=((0, 1, 2, 3),),
		candidate_cluster_weights=(1.0,),
	)

	result = evolutionary_best_circuit(
		config=config,
		gate_catalog=gate_catalog,
		target_circuit=None,
		target_matrix=target_matrix,
		noise_model=noise_model,
		seed=20260606,
	)
	evolved = fitness_circuit(
		result.best_circuit,
		target=None,
		target_matrix=target_matrix,
		target_kind="unitary",
		noise_model=noise_model,
		behavior_weight=0.55,
		robustness_weight=0.40,
		complexity_weight=0.05,
		qpc_enabled=True,
		qpc_mode="exact",
		working_qubits=working_qubits,
	)

	ranking = _rank_qubits_for_protection(given_circuit, noise_model)

	base_critical, base_total, base_ratio = _critical_usage(given_circuit, critical_qubits)
	evo_critical, evo_total, evo_ratio = _critical_usage(result.best_circuit, critical_qubits)

	print()
	print("Given circuit:")
	_print_circuit("Given", given_circuit)

	print()
	print("Baseline QPC fitness:")
	print(f"  total fitness      : {baseline.total_fitness:.6f}")
	print(f"  behavior score     : {baseline.behavior_score:.6f}")
	print(f"  robustness score   : {baseline.robustness_score:.6f}")
	print(f"  complexity score   : {baseline.complexity_score:.6f}")
	print(f"  critical gate use  : {base_critical}/{base_total} ({100.0 * base_ratio:.1f}%)")

	print()
	print("Best evolved candidate:")
	_print_circuit("Evolved", result.best_circuit)
	print(f"  stop reason        : {result.stop_reason}")
	print(f"  generations run    : {result.generations_run}")
	print(f"  cache entries      : {result.cache_entries}")

	print()
	print("Evolved QPC fitness:")
	print(f"  total fitness      : {evolved.total_fitness:.6f}")
	print(f"  behavior score     : {evolved.behavior_score:.6f}")
	print(f"  robustness score   : {evolved.robustness_score:.6f}")
	print(f"  complexity score   : {evolved.complexity_score:.6f}")
	print(f"  critical gate use  : {evo_critical}/{evo_total} ({100.0 * evo_ratio:.1f}%)")

	print()
	print("Improvement summary:")
	print(f"  total delta        : {evolved.total_fitness - baseline.total_fitness:+.6f}")
	print(f"  behavior delta     : {evolved.behavior_score - baseline.behavior_score:+.6f}")
	print(f"  robustness delta   : {evolved.robustness_score - baseline.robustness_score:+.6f}")
	print(f"  complexity delta   : {evolved.complexity_score - baseline.complexity_score:+.6f}")
	print(f"  critical use delta : {100.0 * (evo_ratio - base_ratio):+.1f} percentage points")

	print()
	print("Interpretation:")
	print("  - The objective checks q0,q1,q2 and excludes ancilla q3.")
	print("  - Under heterogeneous noise, evolution can trade structure and qubit usage")
	print("    to improve robustness without requiring protection on every qubit.")
	print()
	print("Recommended protection order (noise x gate participation):")
	for idx, (qubit, score, hits, noise) in enumerate(ranking, start=1):
		role = "target" if qubit == 0 else ("critical-control" if qubit in (1, 2) else "ancilla")
		print(f"  {idx}. q{qubit} ({role}) -> score={score:.6f} [noise={noise:.3f}, gate_hits={hits}]")


if __name__ == "__main__":
	main()
