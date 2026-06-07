"""Demo: QPC-guided evolution for error-correcting wrappers around a given circuit.

This demo treats a one-qubit logical circuit as the protected workload and allows
an ancilla qubit for evolved correction structure. Fitness is evaluated only on
the logical working qubit so ancilla outputs are ignored.
"""

from __future__ import annotations

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


def main() -> None:
	print("=" * 110)
	print("QPC Demo: Evolving Error-Correcting Structure for a Given Circuit")
	print("=" * 110)

	# Given logical circuit to protect (workload): X on logical qubit 0.
	# We place it in a 2-qubit register (qubit 0 = working, qubit 1 = ancilla).
	given_circuit = CircuitList(
		gates=[_gate("x", (0,), 1)],
		cluster=(0, 1),
		max_depth=4,
		max_gates=6,
	)

	# Target representation uses two qubits so evolution can use ancilla wiring,
	# while scoring only checks the working qubit via working_qubits=(0,).
	target_matrix = circuit_to_matrix(given_circuit, target_kind="unitary", num_qubits=2)

	# Qubit 0 is intentionally noisy. Qubit 1 is cleaner and can act as helper ancilla.
	noise_model = DemoNoiseModel(qubit_error_rates={0: 0.03, 1: 0.004})

	gate_catalog = ["id", "x", "h", "sx", "rz", "cx", "cz", "ecr"]

	baseline = fitness_circuit(
		given_circuit,
		target=None,
		target_matrix=target_matrix,
		target_kind="unitary",
		noise_model=noise_model,
		qpc_enabled=True,
		qpc_mode="exact",
		behavior_weight=0.6,
		robustness_weight=0.35,
		complexity_weight=0.05,
		working_qubits=(0,),
	)

	config = CircuitEvolutionConfig(
		device_qubits=2,
		cluster_size=2,
		population_size=24,
		max_generations=20,
		lambda_ratio=2,
		crossover_rate=0.9,
		mutation_rate=0.35,
		tournament_k=3,
		max_depth=6,
		max_gates=10,
		diversity_floor=0.02,
		stagnation_generations=6,
		target_fitness_threshold=0.999,
		target_kind="unitary",
		behavior_weight=0.6,
		robustness_weight=0.35,
		complexity_weight=0.05,
		fidelity_weight=0.7,
		frobenius_weight=0.3,
		qpc_enabled=True,
		qpc_mode="exact",
		working_qubits=(0,),
		candidate_clusters=((0, 1),),
		candidate_cluster_weights=(1.0,),
	)

	result = evolutionary_best_circuit(
		config=config,
		gate_catalog=gate_catalog,
		target_circuit=None,
		target_matrix=target_matrix,
		noise_model=noise_model,
		seed=1234,
	)

	evolved = fitness_circuit(
		result.best_circuit,
		target=None,
		target_matrix=target_matrix,
		target_kind="unitary",
		noise_model=noise_model,
		qpc_enabled=True,
		qpc_mode="exact",
		behavior_weight=0.6,
		robustness_weight=0.35,
		complexity_weight=0.05,
		working_qubits=(0,),
	)

	print()
	print("Given circuit (workload):")
	_print_circuit("Given", given_circuit)

	print()
	print("Baseline under noisy QPC model:")
	print(f"  total fitness      : {baseline.total_fitness:.6f}")
	print(f"  behavior score     : {baseline.behavior_score:.6f}")
	print(f"  robustness score   : {baseline.robustness_score:.6f}")
	print(f"  complexity score   : {baseline.complexity_score:.6f}")

	print()
	print("Best evolved correction candidate:")
	_print_circuit("Evolved", result.best_circuit)
	print(f"  stop reason        : {result.stop_reason}")
	print(f"  generations run    : {result.generations_run}")
	print(f"  cache entries      : {result.cache_entries}")

	print()
	print("Evolved under same noisy QPC model:")
	print(f"  total fitness      : {evolved.total_fitness:.6f}")
	print(f"  behavior score     : {evolved.behavior_score:.6f}")
	print(f"  robustness score   : {evolved.robustness_score:.6f}")
	print(f"  complexity score   : {evolved.complexity_score:.6f}")

	print()
	print("Improvement summary:")
	print(f"  total delta        : {evolved.total_fitness - baseline.total_fitness:+.6f}")
	print(f"  behavior delta     : {evolved.behavior_score - baseline.behavior_score:+.6f}")
	print(f"  robustness delta   : {evolved.robustness_score - baseline.robustness_score:+.6f}")
	print(f"  complexity delta   : {evolved.complexity_score - baseline.complexity_score:+.6f}")
	print()
	print("Note: working_qubits=(0,) means ancilla output is ignored in scoring.")
	print("Any extra structure on qubit 1 is treated as correction/helper logic.")


if __name__ == "__main__":
	main()
