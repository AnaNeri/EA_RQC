"""Fitness functions for circuit-level evolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from circuit.entities.circuit_list import CircuitList
from .operators import weighted_lcs_indices


@dataclass(frozen=True)
class CircuitFitnessBreakdown:
	behavior_score: float
	robustness_score: float
	complexity_score: float
	total_fitness: float


def _gate_noise_penalty(circuit: CircuitList, noise_model: Any) -> float:
	if noise_model is None:
		return 0.0

	rates = getattr(noise_model, "qubit_error_rates", noise_model)
	if not isinstance(rates, dict):
		return 0.0

	total = 0.0
	count = 0
	for gate in circuit.gates:
		for qubit in gate.qubits:
			value = rates.get(qubit, rates.get(str(qubit), 0.0))
			total += float(value)
			count += 1

	return total / count if count else 0.0


def _behavior_score(circuit: CircuitList, target: CircuitList | None) -> float:
	if target is None:
		return 1.0
	if not circuit.gates and not target.gates:
		return 1.0
	denominator = max(len(circuit.gates), len(target.gates), 1)
	match = len(weighted_lcs_indices(circuit, target))
	return max(0.0, min(1.0, match / denominator))


def _complexity_score(circuit: CircuitList) -> float:
	gate_limit = max(1, circuit.max_gates or len(circuit.gates) or 1)
	depth_limit = max(1, circuit.max_depth or max((gate.depth for gate in circuit.gates), default=1))

	gate_penalty = min(1.0, len(circuit.gates) / gate_limit)
	max_depth_seen = max((gate.depth for gate in circuit.gates), default=0)
	depth_penalty = min(1.0, max_depth_seen / depth_limit)

	return 1.0 - ((gate_penalty + depth_penalty) / 2.0)


def fitness_circuit(
	circuit: CircuitList,
	*,
	target: CircuitList | None,
	noise_model: Any,
	behavior_weight: float = 0.5,
	robustness_weight: float = 0.35,
	complexity_weight: float = 0.15,
) -> CircuitFitnessBreakdown:
	"""Compute a multi-objective fitness in [0, 1]."""

	weight_sum = behavior_weight + robustness_weight + complexity_weight
	if weight_sum <= 0:
		raise ValueError("fitness weights must sum to a positive value")

	bw = behavior_weight / weight_sum
	rw = robustness_weight / weight_sum
	cw = complexity_weight / weight_sum

	behavior = _behavior_score(circuit, target)
	robustness = 1.0 / (1.0 + max(0.0, _gate_noise_penalty(circuit, noise_model)))
	complexity = _complexity_score(circuit)

	total = (bw * behavior) + (rw * robustness) + (cw * complexity)
	return CircuitFitnessBreakdown(
		behavior_score=behavior,
		robustness_score=robustness,
		complexity_score=complexity,
		total_fitness=max(0.0, min(1.0, total)),
	)
