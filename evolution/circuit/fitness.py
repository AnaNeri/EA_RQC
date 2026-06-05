"""Fitness functions for circuit-level evolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from circuit.entities.circuit_list import CircuitList
from .matrix_eval import (
	circuit_to_matrix,
	circuit_to_faulty_channel,
	fidelity_similarity,
	frobenius_similarity,
	infer_num_qubits_from_matrix,
	mix_channels,
	unitary_to_superoperator,
)


@dataclass(frozen=True)
class CircuitFitnessBreakdown:
	behavior_score: float
	robustness_score: float
	complexity_score: float
	total_fitness: float
	# Optional QPC diagnostics
	qpc_distance: float = 0.0
	approximation_error: float = 0.0


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


def _behavior_score(
	circuit: CircuitList,
	*,
	target: CircuitList | None,
	target_matrix: np.ndarray | None,
	target_kind: str,
	fidelity_weight: float,
	frobenius_weight: float,
) -> float:
	if target_matrix is None and target is None:
		return 1.0

	effective_target_matrix = target_matrix
	if effective_target_matrix is None and target is not None:
		effective_target_matrix = circuit_to_matrix(target, target_kind=target_kind)

	if effective_target_matrix is None:
		return 1.0

	num_qubits = infer_num_qubits_from_matrix(effective_target_matrix, target_kind=target_kind)
	candidate_matrix = circuit_to_matrix(circuit, target_kind=target_kind, num_qubits=num_qubits)
	fidelity = fidelity_similarity(candidate_matrix, effective_target_matrix)
	frobenius = frobenius_similarity(candidate_matrix, effective_target_matrix)

	weight_sum = fidelity_weight + frobenius_weight
	if weight_sum <= 0:
		return 0.0
	fw = fidelity_weight / weight_sum
	frw = frobenius_weight / weight_sum
	return max(0.0, min(1.0, (fw * fidelity) + (frw * frobenius)))


def _behavior_score_with_qpc(
	circuit: CircuitList,
	*,
	target: CircuitList | None,
	target_matrix: np.ndarray | None,
	target_kind: str,
	fidelity_weight: float,
	frobenius_weight: float,
	noise_model: Any,
	qpc_mode: str = "exact",
) -> tuple[float, float, float]:
	"""Compute behavior score using probabilistic combinator channels.
	
	Args:
		qpc_mode: "exact" for full gate-wise QPC, "approx" for first-order approx, "mixed" for both.
	
	Returns:
		(behavior_score, qpc_distance, approximation_error).
	"""
	if target_matrix is None and target is None:
		return 1.0, 0.0, 0.0

	effective_target_matrix = target_matrix
	if effective_target_matrix is None and target is not None:
		effective_target_matrix = circuit_to_matrix(target, target_kind=target_kind)

	if effective_target_matrix is None:
		return 1.0, 0.0, 0.0

	num_qubits = infer_num_qubits_from_matrix(effective_target_matrix, target_kind=target_kind)
	
	# Get error rates from noise model
	error_rates = {}
	if noise_model is not None:
		error_rates = getattr(noise_model, "qubit_error_rates", {})
	
	# Ideal channel
	ideal_unitary = circuit_to_matrix(circuit, target_kind="unitary", num_qubits=num_qubits)
	ideal_channel = circuit_to_matrix(circuit, target_kind="channel", num_qubits=num_qubits)
	
	qpc_distance = 0.0
	approximation_error = 0.0
	
	if qpc_mode in ("exact", "mixed"):
		# Build faulty channel from gate-wise depolarizing
		faulty_channel = circuit_to_faulty_channel(
			circuit,
			error_rates=error_rates,
			num_qubits=num_qubits,
		)
		
		# Distance between ideal and faulty
		qpc_distance = float(np.linalg.norm(faulty_channel - ideal_channel, "fro"))
	
	if qpc_mode in ("approx", "mixed"):
		# First-order approximation: only consider zero-error and one-error paths
		# Simplified: use a reduced error probability for approximation check
		mean_error = sum(error_rates.values()) / max(len(error_rates), 1) if error_rates else 0.0
		
		# Approximate as single composite error
		approx_p = mean_error * len(circuit.gates)
		if approx_p > 0.0 and approx_p < 0.2:
			approx_channel = circuit_to_faulty_channel(
				circuit,
				error_rates={q: approx_p for q in error_rates},
				num_qubits=num_qubits,
			)
			exact_channel = circuit_to_faulty_channel(
				circuit,
				error_rates=error_rates,
				num_qubits=num_qubits,
			)
			approximation_error = float(np.linalg.norm(approx_channel - exact_channel, "fro"))
	
	# Determine target representation based on mode
	# In exact mode, we compare channels (superoperators)
	# In approx/mixed mode, we compare unitaries (simpler/faster)
	if qpc_mode == "exact":
		# Convert target to channel space for comparison
		if target_kind == "unitary":
			target_for_comparison = unitary_to_superoperator(effective_target_matrix)
		else:
			target_for_comparison = effective_target_matrix
		
		candidate_matrix = circuit_to_faulty_channel(
			circuit,
			error_rates=error_rates,
			num_qubits=num_qubits,
		)
	else:
		# approx/mixed mode: use ideal unitaries for faster comparison
		# (Approximation mode doesn't penalize against faulty channels directly)
		target_for_comparison = effective_target_matrix
		candidate_matrix = circuit_to_matrix(circuit, target_kind=target_kind, num_qubits=num_qubits)
	
	fidelity = fidelity_similarity(candidate_matrix, target_for_comparison)
	frobenius = frobenius_similarity(candidate_matrix, target_for_comparison)

	weight_sum = fidelity_weight + frobenius_weight
	if weight_sum <= 0:
		return 0.0, qpc_distance, approximation_error
	fw = fidelity_weight / weight_sum
	frw = frobenius_weight / weight_sum
	behavior = max(0.0, min(1.0, (fw * fidelity) + (frw * frobenius)))
	
	return behavior, qpc_distance, approximation_error



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
	target_matrix: np.ndarray | None = None,
	target_kind: str = "unitary",
	noise_model: Any,
	behavior_weight: float = 0.5,
	robustness_weight: float = 0.35,
	complexity_weight: float = 0.15,
	fidelity_weight: float = 0.7,
	frobenius_weight: float = 0.3,
	qpc_enabled: bool = False,
	qpc_mode: str = "exact",
) -> CircuitFitnessBreakdown:
	"""Compute a multi-objective fitness in [0, 1].
	
	Args:
		qpc_enabled: if True, use probabilistic combinator channels for behavior scoring.
		qpc_mode: "exact", "approx", or "mixed" for QPC evaluation mode.
	"""

	weight_sum = behavior_weight + robustness_weight + complexity_weight
	if weight_sum <= 0:
		raise ValueError("fitness weights must sum to a positive value")

	bw = behavior_weight / weight_sum
	rw = robustness_weight / weight_sum
	cw = complexity_weight / weight_sum

	if qpc_enabled:
		behavior, qpc_distance, approx_error = _behavior_score_with_qpc(
			circuit,
			target=target,
			target_matrix=target_matrix,
			target_kind=target_kind,
			fidelity_weight=fidelity_weight,
			frobenius_weight=frobenius_weight,
			noise_model=noise_model,
			qpc_mode=qpc_mode,
		)
	else:
		behavior = _behavior_score(
			circuit,
			target=target,
			target_matrix=target_matrix,
			target_kind=target_kind,
			fidelity_weight=fidelity_weight,
			frobenius_weight=frobenius_weight,
		)
		qpc_distance = 0.0
		approx_error = 0.0

	robustness = 1.0 / (1.0 + max(0.0, _gate_noise_penalty(circuit, noise_model)))
	complexity = _complexity_score(circuit)

	total = (bw * behavior) + (rw * robustness) + (cw * complexity)
	return CircuitFitnessBreakdown(
		behavior_score=behavior,
		robustness_score=robustness,
		complexity_score=complexity,
		total_fitness=max(0.0, min(1.0, total)),
		qpc_distance=qpc_distance,
		approximation_error=approx_error,
	)
