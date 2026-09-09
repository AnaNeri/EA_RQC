"""Fitness functions for circuit-level evolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from circuit.entities.circuit_list import CircuitList
from .matrix_eval import (
	apply_channel_to_state,
	circuit_to_matrix,
	circuit_to_faulty_channel,
	infer_num_qubits_from_matrix,
	partial_trace_density_matrix,
	state_fidelity,
	subsystem_similarity,
	unitary_to_superoperator,
)
from .targets import SampleTarget


@dataclass(frozen=True)
class CircuitFitnessBreakdown:
	behavior_score: float
	robustness_score: float
	complexity_score: float
	total_fitness: float
	# Optional QPC diagnostics
	qpc_distance: float = 0.0
	approximation_error: float = 0.0
	# Optional sample-based scoring
	sample_score: float = 0.0


def _gate_noise_penalty(circuit: CircuitList, noise_model: Any) -> float:
	if noise_model is None:
		return 0.0

	rates = getattr(noise_model, "qubit_error_rates", noise_model)
	if not isinstance(rates, dict):
		return 0.0
	gate_error_rates = getattr(noise_model, "gate_error_rates", None)

	total = 0.0
	count = 0
	for gate in circuit.gates:
		gate_factor = 1.0
		if gate_error_rates is not None:
			gate_factor = float(gate_error_rates.get(gate.name.lower().strip(), 0.0))
		for qubit in gate.qubits:
			value = rates.get(qubit, rates.get(str(qubit), 0.0))
			total += float(value) * gate_factor
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
	working_qubits: Sequence[int] | None = None,
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
	fidelity, frobenius = subsystem_similarity(
		candidate_matrix,
		effective_target_matrix,
		target_kind=target_kind,
		num_qubits=num_qubits,
		working_qubits=working_qubits,
	)

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
	working_qubits: Sequence[int] | None = None,
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
	gate_error_rates = None
	if noise_model is not None:
		error_rates = getattr(noise_model, "qubit_error_rates", {})
		gate_error_rates = getattr(noise_model, "gate_error_rates", None)
	
	# Ideal channel
	ideal_channel = circuit_to_matrix(circuit, target_kind="channel", num_qubits=num_qubits)
	
	qpc_distance = 0.0
	approximation_error = 0.0
	
	if qpc_mode in ("exact", "mixed"):
		# Build faulty channel from gate-wise depolarizing
		faulty_channel = circuit_to_faulty_channel(
			circuit,
			error_rates=error_rates,
			gate_error_rates=gate_error_rates,
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
				gate_error_rates=gate_error_rates,
				num_qubits=num_qubits,
			)
			exact_channel = circuit_to_faulty_channel(
				circuit,
				error_rates=error_rates,
				gate_error_rates=gate_error_rates,
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

		# Exact mode evaluates the explicit composed faulty channel for the circuit.
		# It does not call mix_channels() directly; the same algebraic rules still
		# come from linear channel/superoperator operations under the hood.
		candidate_matrix = circuit_to_faulty_channel(
			circuit,
			error_rates=error_rates,
			gate_error_rates=gate_error_rates,
			num_qubits=num_qubits,
		)
		comparison_kind = "channel"
	else:
		# approx/mixed mode: use ideal unitaries for faster comparison
		# (Approximation mode doesn't penalize against faulty channels directly)
		target_for_comparison = effective_target_matrix
		candidate_matrix = circuit_to_matrix(circuit, target_kind=target_kind, num_qubits=num_qubits)
		comparison_kind = target_kind

	fidelity, frobenius = subsystem_similarity(
		candidate_matrix,
		target_for_comparison,
		target_kind=comparison_kind,
		num_qubits=num_qubits,
		working_qubits=working_qubits,
	)

	weight_sum = fidelity_weight + frobenius_weight
	if weight_sum <= 0:
		return 0.0, qpc_distance, approximation_error
	fw = fidelity_weight / weight_sum
	frw = frobenius_weight / weight_sum
	behavior = max(0.0, min(1.0, (fw * fidelity) + (frw * frobenius)))
	
	return behavior, qpc_distance, approximation_error



def _sample_score(
	circuit: CircuitList,
	*,
	samples: SampleTarget,
	noise_model: Any,
	qpc_enabled: bool,
	qpc_mode: str,
	working_qubits: Sequence[int] | None = None,
) -> float:
	"""Compute mean phase-sensitive fidelity over all input→output sample pairs.

	Without QPC: applies the ideal unitary and measures real(Re(\u27e8output|U|input\u27e9)) scaled to [0,1].
	Phase-sensitive: identity maps |11\u27e9 \u2192 |11\u27e9 scores lower than CZ which maps |11\u27e9 \u2192 -|11\u27e9.
	With QPC: applies the faulty channel and measures channel-output vs target via density matrix overlap.
	"""
	if not samples.samples:
		return 1.0

	def _infer_circuit_num_qubits() -> int:
		inferred = max((qubit for gate in circuit.gates for qubit in gate.qubits), default=-1) + 1
		inferred = max(inferred, len(circuit.cluster), 1)
		if working_qubits:
			inferred = max(inferred, max(int(qubit) for qubit in working_qubits) + 1)
		return inferred

	def _normalise_working_qubits(num_qubits: int) -> tuple[int, ...]:
		if not working_qubits:
			return tuple(range(num_qubits))
		normalised = tuple(sorted(int(qubit) for qubit in working_qubits))
		if len(set(normalised)) != len(normalised):
			raise ValueError("working_qubits must not contain duplicates")
		if any(qubit < 0 or qubit >= num_qubits for qubit in normalised):
			raise ValueError("working_qubits contains out-of-range qubit indices")
		return normalised

	def _embed_working_state_on_full_register(
		state: np.ndarray,
		*,
		num_qubits: int,
		keep_qubits: Sequence[int],
	) -> np.ndarray:
		if len(keep_qubits) == num_qubits:
			return np.asarray(state, dtype=np.complex128)
		expected_dim = 2 ** len(keep_qubits)
		reduced = np.asarray(state, dtype=np.complex128).reshape(expected_dim)
		full = np.zeros(2**num_qubits, dtype=np.complex128)
		for basis_index, amplitude in enumerate(reduced):
			if abs(amplitude) < 1e-15:
				continue
			full_index = 0
			for bit_pos, qubit in enumerate(keep_qubits):
				if (basis_index >> bit_pos) & 1:
					full_index |= 1 << int(qubit)
			full[full_index] = amplitude
		return full

	full_num_qubits = _infer_circuit_num_qubits()
	keep_qubits = _normalise_working_qubits(full_num_qubits)
	if len(keep_qubits) != samples.num_qubits:
		raise ValueError("samples.num_qubits must match len(working_qubits)")

	error_rates: dict[int, float] = {}
	gate_error_rates: dict[str, float] | None = None
	if noise_model is not None:
		error_rates = getattr(noise_model, "qubit_error_rates", {})
		gate_error_rates = getattr(noise_model, "gate_error_rates", None)

	if qpc_enabled:
		superop = circuit_to_faulty_channel(
			circuit,
			error_rates=error_rates,
			gate_error_rates=gate_error_rates,
			num_qubits=full_num_qubits,
		)
		scores = []
		for s in samples.samples:
			input_state_full = _embed_working_state_on_full_register(
				s.input_state,
				num_qubits=full_num_qubits,
				keep_qubits=keep_qubits,
			)
			rho_out_full = apply_channel_to_state(superop, input_state_full)
			rho_out = partial_trace_density_matrix(
				rho_out_full,
				num_qubits=full_num_qubits,
				keep_qubits=keep_qubits,
			)
			score = state_fidelity(rho_out, s.output_state)
			scores.append(score)
	else:
		unitary = circuit_to_matrix(circuit, target_kind="unitary", num_qubits=full_num_qubits)
		# Phase-sensitive: compare each output column of U against target output state.
		# Use real part of complex overlap ⟨target|U|input⟩, scaled to [0,1] via (1 + Re(overlap)) / 2.
		scores = []
		for s in samples.samples:
			input_state_full = _embed_working_state_on_full_register(
				s.input_state,
				num_qubits=full_num_qubits,
				keep_qubits=keep_qubits,
			)
			output_full = unitary @ input_state_full
			rho_out_full = np.outer(output_full, output_full.conjugate())
			rho_out = partial_trace_density_matrix(
				rho_out_full,
				num_qubits=full_num_qubits,
				keep_qubits=keep_qubits,
			)
			score = state_fidelity(rho_out, s.output_state)
			scores.append(score)

	return sum(scores) / len(scores)


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
	samples: SampleTarget | None = None,
	sample_weight: float = 0.0,
	working_qubits: Sequence[int] | None = None,
) -> CircuitFitnessBreakdown:
	"""Compute a multi-objective fitness in [0, 1].

	Args:
		qpc_enabled: if True, use probabilistic combinator channels for behavior scoring.
		qpc_mode: "exact", "approx", or "mixed" for QPC evaluation mode.
		samples: optional SampleTarget for sample-based synthesis.
		sample_weight: weight of sample score in total fitness (0 = disabled).
	"""
	effective_sample_weight = sample_weight if (samples is not None and sample_weight > 0.0) else 0.0
	weight_sum = behavior_weight + robustness_weight + complexity_weight + effective_sample_weight
	if weight_sum <= 0:
		raise ValueError("fitness weights must sum to a positive value")

	bw = behavior_weight / weight_sum
	rw = robustness_weight / weight_sum
	cw = complexity_weight / weight_sum
	sw = effective_sample_weight / weight_sum

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
			working_qubits=working_qubits,
		)
	else:
		behavior = _behavior_score(
			circuit,
			target=target,
			target_matrix=target_matrix,
			target_kind=target_kind,
			fidelity_weight=fidelity_weight,
			frobenius_weight=frobenius_weight,
			working_qubits=working_qubits,
		)
		qpc_distance = 0.0
		approx_error = 0.0

	robustness = 1.0 / (1.0 + max(0.0, _gate_noise_penalty(circuit, noise_model)))
	complexity = _complexity_score(circuit)

	sample = (
		_sample_score(
			circuit,
			samples=samples,
			noise_model=noise_model,
			qpc_enabled=qpc_enabled,
			qpc_mode=qpc_mode,
			working_qubits=working_qubits,
		)
		if sw > 0.0
		else 0.0
	)

	total = (bw * behavior) + (rw * robustness) + (cw * complexity) + (sw * sample)
	return CircuitFitnessBreakdown(
		behavior_score=behavior,
		robustness_score=robustness,
		complexity_score=complexity,
		total_fitness=max(0.0, min(1.0, total)),
		qpc_distance=qpc_distance,
		approximation_error=approx_error,
		sample_score=sample,
	)
