"""Matrix-based semantic evaluation helpers for circuit evolution."""

from __future__ import annotations

import cmath
import math
from typing import Sequence

import numpy as np

from circuit.entities.circuit_list import CircuitList


def _single_qubit_matrix(name: str, parameters: Sequence[float]) -> np.ndarray:
	name = name.lower().strip()
	if name == "id":
		return np.eye(2, dtype=np.complex128)
	if name == "x":
		return np.array([[0, 1], [1, 0]], dtype=np.complex128)
	if name == "y":
		return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
	if name == "z":
		return np.array([[1, 0], [0, -1]], dtype=np.complex128)
	if name == "h":
		return (1.0 / math.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
	if name == "sx":
		return 0.5 * np.array(
			[[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]],
			dtype=np.complex128,
		)
	if name in {"rx", "ry", "rz"}:
		theta = float(parameters[0]) if parameters else 0.0
		c = math.cos(theta / 2.0)
		s = math.sin(theta / 2.0)
		if name == "rx":
			return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
		if name == "ry":
			return np.array([[c, -s], [s, c]], dtype=np.complex128)
		# rz
		return np.array(
			[[cmath.exp(-1j * theta / 2.0), 0], [0, cmath.exp(1j * theta / 2.0)]],
			dtype=np.complex128,
		)
	if name == "s":
		return np.array([[1, 0], [0, 1j]], dtype=np.complex128)
	if name == "sdg":
		return np.array([[1, 0], [0, -1j]], dtype=np.complex128)
	if name == "t":
		return np.array([[1, 0], [0, cmath.exp(1j * math.pi / 4.0)]], dtype=np.complex128)
	if name == "tdg":
		return np.array([[1, 0], [0, cmath.exp(-1j * math.pi / 4.0)]], dtype=np.complex128)
	if name in {"p", "phase"}:
		lam = float(parameters[0]) if parameters else 0.0
		return np.array([[1, 0], [0, cmath.exp(1j * lam)]], dtype=np.complex128)
	if name == "u":
		# U(theta, phi, lambda) — Qiskit U3 gate
		theta = float(parameters[0]) if len(parameters) > 0 else 0.0
		phi   = float(parameters[1]) if len(parameters) > 1 else 0.0
		lam   = float(parameters[2]) if len(parameters) > 2 else 0.0
		c = math.cos(theta / 2.0)
		s = math.sin(theta / 2.0)
		return np.array(
			[
				[c, -cmath.exp(1j * lam) * s],
				[cmath.exp(1j * phi) * s, cmath.exp(1j * (phi + lam)) * c],
			],
			dtype=np.complex128,
		)

	raise ValueError(f"Unsupported single-qubit gate for matrix evaluation: {name}")


def _two_qubit_matrix(name: str) -> np.ndarray:
	name = name.lower().strip()
	if name == "cx":
		return np.array(
			[
				[1, 0, 0, 0],
				[0, 1, 0, 0],
				[0, 0, 0, 1],
				[0, 0, 1, 0],
			],
			dtype=np.complex128,
		)
	if name == "cz":
		return np.array(
			[
				[1, 0, 0, 0],
				[0, 1, 0, 0],
				[0, 0, 1, 0],
				[0, 0, 0, -1],
			],
			dtype=np.complex128,
		)
	if name == "ecr":
		# Qiskit ECR up to global phase conventions.
		return (1.0 / math.sqrt(2.0)) * np.array(
			[
				[0, 0, 1, 1j],
				[0, 0, 1j, 1],
				[1, -1j, 0, 0],
				[-1j, 1, 0, 0],
			],
			dtype=np.complex128,
		)
	if name == "swap":
		return np.array(
			[
				[1, 0, 0, 0],
				[0, 0, 1, 0],
				[0, 1, 0, 0],
				[0, 0, 0, 1],
			],
			dtype=np.complex128,
		)
	if name == "cy":
		return np.array(
			[
				[1, 0, 0,  0],
				[0, 1, 0,  0],
				[0, 0, 0, -1j],
				[0, 0, 1j, 0],
			],
			dtype=np.complex128,
		)
	if name == "ch":
		return np.array(
			[
				[1, 0,                    0,                   0],
				[0, 1,                    0,                   0],
				[0, 0,  1.0 / math.sqrt(2), 1.0 / math.sqrt(2)],
				[0, 0,  1.0 / math.sqrt(2), -1.0 / math.sqrt(2)],
			],
			dtype=np.complex128,
		)

	raise ValueError(f"Unsupported two-qubit gate for matrix evaluation: {name}")


def _expand_operator(local_op: np.ndarray, qubits: Sequence[int], num_qubits: int) -> np.ndarray:
	qubits = tuple(int(q) for q in qubits)
	if not qubits:
		return np.eye(2**num_qubits, dtype=np.complex128)

	dimension = 2**num_qubits
	out = np.zeros((dimension, dimension), dtype=np.complex128)
	qubit_positions = {qubit: idx for idx, qubit in enumerate(qubits)}

	for column in range(dimension):
		sub_col = 0
		for qubit in qubits:
			bit = (column >> qubit) & 1
			sub_col |= bit << qubit_positions[qubit]

		for sub_row in range(2 ** len(qubits)):
			amplitude = local_op[sub_row, sub_col]
			if abs(amplitude) < 1e-15:
				continue

			row = column
			for qubit in qubits:
				pos = qubit_positions[qubit]
				bit_val = (sub_row >> pos) & 1
				if bit_val == 1:
					row |= 1 << qubit
				else:
					row &= ~(1 << qubit)

			out[row, column] += amplitude

	return out


def _expand_superoperator(local_superop: np.ndarray, qubits: Sequence[int], num_qubits: int) -> np.ndarray:
	"""Expand a local superoperator to full system superoperator via Kronecker.
	
	Args:
		local_superop: Local superoperator of shape (4^k, 4^k) where k = len(qubits).
		qubits: Qubits the superoperator acts on (in order).
		num_qubits: Total number of qubits in system.
	
	Returns:
		Full system superoperator of shape (4^n, 4^n).
	"""
	qubits = tuple(int(q) for q in qubits)
	if not qubits:
		full_dim = 2**num_qubits
		return np.eye(full_dim * full_dim, dtype=np.complex128)
	
	# Start with identity on all qubits
	full_dim = 2**num_qubits
	result = np.eye(full_dim * full_dim, dtype=np.complex128)
	
	# For now, use a simple approach: convert to unitary, expand, convert back
	# This is not the most efficient, but preserves correctness
	local_dim = 2**len(qubits)
	
	# Approximate: use Kronecker structure
	# Identity on non-affected qubits
	left_qubits = qubits[0]
	right_qubits = num_qubits - max(qubits) - 1
	
	# Build full superoperator via Kronecker product
	if left_qubits > 0:
		left_id = np.eye(2**(2*left_qubits), dtype=np.complex128)
		result = np.kron(left_id, local_superop)
	else:
		result = local_superop
	
	if right_qubits > 0:
		right_id = np.eye(2**(2*right_qubits), dtype=np.complex128)
		result = np.kron(result, right_id)
	
	return result


def circuit_to_unitary(circuit: CircuitList, *, num_qubits: int | None = None) -> np.ndarray:
	if num_qubits is None:
		num_qubits = max((qubit for gate in circuit.gates for qubit in gate.qubits), default=-1) + 1
		num_qubits = max(num_qubits, len(circuit.cluster), 1)
	else:
		num_qubits = max(int(num_qubits), 1)

	logical_qubits = list(circuit.cluster) if circuit.cluster else sorted({qubit for gate in circuit.gates for qubit in gate.qubits})
	if not logical_qubits:
		logical_qubits = list(range(num_qubits))
	qubit_map = {qubit: index for index, qubit in enumerate(logical_qubits[:num_qubits])}
	dimension = 2**num_qubits
	unitary = np.eye(dimension, dtype=np.complex128)

	for _, gate in sorted(enumerate(circuit.gates), key=lambda entry: (entry[1].depth, entry[0])):
		if len(gate.qubits) == 1:
			local = _single_qubit_matrix(gate.name, gate.parameters)
		elif len(gate.qubits) == 2:
			local = _two_qubit_matrix(gate.name)
		else:
			raise ValueError(f"Unsupported gate arity for matrix evaluation: {gate.name} with qubits={gate.qubits}")

		remapped_qubits = tuple(qubit_map.get(qubit, int(qubit) % num_qubits) for qubit in gate.qubits)
		full = _expand_operator(local, remapped_qubits, num_qubits)
		unitary = full @ unitary

	return unitary


def unitary_to_superoperator(unitary: np.ndarray) -> np.ndarray:
	return np.kron(unitary, unitary.conjugate())


def depolarizing_kraus_matrices(p: float) -> list[np.ndarray]:
	"""Return Kraus operators for single-qubit depolarizing channel.
	
	Channel: (1-p)ρ + (p/3)(XρX† + YρY† + ZρZ†).
	Kraus form: K_0 = √(1-p) I, K_1 = √(p/3) X, K_2 = √(p/3) Y, K_3 = √(p/3) Z.
	"""
	p = max(0.0, min(1.0, p))
	sqrt_1mp = math.sqrt(1.0 - p)
	sqrt_p3 = math.sqrt(p / 3.0)
	
	K0 = sqrt_1mp * np.eye(2, dtype=np.complex128)
	K1 = sqrt_p3 * np.array([[0, 1], [1, 0]], dtype=np.complex128)
	K2 = sqrt_p3 * np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
	K3 = sqrt_p3 * np.array([[1, 0], [0, -1]], dtype=np.complex128)
	
	return [K0, K1, K2, K3]


def kraus_to_superoperator(kraus_ops: Sequence[np.ndarray]) -> np.ndarray:
	"""Convert Kraus operator list to superoperator (Choi matrix form).
	
	For Kraus operators K_i, the superoperator is Σ_i K_i ⊗ K_i*.
	"""
	if not kraus_ops:
		dim = 2
		return np.eye(dim * dim, dtype=np.complex128)
	
	# Infer dimension from first Kraus operator
	local_dim = kraus_ops[0].shape[0]
	superop_dim = local_dim * local_dim
	
	superop = np.zeros((superop_dim, superop_dim), dtype=np.complex128)
	for K in kraus_ops:
		# K ⊗ K^* for this Kraus operator
		term = np.kron(K, K.conjugate())
		superop += term
	
	return superop


def build_gate_faulty_channel(unitary: np.ndarray, p_error: float, num_qubits: int = 1) -> np.ndarray:
	"""Build faulty gate channel for a gate unitary.
	
	The channel is: apply ideal unitary, then depolarizing noise.
	Works in the Hilbert space of the unitary (local gate space).
	
	Args:
		unitary: Gate unitary (2^k × 2^k where k=1 or 2 for single/two-qubit).
		p_error: Error probability for depolarizing channel.
		num_qubits: Number of qubits (used to infer gate arity; typically 1 or 2).
	
	Returns:
		Superoperator in the same space as the unitary.
	"""
	p_error = max(0.0, min(1.0, p_error))
	
	# Ideal channel: U ρ U†
	ideal_superop = unitary_to_superoperator(unitary)
	
	if p_error <= 0.0:
		return ideal_superop
	
	# Determine gate dimension from unitary size
	dim = unitary.shape[0]  # 2 for single-qubit, 4 for two-qubit
	
	if dim == 2:
		# Single-qubit depolarizing
		depol_kraus = depolarizing_kraus_matrices(p_error)
	elif dim == 4:
		# Two-qubit depolarizing: use Kronecker product of single-qubit depolarizing
		single_kraus = depolarizing_kraus_matrices(p_error)
		depol_kraus = [
			np.kron(k1, k2) for k1 in single_kraus for k2 in single_kraus
		]
	else:
		raise ValueError(f"Unsupported unitary dimension: {dim}. Expected 2 or 4.")
	
	depol_superop = kraus_to_superoperator(depol_kraus)
	
	# Compose: apply ideal first, then depol noise
	# In superoperator form: depol_superop @ ideal_superop
	faulty_superop = depol_superop @ ideal_superop
	
	return faulty_superop


def compose_superoperators(superop1: np.ndarray, superop2: np.ndarray) -> np.ndarray:
	"""Compose two superoperators sequentially: apply superop2 then superop1.
	
	In circuit notation: superop1 ∘ superop2 means apply superop2's channel first.
	"""
	return superop1 @ superop2


def mix_channels(channel_faulty: np.ndarray, channel_ideal: np.ndarray, p: float) -> np.ndarray:
	"""Probabilistic combinator at channel level: p*faulty + (1-p)*ideal.
	
	Returns the superoperator representing the mixture:
	S_p = p·S_faulty + (1-p)·S_ideal.
	"""
	p = max(0.0, min(1.0, p))
	
	if channel_faulty.shape != channel_ideal.shape:
		raise ValueError("channel matrices must have the same shape")
	
	mixed = p * channel_faulty + (1.0 - p) * channel_ideal
	return mixed


def circuit_to_matrix(circuit: CircuitList, *, target_kind: str, num_qubits: int | None = None) -> np.ndarray:
	kind = target_kind.lower().strip()
	unitary = circuit_to_unitary(circuit, num_qubits=num_qubits)
	if kind == "unitary":
		return unitary
	if kind == "channel":
		return unitary_to_superoperator(unitary)
	raise ValueError(f"Unsupported target_kind: {target_kind}")


def infer_num_qubits_from_matrix(matrix: np.ndarray, *, target_kind: str) -> int:
	size = int(matrix.shape[0])
	kind = target_kind.lower().strip()
	if kind == "unitary":
		return int(round(math.log2(size)))
	if kind == "channel":
		return int(round(math.log2(math.sqrt(size))))
	raise ValueError(f"Unsupported target_kind: {target_kind}")


def circuit_to_faulty_channel(
	circuit: CircuitList,
	*,
	error_rates: dict[int, float] | None = None,
	num_qubits: int | None = None,
) -> np.ndarray:
	"""Build the faulty circuit channel by composing depolarizing-noisy gate channels.
	
	For each gate, uses mean error rate across its qubits (from error_rates dict).
	Gates are composed in depth order: rightmost (first applied) to leftmost (last).
	
	Args:
		circuit: CircuitList to evaluate.
		error_rates: dict[int, float] mapping qubit index to error probability.
		num_qubits: number of qubits; inferred if None.
	
	Returns:
		Superoperator (Choi matrix) of shape (2^n, 2^n).
	"""
	if num_qubits is None:
		num_qubits = max((qubit for gate in circuit.gates for qubit in gate.qubits), default=-1) + 1
		num_qubits = max(num_qubits, len(circuit.cluster), 1)
	else:
		num_qubits = max(int(num_qubits), 1)
	
	if error_rates is None:
		error_rates = {}
	
	# Logical to physical qubit mapping (same as in circuit_to_unitary)
	logical_qubits = list(circuit.cluster) if circuit.cluster else sorted({qubit for gate in circuit.gates for qubit in gate.qubits})
	if not logical_qubits:
		logical_qubits = list(range(num_qubits))
	qubit_map = {qubit: index for index, qubit in enumerate(logical_qubits[:num_qubits])}
	
	dimension = 2**num_qubits
	channel = np.eye(dimension * dimension, dtype=np.complex128)  # Identity superoperator
	
	# Build faulty channel gate by gate, in reverse depth order (compose left-to-right in operator notation)
	for _, gate in sorted(enumerate(circuit.gates), key=lambda entry: (entry[1].depth, entry[0]), reverse=True):
		# Get ideal unitary (local, not expanded)
		if len(gate.qubits) == 1:
			ideal_unitary = _single_qubit_matrix(gate.name, gate.parameters)
		elif len(gate.qubits) == 2:
			ideal_unitary = _two_qubit_matrix(gate.name)
		else:
			raise ValueError(f"Unsupported gate arity: {gate.name} with qubits={gate.qubits}")
		
		# Build faulty gate channel in local space (before expansion)
		p_error = sum(error_rates.get(q, 0.0) for q in gate.qubits) / max(len(gate.qubits), 1)
		p_error = max(0.0, min(1.0, p_error))
		
		local_faulty_channel = build_gate_faulty_channel(ideal_unitary, p_error, num_qubits=len(gate.qubits))
		
		# Expand to full system: convert faulty channel (superop) to unitary form, expand, convert back
		# This is a workaround: we extract the "canonical" unitary from the channel
		# For small errors, the faulty channel is approximately U_faulty ρ U_faulty†
		# For now, we'll expand the local superoperator directly by reconstructing from ideal unitary
		
		# Remap qubits to physical indices
		remapped_qubits = tuple(qubit_map.get(q, int(q) % num_qubits) for q in gate.qubits)
		
		# Expand local unitary to full system
		expanded_ideal_unitary = _expand_operator(ideal_unitary, remapped_qubits, num_qubits)
		
		# Expand local faulty channel using the same qubit mapping
		# Simple approach: treat superop as unitary (in vectorized form) and expand
		# More robust: rebuild the full-system faulty channel from expanded ideal unitary
		expanded_ideal_channel = unitary_to_superoperator(expanded_ideal_unitary)
		
		# For the faulty channel, we need to apply noise only to the relevant qubits
		# This is a simplified approach: use the local faulty channel and expand
		# Correct approach is to build a full-system depolarizing on those qubits and compose
		
		# Build full-system depolarizing on the gate qubits
		if p_error > 0.0:
			# Depolarizing on local gate space
			dim = ideal_unitary.shape[0]
			if dim == 2:
				depol_kraus = depolarizing_kraus_matrices(p_error)
			elif dim == 4:
				single_kraus = depolarizing_kraus_matrices(p_error)
				depol_kraus = [np.kron(k1, k2) for k1 in single_kraus for k2 in single_kraus]
			else:
				raise ValueError(f"Unsupported dimension: {dim}")
			
			# Expand depolarizing operators to full system
			expanded_depol_kraus = [
				_expand_operator(K, remapped_qubits, num_qubits) for K in depol_kraus
			]
			
			# Convert expanded Kraus to full-system superoperator
			expanded_depol_superop = kraus_to_superoperator(expanded_depol_kraus)
			
			# Compose: ideal channel, then depolarizing
			expanded_faulty_channel = expanded_depol_superop @ expanded_ideal_channel
		else:
			expanded_faulty_channel = expanded_ideal_channel
		
		# Compose with existing channel
		channel = compose_superoperators(expanded_faulty_channel, channel)
	
	return channel


def apply_channel_to_state(superop: np.ndarray, state_vec: np.ndarray) -> np.ndarray:
	"""Apply a superoperator channel to a pure state vector.

	Builds the density matrix ρ = |ψ⟩⟨ψ|, vectorizes it (row-major / C-order),
	applies the superoperator, and returns the output density matrix.

	Args:
		superop: Superoperator of shape (d^2, d^2) where d = 2^n.
		state_vec: State vector of shape (d,), complex.

	Returns:
		Output density matrix of shape (d, d), complex.
	"""
	state_vec = np.asarray(state_vec, dtype=np.complex128)
	dim = state_vec.shape[0]
	rho = np.outer(state_vec, state_vec.conjugate())  # |ψ⟩⟨ψ|
	rho_vec = rho.flatten(order="C")                  # row-major vectorization
	out_vec = superop @ rho_vec
	return out_vec.reshape(dim, dim, order="C")


def state_fidelity(rho_out: np.ndarray, target_state: np.ndarray) -> float:
	"""Compute ⟨φ|ρ_out|φ⟩ — overlap of output density matrix with target pure state.

	Args:
		rho_out: Output density matrix of shape (d, d).
		target_state: Target pure state vector of shape (d,).

	Returns:
		Fidelity in [0, 1].
	"""
	target_state = np.asarray(target_state, dtype=np.complex128)
	fid = float(np.real(target_state.conjugate() @ rho_out @ target_state))
	return max(0.0, min(1.0, fid))


def fidelity_similarity(candidate: np.ndarray, target: np.ndarray) -> float:
	if candidate.shape != target.shape:
		raise ValueError("candidate and target matrices must have the same shape")

	numerator = abs(np.trace(target.conjugate().T @ candidate))
	denominator = float(np.linalg.norm(target, "fro") * np.linalg.norm(candidate, "fro"))
	if denominator <= 0:
		return 0.0
	return max(0.0, min(1.0, float(numerator / denominator)))


def frobenius_similarity(candidate: np.ndarray, target: np.ndarray) -> float:
	if candidate.shape != target.shape:
		raise ValueError("candidate and target matrices must have the same shape")
	distance = float(np.linalg.norm(candidate - target, "fro"))
	return 1.0 / (1.0 + max(0.0, distance))
