"""Circuit-level genetic operators.

This module keeps the current cluster EA untouched and provides an additional
operator set for list-based circuit evolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Sequence

from circuit.entities.circuit_list import CircuitGate, CircuitList


@dataclass(frozen=True)
class AnchorBlock:
	start1: int
	end1: int
	start2: int
	end2: int
	weight: float


def _as_rng(seed: int | None) -> Random:
	return Random(seed)


def _gate_weight(gate: CircuitGate) -> float:
	weight = 1.0
	if len(gate.qubits) > 1:
		weight += 0.5
	if gate.parameters:
		weight += 0.25
	if gate.immutable:
		weight += 1.0
	return weight


def weighted_lcs_indices(parent1: CircuitList, parent2: CircuitList) -> list[tuple[int, int]]:
	"""Return index matches from a weighted LCS over gate signatures."""

	sig1 = parent1.signatures(include_depth=False)
	sig2 = parent2.signatures(include_depth=False)
	n = len(sig1)
	m = len(sig2)

	dp = [[0.0] * (m + 1) for _ in range(n + 1)]

	for i in range(1, n + 1):
		for j in range(1, m + 1):
			if sig1[i - 1] == sig2[j - 1]:
				gain = (_gate_weight(parent1.gates[i - 1]) + _gate_weight(parent2.gates[j - 1])) / 2.0
				dp[i][j] = dp[i - 1][j - 1] + gain
			else:
				dp[i][j] = dp[i - 1][j] if dp[i - 1][j] >= dp[i][j - 1] else dp[i][j - 1]

	indices: list[tuple[int, int]] = []
	i = n
	j = m
	while i > 0 and j > 0:
		if sig1[i - 1] == sig2[j - 1]:
			prev = dp[i - 1][j - 1]
			gain = (_gate_weight(parent1.gates[i - 1]) + _gate_weight(parent2.gates[j - 1])) / 2.0
			if abs(dp[i][j] - (prev + gain)) < 1e-9:
				indices.append((i - 1, j - 1))
				i -= 1
				j -= 1
				continue
		if dp[i - 1][j] >= dp[i][j - 1]:
			i -= 1
		else:
			j -= 1

	indices.reverse()
	return indices


def _blocks_from_matches(matches: list[tuple[int, int]], parent1: CircuitList, parent2: CircuitList) -> list[AnchorBlock]:
	if not matches:
		return []

	blocks: list[AnchorBlock] = []
	start1, start2 = matches[0]
	end1, end2 = start1, start2

	for i1, i2 in matches[1:]:
		if i1 == end1 + 1 and i2 == end2 + 1:
			end1, end2 = i1, i2
			continue

		block_weight = sum(_gate_weight(gate) for gate in parent1.gates[start1 : end1 + 1])
		block_weight += sum(_gate_weight(gate) for gate in parent2.gates[start2 : end2 + 1])
		block_weight /= max(2.0, float((end1 - start1 + 1) + (end2 - start2 + 1)))
		blocks.append(AnchorBlock(start1=start1, end1=end1, start2=start2, end2=end2, weight=block_weight))

		start1, start2 = i1, i2
		end1, end2 = i1, i2

	block_weight = sum(_gate_weight(gate) for gate in parent1.gates[start1 : end1 + 1])
	block_weight += sum(_gate_weight(gate) for gate in parent2.gates[start2 : end2 + 1])
	block_weight /= max(2.0, float((end1 - start1 + 1) + (end2 - start2 + 1)))
	blocks.append(AnchorBlock(start1=start1, end1=end1, start2=start2, end2=end2, weight=block_weight))
	return blocks


def find_anchor_blocks(parent1: CircuitList, parent2: CircuitList) -> list[AnchorBlock]:
	"""Public helper that exposes matched anchor blocks."""

	return _blocks_from_matches(weighted_lcs_indices(parent1, parent2), parent1, parent2)


def _remap_gate_to_cluster(gate: CircuitGate, source_cluster: Sequence[int], target_cluster: Sequence[int]) -> CircuitGate:
	if tuple(source_cluster) == tuple(target_cluster):
		return gate

	mapping = {source: target for source, target in zip(source_cluster, target_cluster)}
	remapped_qubits = tuple(mapping.get(qubit, qubit) for qubit in gate.qubits)
	return CircuitGate.from_values(
		name=gate.name,
		qubits=remapped_qubits,
		parameters=gate.parameters,
		depth=gate.depth,
		immutable=gate.immutable,
	)


def _remap_gates_to_cluster(
	gates: Sequence[CircuitGate],
	*,
	source_cluster: Sequence[int],
	target_cluster: Sequence[int],
) -> list[CircuitGate]:
	return [_remap_gate_to_cluster(gate, source_cluster, target_cluster) for gate in gates]


def probabilistic_anchor_crossover(
	parent1: CircuitList,
	parent2: CircuitList,
	*,
	crossover_rate: float = 0.9,
	base_keep_probability: float = 0.7,
	seed: int | None = None,
) -> tuple[CircuitList, CircuitList]:
	"""Crossover likely to preserve shared structures, not guaranteed.

	Shared structures are discovered as weighted-LCS anchor blocks and each
	anchor is kept with a probability that increases with its relevance.
	"""

	rng = _as_rng(seed)
	if rng.random() > crossover_rate:
		child1 = parent1.clone()
		child2 = parent2.clone()
		child1.crop_to_limits()
		child2.crop_to_limits()
		return child1, child2

	matches = weighted_lcs_indices(parent1, parent2)
	blocks = _blocks_from_matches(matches, parent1, parent2)

	if not blocks:
		split1 = rng.randrange(len(parent1.gates) + 1)
		split2 = rng.randrange(len(parent2.gates) + 1)
		child1 = CircuitList(
			gates=parent1.gates[:split1] + _remap_gates_to_cluster(
				parent2.gates[split2:],
				source_cluster=parent2.cluster,
				target_cluster=parent1.cluster,
			),
			cluster=parent1.cluster,
			max_depth=parent1.max_depth,
			max_gates=parent1.max_gates,
		)
		child2 = CircuitList(
			gates=parent2.gates[:split2] + _remap_gates_to_cluster(
				parent1.gates[split1:],
				source_cluster=parent1.cluster,
				target_cluster=parent2.cluster,
			),
			cluster=parent2.cluster,
			max_depth=parent2.max_depth,
			max_gates=parent2.max_gates,
		)
		child1.crop_to_limits()
		child2.crop_to_limits()
		return child1, child2

	max_weight = max(block.weight for block in blocks)
	selected: list[AnchorBlock] = []
	for block in blocks:
		relative = block.weight / max_weight if max_weight > 0 else 1.0
		keep_prob = min(0.98, max(0.05, base_keep_probability * relative))
		if rng.random() <= keep_prob:
			selected.append(block)

	if not selected:
		selected.append(blocks[rng.randrange(len(blocks))])

	selected.sort(key=lambda block: block.start1)

	child1_gates: list[CircuitGate] = []
	child2_gates: list[CircuitGate] = []
	p1_cursor = 0
	p2_cursor = 0

	for block in selected:
		gap1 = parent1.gates[p1_cursor : block.start1]
		gap2 = parent2.gates[p2_cursor : block.start2]

		if rng.random() < 0.5:
			child1_gates.extend(gap1)
			child2_gates.extend(gap2)
		else:
			child1_gates.extend(_remap_gates_to_cluster(gap2, source_cluster=parent2.cluster, target_cluster=parent1.cluster))
			child2_gates.extend(_remap_gates_to_cluster(gap1, source_cluster=parent1.cluster, target_cluster=parent2.cluster))

		child1_gates.extend(parent1.gates[block.start1 : block.end1 + 1])
		child2_gates.extend(parent2.gates[block.start2 : block.end2 + 1])

		p1_cursor = block.end1 + 1
		p2_cursor = block.end2 + 1

	tail1 = parent1.gates[p1_cursor:]
	tail2 = parent2.gates[p2_cursor:]
	if rng.random() < 0.5:
		child1_gates.extend(tail1)
		child2_gates.extend(tail2)
	else:
		child1_gates.extend(_remap_gates_to_cluster(tail2, source_cluster=parent2.cluster, target_cluster=parent1.cluster))
		child2_gates.extend(_remap_gates_to_cluster(tail1, source_cluster=parent1.cluster, target_cluster=parent2.cluster))

	child1 = CircuitList(
		gates=child1_gates,
		cluster=parent1.cluster,
		max_depth=parent1.max_depth,
		max_gates=parent1.max_gates,
	)
	child2 = CircuitList(
		gates=child2_gates,
		cluster=parent2.cluster,
		max_depth=parent2.max_depth,
		max_gates=parent2.max_gates,
	)
	child1.crop_to_limits()
	child2.crop_to_limits()
	return child1, child2


def random_circuit(
	*,
	cluster: Sequence[int],
	max_depth: int,
	max_gates: int,
	gate_catalog: Sequence[str],
	seed: int | None = None,
) -> CircuitList:
	"""Create a random list-based circuit within cluster/depth constraints."""

	rng = _as_rng(seed)
	cluster_list = [int(qubit) for qubit in cluster]
	gates: list[CircuitGate] = []
	gate_count = rng.randint(1, max(1, max_gates))

	one_qubit_gates = {"x", "y", "z", "h", "sx", "rx", "ry", "rz", "s", "sdg", "t", "tdg", "p", "phase", "u", "id"}
	two_qubit_gates = {"cx", "cz", "ecr", "cy", "ch", "swap"}
	parameterized_1 = {"rx", "ry", "rz", "p", "phase"}
	parameterized_3 = {"u"}

	for _ in range(gate_count):
		name = str(rng.choice(list(gate_catalog))).lower().strip()
		depth = rng.randint(1, max(1, max_depth))

		if name in two_qubit_gates and len(cluster_list) >= 2:
			qubits = tuple(rng.sample(cluster_list, 2))
		elif name in one_qubit_gates or len(cluster_list) < 2:
			qubits = (rng.choice(cluster_list),)
		else:
			qubits = (rng.choice(cluster_list),)

		if name in parameterized_3:
			parameters = tuple(rng.uniform(-3.1416, 3.1416) for _ in range(3))
		elif name in parameterized_1:
			parameters = (rng.uniform(-3.1416, 3.1416),)
		else:
			parameters = ()

		gates.append(CircuitGate.from_values(name=name, qubits=qubits, parameters=parameters, depth=depth))

	circuit = CircuitList(gates=gates, cluster=tuple(cluster_list), max_depth=max_depth, max_gates=max_gates)
	circuit.crop_to_limits()
	return circuit


def mutate_circuit(
	circuit: CircuitList,
	*,
	device_qubits: int,
	mutation_rate: float,
	gate_catalog: Sequence[str],
	seed: int | None = None,
) -> CircuitList:
	"""Mutate a circuit with add/remove/change-parameter/change-cluster moves."""

	rng = _as_rng(seed)
	child = circuit.clone()

	def _finalize(candidate: CircuitList) -> CircuitList:
		candidate.crop_to_limits()
		return candidate

	if rng.random() > mutation_rate:
		return _finalize(child)

	mutation = rng.choice(["add", "remove", "change_params", "change_cluster"])
	parameterized_1 = {"rx", "ry", "rz", "p", "phase"}
	parameterized_3 = {"u"}
	parameterized = parameterized_1 | parameterized_3

	mutable_indices = [index for index, gate in enumerate(child.gates) if not gate.immutable]

	if mutation == "remove" and mutable_indices:
		remove_index = rng.choice(mutable_indices)
		del child.gates[remove_index]
		return _finalize(child)

	if mutation == "change_params" and mutable_indices:
		candidate_indices = [index for index in mutable_indices if child.gates[index].name in parameterized]
		if candidate_indices:
			gate_index = rng.choice(candidate_indices)
			gate = child.gates[gate_index]
			updated = list(gate.parameters) if gate.parameters else [0.0]
			updated[0] = float(updated[0]) + rng.uniform(-0.5, 0.5)
			child.gates[gate_index] = CircuitGate.from_values(
				name=gate.name,
				qubits=gate.qubits,
				parameters=tuple(updated),
				depth=gate.depth,
				immutable=gate.immutable,
			)
			return _finalize(child)

	if mutation == "change_cluster" and child.cluster:
		cluster_size = len(child.cluster)
		if cluster_size <= device_qubits:
			new_cluster = tuple(sorted(rng.sample(range(device_qubits), cluster_size)))
			remap = {old: new for old, new in zip(child.cluster, new_cluster)}
			remapped_gates: list[CircuitGate] = []
			for gate in child.gates:
				remapped_qubits = tuple(remap.get(qubit, qubit) for qubit in gate.qubits)
				remapped_gates.append(
					CircuitGate.from_values(
						name=gate.name,
						qubits=remapped_qubits,
						parameters=gate.parameters,
						depth=gate.depth,
						immutable=gate.immutable,
					)
				)
			child.gates = remapped_gates
			child.cluster = new_cluster
			return _finalize(child)

	cluster_list = list(child.cluster) if child.cluster else [0]
	new_gate = random_circuit(
		cluster=cluster_list,
		max_depth=max(1, child.max_depth or 1),
		max_gates=1,
		gate_catalog=gate_catalog,
		seed=rng.randrange(2**32),
	).gates[0]
	insert_at = rng.randrange(len(child.gates) + 1)
	child.gates.insert(insert_at, new_gate)
	return _finalize(child)
