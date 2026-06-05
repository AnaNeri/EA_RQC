"""Minimal graph projection for list-based circuits."""

from __future__ import annotations

from dataclasses import dataclass

from .circuit_list import CircuitList


@dataclass(frozen=True)
class CircuitEdge:
	source: int
	target: int
	qubit: int
	edge_type: str = "sequential"


@dataclass
class CircuitGraph:
	"""Directed acyclic view of a list-based circuit.

	Nodes are gate indices from the source list.
	"""

	nodes: list[int]
	edges: list[CircuitEdge]

	@staticmethod
	def from_circuit_list(circuit: CircuitList) -> "CircuitGraph":
		nodes = list(range(len(circuit.gates)))
		edges: list[CircuitEdge] = []
		last_on_qubit: dict[int, int] = {}

		for gate_index, gate in enumerate(circuit.gates):
			for qubit in gate.qubits:
				if qubit in last_on_qubit:
					edges.append(CircuitEdge(source=last_on_qubit[qubit], target=gate_index, qubit=qubit))
				last_on_qubit[qubit] = gate_index

		return CircuitGraph(nodes=nodes, edges=edges)

