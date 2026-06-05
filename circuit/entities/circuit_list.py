"""List-based circuit genotype entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass(frozen=True)
class CircuitGate:
	"""Represents one circuit operation in list form."""

	name: str
	qubits: tuple[int, ...]
	parameters: tuple[float, ...] = ()
	depth: int = 0
	immutable: bool = False

	@staticmethod
	def from_values(
		name: str,
		qubits: Sequence[int],
		parameters: Sequence[float] | None = None,
		depth: int = 0,
		immutable: bool = False,
	) -> "CircuitGate":
		return CircuitGate(
			name=str(name).lower().strip(),
			qubits=tuple(int(qubit) for qubit in qubits),
			parameters=tuple(float(param) for param in (parameters or ())),
			depth=int(depth),
			immutable=bool(immutable),
		)

	def signature(self, *, precision: int = 3, include_depth: bool = False) -> tuple[object, ...]:
		"""Signature used by sequence matching algorithms."""

		params = tuple(round(value, precision) for value in self.parameters)
		signature: list[object] = [self.name, self.qubits, params]
		if include_depth:
			signature.append(self.depth)
		return tuple(signature)


@dataclass
class CircuitList:
	"""Simple list-based circuit genotype."""

	gates: list[CircuitGate] = field(default_factory=list)
	cluster: tuple[int, ...] = ()
	max_depth: int = 0
	max_gates: int | None = None

	def __len__(self) -> int:
		return len(self.gates)

	def clone(self) -> "CircuitList":
		return CircuitList(
			gates=list(self.gates),
			cluster=tuple(self.cluster),
			max_depth=self.max_depth,
			max_gates=self.max_gates,
		)

	@property
	def is_empty(self) -> bool:
		return len(self.gates) == 0

	def append(self, gate: CircuitGate) -> None:
		if self.max_gates is not None and len(self.gates) >= self.max_gates:
			return
		self.gates.append(gate)

	def extend(self, gates: Iterable[CircuitGate]) -> None:
		for gate in gates:
			self.append(gate)

	def crop_to_limits(self) -> None:
		if self.max_gates is not None and len(self.gates) > self.max_gates:
			self.gates = self.gates[: self.max_gates]
		if self.max_depth > 0:
			self.gates = [gate for gate in self.gates if gate.depth <= self.max_depth]

	def signatures(self, *, precision: int = 3, include_depth: bool = False) -> list[tuple[object, ...]]:
		return [gate.signature(precision=precision, include_depth=include_depth) for gate in self.gates]

