"""Data structures for parsed Qiskit noise models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


NoiseEntry = tuple[float, dict[str, Any]]
QubitNoiseMap = dict[str, list[NoiseEntry]]


@dataclass(frozen=True)
class ParsedDeviceNoiseModel:
    """Container for parsed device metadata and per-qubit noise channels."""

    device_name: str
    version: str
    date: str
    qubits: dict[str, QubitNoiseMap]
    qubit_error_rates: dict[int, float]

    def to_dict(self) -> dict[str, Any]:
        serialized_qubits: dict[str, dict[str, list[list[Any]]]] = {}
        for qubit, noise_types in self.qubits.items():
            serialized_qubits[qubit] = {}
            for noise_type, entries in noise_types.items():
                serialized_qubits[qubit][noise_type] = [[float(prob), matrix] for prob, matrix in entries]

        return {
            "device_name": self.device_name,
            "version": self.version,
            "date": self.date,
            "qubits": serialized_qubits,
            "qubit_error_rates": {str(key): value for key, value in self.qubit_error_rates.items()},
        }
