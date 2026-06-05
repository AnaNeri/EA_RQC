"""Target-matrix parsing and validation for circuit EA."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MatrixTarget:
	target_kind: str
	matrix: np.ndarray
	num_qubits: int


def _parse_complex(value: Any) -> complex:
	if isinstance(value, (int, float)):
		return complex(float(value), 0.0)
	if isinstance(value, list) and len(value) == 2 and all(isinstance(part, (int, float)) for part in value):
		return complex(float(value[0]), float(value[1]))
	raise ValueError(f"Unsupported matrix element format: {value!r}")


def _parse_matrix(raw_matrix: Any) -> np.ndarray:
	if not isinstance(raw_matrix, list) or not raw_matrix:
		raise ValueError("matrix must be a non-empty 2D list")

	parsed_rows: list[list[complex]] = []
	row_length = None
	for row in raw_matrix:
		if not isinstance(row, list) or not row:
			raise ValueError("matrix rows must be non-empty lists")
		if row_length is None:
			row_length = len(row)
		elif len(row) != row_length:
			raise ValueError("matrix rows must all have the same length")
		parsed_rows.append([_parse_complex(value) for value in row])

	matrix = np.array(parsed_rows, dtype=np.complex128)
	if matrix.shape[0] != matrix.shape[1]:
		raise ValueError("matrix must be square")
	return matrix


def _validate_dimensions(kind: str, matrix: np.ndarray, num_qubits: int) -> None:
	rows, cols = matrix.shape
	if rows != cols:
		raise ValueError("matrix must be square")

	if kind == "unitary":
		expected = 2**num_qubits
	elif kind == "channel":
		expected = (2**num_qubits) ** 2
	else:
		raise ValueError(f"Unsupported target_kind: {kind}")

	if rows != expected:
		raise ValueError(f"matrix dimension {rows} does not match expected {expected} for kind={kind}, num_qubits={num_qubits}")


def load_matrix_target_json(path: str | Path) -> MatrixTarget:
	payload = json.loads(Path(path).read_text(encoding="utf-8"))
	kind = str(payload.get("target_kind", "unitary")).lower().strip()
	matrix = _parse_matrix(payload.get("matrix"))

	num_qubits_raw = payload.get("num_qubits")
	if num_qubits_raw is None:
		size = matrix.shape[0]
		if kind == "unitary":
			num_qubits = int(round(math.log2(size)))
		else:
			num_qubits = int(round(math.log2(math.sqrt(size))))
	else:
		num_qubits = int(num_qubits_raw)

	_validate_dimensions(kind, matrix, num_qubits)
	return MatrixTarget(target_kind=kind, matrix=matrix, num_qubits=num_qubits)
