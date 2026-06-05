"""Backend metadata and property extraction helpers."""

from __future__ import annotations

from typing import Any

from .channels import safe_float, unit_factor_to_seconds


def backend_name(backend: Any) -> str:
    name_attr = getattr(backend, "name", None)
    if callable(name_attr):
        try:
            return str(name_attr())
        except TypeError:
            pass
    if isinstance(name_attr, str) and name_attr:
        return name_attr
    return backend.__class__.__name__


def backend_version(backend: Any) -> str:
    for attr in ("backend_version", "version"):
        value = getattr(backend, attr, None)
        if callable(value):
            try:
                value = value()
            except TypeError:
                value = None
        if value is not None:
            return str(value)
    return "unknown"


def backend_num_qubits(backend: Any, properties: Any | None) -> int:
    direct = getattr(backend, "num_qubits", None)
    if isinstance(direct, int) and direct > 0:
        return direct

    configuration = getattr(backend, "configuration", None)
    if callable(configuration):
        try:
            config = configuration()
            count = getattr(config, "n_qubits", None)
            if isinstance(count, int) and count > 0:
                return count
        except Exception:
            pass

    if properties is not None and hasattr(properties, "qubits"):
        try:
            return len(properties.qubits)
        except Exception:
            return 0

    return 0


def extract_qubit_properties(properties: Any, num_qubits: int) -> dict[int, dict[str, float]]:
    data: dict[int, dict[str, float]] = {index: {} for index in range(num_qubits)}
    if properties is None or not hasattr(properties, "qubits"):
        return data

    try:
        qubits = list(properties.qubits)
    except Exception:
        return data

    for index, ndv_list in enumerate(qubits):
        if index >= num_qubits:
            break
        for ndv in ndv_list:
            key = getattr(ndv, "name", None)
            if not key:
                continue
            value = safe_float(getattr(ndv, "value", None), default=0.0)
            unit = getattr(ndv, "unit", None)
            if key.lower() in {"t1", "t2"}:
                value *= unit_factor_to_seconds(unit)
            data[index][str(key)] = value

    return data


def extract_single_qubit_gate_data(properties: Any, num_qubits: int) -> dict[int, dict[str, float]]:
    gate_data: dict[int, dict[str, float]] = {
        index: {"gate_error_avg": 0.0, "gate_error_count": 0.0, "gate_time_avg": 0.0, "gate_time_count": 0.0}
        for index in range(num_qubits)
    }
    if properties is None or not hasattr(properties, "gates"):
        return gate_data

    try:
        gates = list(properties.gates)
    except Exception:
        return gate_data

    for gate in gates:
        qubits = getattr(gate, "qubits", None)
        if not isinstance(qubits, (list, tuple)) or len(qubits) != 1:
            continue
        qubit = qubits[0]
        if not isinstance(qubit, int) or qubit < 0 or qubit >= num_qubits:
            continue

        for parameter in getattr(gate, "parameters", []):
            name = str(getattr(parameter, "name", "")).lower()
            value = safe_float(getattr(parameter, "value", None), default=0.0)
            if name == "gate_error":
                gate_data[qubit]["gate_error_avg"] += value
                gate_data[qubit]["gate_error_count"] += 1.0
            elif name in {"gate_length", "duration"}:
                unit = getattr(parameter, "unit", None)
                gate_data[qubit]["gate_time_avg"] += value * unit_factor_to_seconds(unit)
                gate_data[qubit]["gate_time_count"] += 1.0

    for qubit in range(num_qubits):
        error_count = gate_data[qubit]["gate_error_count"]
        if error_count > 0:
            gate_data[qubit]["gate_error_avg"] /= error_count
        time_count = gate_data[qubit]["gate_time_count"]
        if time_count > 0:
            gate_data[qubit]["gate_time_avg"] /= time_count

    return gate_data
