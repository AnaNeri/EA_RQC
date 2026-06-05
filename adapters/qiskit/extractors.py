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


def extract_backend_base_gates(
    backend: Any,
    properties: Any | None = None,
    *,
    include_non_unitary: bool = False,
) -> list[str]:
    """Extract gate names directly from backend metadata.

    Sources are consulted in order:
    1) configuration().basis_gates
    2) target.operation_names
    3) backend.operation_names
    4) properties.gates[*].name
    """

    names: list[str] = []

    configuration = getattr(backend, "configuration", None)
    if callable(configuration):
        try:
            config = configuration()
            basis = getattr(config, "basis_gates", None)
            if isinstance(basis, (list, tuple, set)):
                names.extend(str(item) for item in basis if item is not None)
        except Exception:
            pass

    target = getattr(backend, "target", None)
    if target is not None:
        operation_names = getattr(target, "operation_names", None)
        if callable(operation_names):
            try:
                operation_names = operation_names()
            except TypeError:
                operation_names = None
        if isinstance(operation_names, (list, tuple, set)):
            names.extend(str(item) for item in operation_names if item is not None)

    backend_operation_names = getattr(backend, "operation_names", None)
    if callable(backend_operation_names):
        try:
            backend_operation_names = backend_operation_names()
        except TypeError:
            backend_operation_names = None
    if isinstance(backend_operation_names, (list, tuple, set)):
        names.extend(str(item) for item in backend_operation_names if item is not None)

    if properties is None:
        properties_fn = getattr(backend, "properties", None)
        if callable(properties_fn):
            try:
                properties = properties_fn()
            except Exception:
                properties = None

    if properties is not None and hasattr(properties, "gates"):
        try:
            gates = list(properties.gates)
        except Exception:
            gates = []
        for gate in gates:
            gate_name = getattr(gate, "name", None)
            if gate_name:
                names.append(str(gate_name))

    normalized = [name.lower().strip() for name in names if str(name).strip()]

    if not include_non_unitary:
        excluded = {"measure", "barrier", "delay", "reset", "snapshot"}
        normalized = [name for name in normalized if name not in excluded]

    unique = list(dict.fromkeys(normalized))
    return unique
