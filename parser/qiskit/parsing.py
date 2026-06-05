"""Qiskit backend parser entry points."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .channels import (
    amplitude_damping_kraus,
    compose_channels,
    depolarizing_kraus,
    dephasing_probability,
    phase_flip_kraus,
    safe_float,
    spam_bitflip_kraus,
    spam_confusion_matrix,
    thermal_dephasing_model,
    thermal_probability,
)
from .extractors import backend_name, backend_num_qubits, backend_version, extract_qubit_properties, extract_single_qubit_gate_data
from .models import ParsedDeviceNoiseModel, QubitNoiseMap


def parse_qiskit_backend_noise_model(
    backend: Any,
    *,
    date: str | None = None,
    default_gate_time_seconds: float = 100e-9,
) -> ParsedDeviceNoiseModel:
    """Build a per-qubit noise model from a Qiskit backend-like object."""

    properties_fn = getattr(backend, "properties", None)
    if callable(properties_fn):
        try:
            properties = properties_fn()
        except Exception:
            properties = None
    else:
        properties = None

    num_qubits = backend_num_qubits(backend, properties)
    if num_qubits <= 0:
        raise ValueError("Could not determine number of qubits from backend")

    qubit_properties = extract_qubit_properties(properties, num_qubits)
    gate_data = extract_single_qubit_gate_data(properties, num_qubits)

    qubits: dict[str, QubitNoiseMap] = {}
    qubit_error_rates: dict[int, float] = {}

    for qubit in range(num_qubits):
        props = qubit_properties.get(qubit, {})
        gates = gate_data.get(qubit, {})

        t1 = safe_float(props.get("T1", props.get("t1", 0.0)), default=0.0)
        t2 = safe_float(props.get("T2", props.get("t2", 0.0)), default=0.0)

        gate_time = safe_float(gates.get("gate_time_avg", 0.0), default=0.0)
        if gate_time <= 0.0:
            gate_time = default_gate_time_seconds

        thermal_prob = thermal_probability(t1, gate_time)
        dephasing_prob = dephasing_probability(t1, t2, gate_time)
        thermal_dephasing = thermal_dephasing_model(t1, t2, gate_time)

        gate_error_avg = safe_float(gates.get("gate_error_avg", 0.0), default=0.0)
        if dephasing_prob <= 0.0 and gate_error_avg > 0.0:
            dephasing_prob = max(0.0, min(1.0, gate_error_avg))

        depolarizing_prob = max(0.0, min(1.0, gate_error_avg))

        readout_error = safe_float(props.get("readout_error", 0.0), default=0.0)
        p01 = safe_float(props.get("prob_meas0_prep1", readout_error / 2.0), default=readout_error / 2.0)
        p10 = safe_float(props.get("prob_meas1_prep0", readout_error / 2.0), default=readout_error / 2.0)
        spam_prob = max(0.0, min(1.0, p01 + p10))

        thermal_channel = amplitude_damping_kraus(thermal_prob)
        spam_pauli_channel = spam_bitflip_kraus(spam_prob)
        spam_readout_channel = spam_confusion_matrix(p01, p10)
        dephasing_channel = phase_flip_kraus(dephasing_prob)
        depolarizing_channel = depolarizing_kraus(depolarizing_prob)

        qubit_noise: QubitNoiseMap = {
            "thermal_noise": [(thermal_prob, thermal_channel)],
            "spam": [(spam_prob, spam_pauli_channel)],
            "spam_readout": [(spam_prob, spam_readout_channel)],
            "dephasing": [(dephasing_prob, dephasing_channel)],
            "depolarizing": [(depolarizing_prob, depolarizing_channel)],
            "thermal_dephasing": [(max(thermal_prob, dephasing_prob), thermal_dephasing)],
            "combo_depolarizing_spam": [
                (max(depolarizing_prob, spam_prob), compose_channels(depolarizing_channel, spam_pauli_channel))
            ],
            "combo_depolarizing_thermal_dephasing": [
                (
                    max(depolarizing_prob, thermal_prob, dephasing_prob),
                    compose_channels(depolarizing_channel, thermal_channel, dephasing_channel),
                )
            ],
            "combo_all": [
                (
                    max(depolarizing_prob, spam_prob, thermal_prob, dephasing_prob),
                    compose_channels(depolarizing_channel, spam_pauli_channel, thermal_channel, dephasing_channel),
                )
            ],
        }

        qubits[str(qubit)] = qubit_noise
        qubit_error_rates[qubit] = max(0.0, min(1.0, thermal_prob + spam_prob + dephasing_prob + depolarizing_prob))

    return ParsedDeviceNoiseModel(
        device_name=backend_name(backend),
        version=backend_version(backend),
        date=date or datetime.now(tz=timezone.utc).isoformat(),
        qubits=qubits,
        qubit_error_rates=qubit_error_rates,
    )
