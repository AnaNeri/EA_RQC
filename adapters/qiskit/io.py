"""Persistence helpers for parsed Qiskit noise models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .channels import safe_float
from .models import NoiseEntry, ParsedDeviceNoiseModel, QubitNoiseMap


def save_noise_model_json(model: ParsedDeviceNoiseModel | dict[str, Any], output_path: str | Path) -> Path:
    """Save a parsed noise model to disk as JSON."""

    path = Path(output_path)
    serializable = model.to_dict() if isinstance(model, ParsedDeviceNoiseModel) else model
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return path


def load_noise_model_json(input_path: str | Path) -> ParsedDeviceNoiseModel:
    """Load a JSON noise model from disk."""

    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    qubits_payload = payload.get("qubits", {})

    qubits: dict[str, QubitNoiseMap] = {}
    for qubit, noise_types in qubits_payload.items():
        qubits[str(qubit)] = {}
        for noise_type, entries in noise_types.items():
            parsed_entries: list[NoiseEntry] = []
            for entry in entries:
                if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                    continue
                parsed_entries.append((safe_float(entry[0], default=0.0), dict(entry[1])))
            qubits[str(qubit)][str(noise_type)] = parsed_entries

    qubit_error_rates_raw = payload.get("qubit_error_rates", {})
    qubit_error_rates = {int(key): safe_float(value, default=0.0) for key, value in qubit_error_rates_raw.items()}

    if not qubit_error_rates:
        for qubit_key, noise_types in qubits.items():
            total = 0.0
            for entries in noise_types.values():
                total += sum(prob for prob, _ in entries)
            qubit_error_rates[int(qubit_key)] = max(0.0, min(1.0, total))

    return ParsedDeviceNoiseModel(
        device_name=str(payload.get("device_name", "unknown")),
        version=str(payload.get("version", "unknown")),
        date=str(payload.get("date", "unknown")),
        qubits=qubits,
        qubit_error_rates=qubit_error_rates,
    )
