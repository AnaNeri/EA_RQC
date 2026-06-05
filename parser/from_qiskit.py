"""Compatibility layer for legacy imports.

Prefer importing from parser.qiskit.
"""

from .qiskit import (
    ParsedDeviceNoiseModel,
    load_noise_model_json,
    parse_qiskit_backend_gate_catalog,
    parse_qiskit_backend_noise_model,
    parse_qiskit_fake_backend,
    save_noise_model_json,
)

__all__ = [
    "ParsedDeviceNoiseModel",
    "parse_qiskit_backend_gate_catalog",
    "parse_qiskit_backend_noise_model",
    "parse_qiskit_fake_backend",
    "save_noise_model_json",
    "load_noise_model_json",
]

