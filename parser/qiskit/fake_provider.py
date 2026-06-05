"""Helpers to load Qiskit fake-provider backends by name."""

from __future__ import annotations

import importlib

from .models import ParsedDeviceNoiseModel
from .parsing import parse_qiskit_backend_noise_model


def parse_qiskit_fake_backend(backend_name: str, *, date: str | None = None) -> ParsedDeviceNoiseModel:
    """Instantiate a Qiskit fake backend by class name and parse its noise model."""

    try:
        module = importlib.import_module("qiskit_ibm_runtime.fake_provider")
    except ModuleNotFoundError:
        module = importlib.import_module("qiskit.providers.fake_provider")

    backend_class = getattr(module, backend_name, None)
    if backend_class is None:
        raise ValueError(f"Unknown fake backend class: {backend_name}")

    backend = backend_class()
    return parse_qiskit_backend_noise_model(backend, date=date)
