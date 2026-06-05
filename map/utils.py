"""Shared helpers for qubit-cluster evolution."""

from __future__ import annotations

import random
from typing import Any, Iterable, Sequence

from .models import Individual


def as_rng(seed: int | None = None) -> random.Random:
    return random.Random(seed)


def normalise_cluster(cluster: Iterable[int]) -> Individual:
    return sorted(dict.fromkeys(int(qubit) for qubit in cluster))


def repair_cluster(cluster: Sequence[int], device_qubits: int, target_qubits: int, rng: random.Random) -> Individual:
    if device_qubits <= 0:
        raise ValueError("device_qubits must be greater than zero")
    if target_qubits <= 0:
        raise ValueError("target_qubits must be greater than zero")
    if target_qubits > device_qubits:
        raise ValueError("target_qubits cannot exceed device_qubits")

    available = list(range(device_qubits))
    repaired = normalise_cluster(qubit for qubit in cluster if 0 <= qubit < device_qubits)

    if len(repaired) > target_qubits:
        repaired = rng.sample(repaired, target_qubits)
        repaired.sort()
        return repaired

    missing = [qubit for qubit in available if qubit not in repaired]
    while len(repaired) < target_qubits and missing:
        selected = rng.choice(missing)
        missing.remove(selected)
        repaired.append(selected)

    repaired = normalise_cluster(repaired)

    while len(repaired) < target_qubits:
        candidate = rng.randrange(device_qubits)
        if candidate not in repaired:
            repaired.append(candidate)
            repaired.sort()

    return repaired


def cluster_noise_penalty(cluster: Sequence[int], noise_model: Any) -> float:
    if noise_model is None:
        return 0.0

    if hasattr(noise_model, "qubit_error_rates"):
        source = getattr(noise_model, "qubit_error_rates")
    elif hasattr(noise_model, "qubit_errors"):
        source = getattr(noise_model, "qubit_errors")
    elif hasattr(noise_model, "errors"):
        source = getattr(noise_model, "errors")
    else:
        source = noise_model

    if callable(source):
        penalties = [float(source(qubit)) for qubit in cluster]
        return sum(penalties) / len(penalties) if penalties else 0.0

    if isinstance(source, dict):
        penalties = []
        for qubit in cluster:
            if qubit in source:
                value = source[qubit]
            elif str(qubit) in source:
                value = source[str(qubit)]
            else:
                value = 0.0
            penalties.append(float(value))
        return sum(penalties) / len(penalties) if penalties else 0.0

    if isinstance(source, (list, tuple)):
        penalties = [float(source[qubit]) for qubit in cluster if 0 <= qubit < len(source)]
        return sum(penalties) / len(penalties) if penalties else 0.0

    penalties: list[float] = []
    for qubit in cluster:
        value = getattr(source, f"q{qubit}", None)
        if value is None:
            value = getattr(source, str(qubit), 0.0)
        penalties.append(float(value))
    return sum(penalties) / len(penalties) if penalties else 0.0


def _edge_error_from_graph(graph: Any, source: int, target: int) -> float | None:
    if graph is None:
        return None

    if hasattr(graph, "get_edge_data"):
        data = graph.get_edge_data(source, target, default=None)
        if data is None:
            return None
        if isinstance(data, dict):
            if "error" in data:
                return float(data["error"])
            if "weight" in data:
                return float(data["weight"])
            if data:
                first_value = next(iter(data.values()))
                if isinstance(first_value, dict):
                    return float(first_value.get("error", first_value.get("weight", 0.0)))
                return float(first_value)
            return None
        return float(data)

    if isinstance(graph, dict):
        if (source, target) in graph:
            return float(graph[(source, target)])
        if (target, source) in graph:
            return float(graph[(target, source)])
    return None


def topology_penalty(cluster: Sequence[int], coupling_graph: Any, coupling_path_penalty: float = 1.0) -> float:
    if coupling_graph is None:
        return 0.0

    penalty = 0.0
    ordered = list(cluster)
    for left, right in zip(ordered, ordered[1:]):
        direct_error = _edge_error_from_graph(coupling_graph, left, right)
        if direct_error is not None:
            penalty += direct_error
            continue

        path_error = None
        if hasattr(coupling_graph, "shortest_path_length"):
            try:
                path_length = coupling_graph.shortest_path_length(left, right)
                if path_length is not None and path_length > 1:
                    path_error = float(path_length - 1) * coupling_path_penalty
            except Exception:
                path_error = None

        if path_error is None:
            penalty += coupling_path_penalty
        else:
            penalty += path_error

    return penalty