"""Noise-channel helpers and probability transforms."""

from __future__ import annotations

import math
from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def unit_factor_to_seconds(unit: str | None) -> float:
    lookup = {
        None: 1.0,
        "": 1.0,
        "s": 1.0,
        "ms": 1e-3,
        "us": 1e-6,
        "ns": 1e-9,
        "ps": 1e-12,
    }
    return lookup.get((unit or "").lower().strip(), 1.0)


def amplitude_damping_kraus(probability: float) -> dict[str, Any]:
    probability = max(0.0, min(1.0, probability))
    return {
        "channel": "amplitude_damping",
        "kraus": [
            [[1.0, 0.0], [0.0, math.sqrt(1.0 - probability)]],
            [[0.0, math.sqrt(probability)], [0.0, 0.0]],
        ],
    }


def phase_flip_kraus(probability: float) -> dict[str, Any]:
    probability = max(0.0, min(1.0, probability))
    return {
        "channel": "phase_flip",
        "kraus": [
            [[math.sqrt(1.0 - probability), 0.0], [0.0, math.sqrt(1.0 - probability)]],
            [[math.sqrt(probability), 0.0], [0.0, -math.sqrt(probability)]],
        ],
    }


def depolarizing_kraus(probability: float) -> dict[str, Any]:
    """Single-qubit depolarizing channel with p/3 Pauli branches."""

    probability = max(0.0, min(1.0, probability))
    return {
        "channel": "depolarizing",
        "pauli_probabilities": {
            "I": 1.0 - probability,
            "X": probability / 3.0,
            "Y": probability / 3.0,
            "Z": probability / 3.0,
        },
        "kraus_real_subset": [
            {
                "label": "K_D0",
                "operator": [[math.sqrt(1.0 - probability), 0.0], [0.0, math.sqrt(1.0 - probability)]],
            },
            {
                "label": "K_D1",
                "operator": [[0.0, math.sqrt(probability / 3.0)], [math.sqrt(probability / 3.0), 0.0]],
            },
            {
                "label": "K_D3",
                "operator": [[math.sqrt(probability / 3.0), 0.0], [0.0, -math.sqrt(probability / 3.0)]],
            },
        ],
        "notes": "Y branch is represented symbolically in pauli_probabilities to keep JSON serialization simple.",
    }


def spam_bitflip_kraus(probability: float) -> dict[str, Any]:
    """SPAM modeled as a Pauli-X channel with probability p."""

    probability = max(0.0, min(1.0, probability))
    return {
        "channel": "spam_bitflip",
        "kraus": [
            [[math.sqrt(1.0 - probability), 0.0], [0.0, math.sqrt(1.0 - probability)]],
            [[0.0, math.sqrt(probability)], [math.sqrt(probability), 0.0]],
        ],
    }


def spam_confusion_matrix(prob_meas0_prep1: float, prob_meas1_prep0: float) -> dict[str, Any]:
    p01 = max(0.0, min(1.0, prob_meas0_prep1))
    p10 = max(0.0, min(1.0, prob_meas1_prep0))
    return {
        "channel": "readout_confusion",
        "matrix": [
            [1.0 - p01, p01],
            [p10, 1.0 - p10],
        ],
    }


def thermal_probability(t1_seconds: float, gate_time_seconds: float) -> float:
    if t1_seconds <= 0.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - math.exp(-gate_time_seconds / t1_seconds)))


def dephasing_probability(t1_seconds: float, t2_seconds: float, gate_time_seconds: float) -> float:
    if t1_seconds <= 0.0 or t2_seconds <= 0.0:
        return 0.0
    dephasing_rate = max(0.0, (1.0 / t2_seconds) - (1.0 / (2.0 * t1_seconds)))
    if dephasing_rate <= 0.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - math.exp(-gate_time_seconds * dephasing_rate)))


def thermal_dephasing_model(t1_seconds: float, t2_seconds: float, gate_time_seconds: float) -> dict[str, Any]:
    """Return Georgopoulos-like thermal/dephasing model data.

    For T2 <= T1 we provide an explicit Kraus-like decomposition with I, Z and reset terms.
    For T1 <= T2 <= 2*T1 we provide the Choi matrix form used in the dissertation model.
    """

    if t1_seconds <= 0.0 or t2_seconds <= 0.0:
        return {
            "channel": "thermal_dephasing",
            "representation": "none",
            "p_t1": 0.0,
            "p_t2": 0.0,
            "p_reset": 0.0,
            "p_z": 0.0,
            "p_i": 1.0,
        }

    p_t1 = max(0.0, min(1.0, math.exp(-gate_time_seconds / t1_seconds)))
    p_t2 = max(0.0, min(1.0, math.exp(-gate_time_seconds / t2_seconds)))

    p_reset = max(0.0, min(1.0, 1.0 - p_t1))
    p_z = max(0.0, min(1.0, (1.0 - p_reset) * (1.0 - (p_t2 / max(p_t1, 1e-12)) / 2.0)))
    p_i = max(0.0, min(1.0, 1.0 - p_z - p_reset))

    if t2_seconds <= t1_seconds:
        return {
            "channel": "thermal_dephasing",
            "representation": "kraus",
            "p_t1": p_t1,
            "p_t2": p_t2,
            "p_reset": p_reset,
            "p_z": p_z,
            "p_i": p_i,
            "kraus": [
                {
                    "label": "K_I",
                    "operator": [[math.sqrt(p_i), 0.0], [0.0, math.sqrt(p_i)]],
                },
                {
                    "label": "K_Z",
                    "operator": [[math.sqrt(p_z), 0.0], [0.0, -math.sqrt(p_z)]],
                },
                {
                    "label": "K_reset",
                    "operator": [[math.sqrt(p_reset), 0.0], [0.0, 0.0]],
                },
            ],
        }

    if t2_seconds <= 2.0 * t1_seconds:
        return {
            "channel": "thermal_dephasing",
            "representation": "choi",
            "p_t1": p_t1,
            "p_t2": p_t2,
            "p_reset": p_reset,
            "p_z": p_z,
            "p_i": p_i,
            "choi": [
                [1.0, 0.0, 0.0, p_t2],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, p_reset, 0.0],
                [p_t2, 0.0, 0.0, 1.0 - p_reset],
            ],
        }

    return {
        "channel": "thermal_dephasing",
        "representation": "unsupported_range",
        "p_t1": p_t1,
        "p_t2": p_t2,
        "p_reset": p_reset,
        "p_z": p_z,
        "p_i": p_i,
    }


def compose_channels(*channels: dict[str, Any]) -> dict[str, Any]:
    names = [str(channel.get("channel", "unknown")) for channel in channels]
    return {
        "channel": "composed",
        "components": names,
        "sequence": list(channels),
    }
