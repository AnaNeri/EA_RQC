from __future__ import annotations

import numpy as np

from evolution.circuit.matrix_eval import compose_superoperators, mix_channels, unitary_to_superoperator


def _channel_from_unitary(unitary: np.ndarray) -> np.ndarray:
    return unitary_to_superoperator(np.asarray(unitary, dtype=np.complex128))


def _norm_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, "fro"))


def test_qpc_identity_unique_behavior() -> None:
    u = _channel_from_unitary(np.array([[0, 1], [1, 0]], dtype=np.complex128))
    p = 0.37

    lhs = mix_channels(u, u, p)
    rhs = u

    assert _norm_diff(lhs, rhs) < 1e-12


def test_qpc_identity_absence_of_noise() -> None:
    u = _channel_from_unitary(np.array([[0, 1], [1, 0]], dtype=np.complex128))
    v = _channel_from_unitary((1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex128))

    lhs = mix_channels(u, v, 0.0)
    rhs = v

    assert _norm_diff(lhs, rhs) < 1e-12


def test_qpc_identity_dominance_of_noise() -> None:
    u = _channel_from_unitary(np.array([[0, 1], [1, 0]], dtype=np.complex128))
    v = _channel_from_unitary((1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex128))

    lhs = mix_channels(u, v, 1.0)
    rhs = u

    assert _norm_diff(lhs, rhs) < 1e-12


def test_qpc_identity_rearrange_probabilistic_combinator() -> None:
    u = _channel_from_unitary(np.array([[0, 1], [1, 0]], dtype=np.complex128))
    v = _channel_from_unitary(np.array([[1, 0], [0, -1]], dtype=np.complex128))
    p = 0.23

    lhs = mix_channels(u, v, p)
    rhs = mix_channels(v, u, 1.0 - p)

    assert _norm_diff(lhs, rhs) < 1e-12


def test_qpc_identity_distributive_law() -> None:
    u = _channel_from_unitary(np.array([[0, 1], [1, 0]], dtype=np.complex128))
    v = _channel_from_unitary((1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex128))
    w = _channel_from_unitary(np.array([[1, 0], [0, -1]], dtype=np.complex128))
    p = 0.31
    q = 0.59

    lhs = mix_channels(u, mix_channels(v, w, q), p)
    rhs = mix_channels(mix_channels(u, v, p), mix_channels(u, w, p), q)

    assert _norm_diff(lhs, rhs) < 1e-12


def test_qpc_identity_choice_fusion_right_composition() -> None:
    u = _channel_from_unitary(np.array([[0, 1], [1, 0]], dtype=np.complex128))
    v = _channel_from_unitary((1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex128))
    w = _channel_from_unitary(np.array([[1, 0], [0, -1]], dtype=np.complex128))
    p = 0.45

    lhs = compose_superoperators(mix_channels(u, v, p), w)
    rhs = mix_channels(compose_superoperators(u, w), compose_superoperators(v, w), p)

    assert _norm_diff(lhs, rhs) < 1e-12


def test_qpc_identity_choice_fusion_left_composition() -> None:
    u = _channel_from_unitary(np.array([[0, 1], [1, 0]], dtype=np.complex128))
    v = _channel_from_unitary((1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex128))
    w = _channel_from_unitary(np.array([[1, 0], [0, -1]], dtype=np.complex128))
    p = 0.45

    lhs = compose_superoperators(w, mix_channels(u, v, p))
    rhs = mix_channels(compose_superoperators(w, u), compose_superoperators(w, v), p)

    assert _norm_diff(lhs, rhs) < 1e-12