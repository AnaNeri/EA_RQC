from __future__ import annotations

import json

import pytest

from evolution.circuit.targets import load_matrix_target_json


def test_load_matrix_target_json_unitary(tmp_path):
    payload = {
        "target_kind": "unitary",
        "num_qubits": 1,
        "matrix": [
            [0, 1],
            [1, 0],
        ],
    }
    path = tmp_path / "target.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    target = load_matrix_target_json(path)

    assert target.target_kind == "unitary"
    assert target.num_qubits == 1
    assert target.matrix.shape == (2, 2)


def test_load_matrix_target_json_rejects_bad_shape(tmp_path):
    payload = {
        "target_kind": "unitary",
        "num_qubits": 2,
        "matrix": [
            [1, 0],
            [0, 1],
        ],
    }
    path = tmp_path / "bad_target.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_matrix_target_json(path)
