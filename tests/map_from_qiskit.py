from __future__ import annotations

from dataclasses import dataclass

from adapters.qiskit import (
    load_noise_model_json,
    parse_qiskit_backend_gate_catalog,
    parse_qiskit_backend_noise_model,
    save_noise_model_json,
)


@dataclass
class _Param:
    name: str
    value: float
    unit: str | None = None


@dataclass
class _Gate:
    name: str
    qubits: list[int]
    parameters: list[_Param]


@dataclass
class _Nduv:
    name: str
    value: float
    unit: str | None = None


@dataclass
class _Properties:
    qubits: list[list[_Nduv]]
    gates: list[_Gate]


class _Backend:
    backend_version = "1.2.3"
    num_qubits = 2

    class _Configuration:
        basis_gates = ["id", "rz", "sx", "x", "ecr", "measure", "barrier"]

    def __init__(self):
        self._properties = _Properties(
            qubits=[
                [
                    _Nduv("T1", 100.0, "us"),
                    _Nduv("T2", 80.0, "us"),
                    _Nduv("readout_error", 0.02),
                    _Nduv("prob_meas0_prep1", 0.01),
                    _Nduv("prob_meas1_prep0", 0.015),
                ],
                [
                    _Nduv("T1", 120.0, "us"),
                    _Nduv("T2", 90.0, "us"),
                    _Nduv("readout_error", 0.03),
                ],
            ],
            gates=[
                _Gate(
                    name="sx",
                    qubits=[0],
                    parameters=[
                        _Param("gate_error", 0.001),
                        _Param("gate_length", 50.0, "ns"),
                    ],
                ),
                _Gate(
                    name="x",
                    qubits=[1],
                    parameters=[
                        _Param("gate_error", 0.002),
                        _Param("gate_length", 70.0, "ns"),
                    ],
                ),
            ],
        )

    def name(self) -> str:
        return "FakeBackendForTests"

    def properties(self):
        return self._properties

    def configuration(self):
        return self._Configuration()


class TestFromQiskitParser:
    def test_builds_requested_structure(self):
        model = parse_qiskit_backend_noise_model(_Backend(), date="2026-06-04T00:00:00Z")

        payload = model.to_dict()

        assert payload["device_name"] == "FakeBackendForTests"
        assert payload["version"] == "1.2.3"
        assert payload["date"] == "2026-06-04T00:00:00Z"

        assert "0" in payload["qubits"]
        assert "thermal_noise" in payload["qubits"]["0"]
        assert "spam" in payload["qubits"]["0"]
        assert "spam_readout" in payload["qubits"]["0"]
        assert "dephasing" in payload["qubits"]["0"]
        assert "depolarizing" in payload["qubits"]["0"]
        assert "thermal_dephasing" in payload["qubits"]["0"]
        assert "combo_depolarizing_spam" in payload["qubits"]["0"]
        assert "combo_depolarizing_thermal_dephasing" in payload["qubits"]["0"]
        assert "combo_all" in payload["qubits"]["0"]

        depolarizing_payload = payload["qubits"]["0"]["depolarizing"][0][1]
        assert depolarizing_payload["channel"] == "depolarizing"
        assert "pauli_probabilities" in depolarizing_payload

        combo_payload = payload["qubits"]["0"]["combo_all"][0][1]
        assert combo_payload["channel"] == "composed"
        assert set(combo_payload["components"]) >= {"depolarizing", "spam_bitflip", "amplitude_damping", "phase_flip"}

        assert 0.0 <= payload["qubit_error_rates"]["0"] <= 1.0
        assert 0.0 <= payload["qubit_error_rates"]["1"] <= 1.0

    def test_roundtrip_json(self, tmp_path):
        model = parse_qiskit_backend_noise_model(_Backend(), date="2026-06-04T00:00:00Z")

        output_file = tmp_path / "noise_model.json"
        save_noise_model_json(model, output_file)
        loaded = load_noise_model_json(output_file)

        assert loaded.device_name == model.device_name
        assert loaded.version == model.version
        assert loaded.date == model.date
        assert loaded.qubits.keys() == model.qubits.keys()
        assert loaded.qubit_error_rates.keys() == model.qubit_error_rates.keys()

    def test_extracts_base_gates_from_backend_configuration(self):
        catalog = parse_qiskit_backend_gate_catalog(_Backend())

        assert "sx" in catalog
        assert "x" in catalog
        assert "ecr" in catalog
        assert "measure" not in catalog
        assert "barrier" not in catalog
