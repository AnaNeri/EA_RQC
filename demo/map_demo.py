"""Demo for the evolutionary qubit-cluster search."""

from __future__ import annotations

from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Sequence

from adapters.qiskit import load_noise_model_json, parse_qiskit_backend_noise_model, save_noise_model_json
from evolution.cluster.evolutionary import evolutionary_best_clusters


@dataclass(frozen=True)
class DemoNoiseModel:
    qubit_error_rates: dict[int, float]


@dataclass(frozen=True)
class DemoCouplingGraph:
    edges: dict[tuple[int, int], float]

    def get_edge_data(self, source: int, target: int, default=None):
        return self.edges.get((source, target), self.edges.get((target, source), default))


@dataclass(frozen=True)
class DemoNduv:
    name: str
    value: float
    unit: str | None = None


@dataclass(frozen=True)
class DemoGateParameter:
    name: str
    value: float
    unit: str | None = None


@dataclass(frozen=True)
class DemoGate:
    qubits: list[int]
    parameters: list[DemoGateParameter]


@dataclass(frozen=True)
class DemoProperties:
    qubits: list[list[DemoNduv]]
    gates: list[DemoGate]


class DemoBackend:
    """Backend-like object shaped similarly to Qiskit backend properties."""

    backend_version = "demo-v1"
    num_qubits = 6

    def __init__(self):
        self._properties = DemoProperties(
            qubits=[
                [DemoNduv("T1", 180.0, "us"), DemoNduv("T2", 150.0, "us"), DemoNduv("readout_error", 0.007)],
                [DemoNduv("T1", 170.0, "us"), DemoNduv("T2", 145.0, "us"), DemoNduv("readout_error", 0.008)],
                [DemoNduv("T1", 160.0, "us"), DemoNduv("T2", 140.0, "us"), DemoNduv("readout_error", 0.009)],
                [DemoNduv("T1", 95.0, "us"), DemoNduv("T2", 70.0, "us"), DemoNduv("readout_error", 0.028)],
                [DemoNduv("T1", 80.0, "us"), DemoNduv("T2", 60.0, "us"), DemoNduv("readout_error", 0.034)],
                [DemoNduv("T1", 110.0, "us"), DemoNduv("T2", 75.0, "us"), DemoNduv("readout_error", 0.022)],
            ],
            gates=self._build_gates(),
        )

    def name(self) -> str:
        return "DemoBackendLikeQiskit"

    def properties(self):
        return self._properties

    @staticmethod
    def _build_gates() -> list[DemoGate]:
        gate_errors = [0.0008, 0.0009, 0.0010, 0.0034, 0.0041, 0.0026]
        gate_lengths_ns = [42.0, 43.0, 45.0, 78.0, 82.0, 65.0]
        gates: list[DemoGate] = []
        for qubit, (gate_error, gate_length_ns) in enumerate(zip(gate_errors, gate_lengths_ns)):
            gates.append(
                DemoGate(
                    qubits=[qubit],
                    parameters=[
                        DemoGateParameter("gate_error", gate_error),
                        DemoGateParameter("gate_length", gate_length_ns, "ns"),
                    ],
                )
            )
        return gates


def build_demo_noise_model() -> DemoNoiseModel:
    parsed = parse_qiskit_backend_noise_model(DemoBackend())

    # Demonstrate persistence with the same parser that handles real Qiskit data.
    with TemporaryDirectory() as tmpdir:
        output = save_noise_model_json(parsed, f"{tmpdir}/demo_noise_model.json")
        loaded = load_noise_model_json(output)

    return DemoNoiseModel(qubit_error_rates=loaded.qubit_error_rates)


def print_qubit_errors(error_rates: dict[int, float], qubits: Sequence[int]) -> None:
    print("Qubit error rates used by mapping:")
    for qubit in qubits:
        print(f"  q{qubit}: {error_rates.get(qubit, 0.0):.6f}")


def main() -> None:
    noise_model = build_demo_noise_model()
    print_qubit_errors(noise_model.qubit_error_rates, range(6))

    coupling_graph = DemoCouplingGraph(
        edges={
            (0, 1): 0.002,
            (1, 2): 0.003,
            (2, 3): 0.040,
            (3, 4): 0.025,
            (4, 5): 0.020,
        }
    )

    result = evolutionary_best_clusters(
        device_qubits=6,
        target_qubits=3,
        max_generation=5,
        population=10,
        noise_model=noise_model,
        type_ranking="linear",
        lambda_ratio=2,
        crossover_rate=0.8,
        mutation_rate=0.2,
        coupling_graph=coupling_graph,
        prioritize="both",
        seed=42,
    )

    print("Best cluster:", result.best_cluster)
    print("Best fitness:", round(result.best_fitness, 6))
    print("Generations run:", result.generations_run)
    print("Final population:")
    for index, cluster in enumerate(result.population, start=1):
        score = result.fitness_scores[index - 1] if index - 1 < len(result.fitness_scores) else None
        print(f"  {index}. {cluster} -> {score}")





if __name__ == "__main__":
    main()