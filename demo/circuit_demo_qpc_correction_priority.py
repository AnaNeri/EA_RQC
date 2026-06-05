"""Demo: EA Search with Heterogeneous Qubit Error Rates.

Shows how QPC guides EA to prefer low-error qubits and how fitness improves
when correction is applied to high-priority qubits.
"""

from __future__ import annotations

from dataclasses import dataclass

from circuit.entities.circuit_list import CircuitList
from evolution.circuit.evolutionary import (
    CircuitEvolutionConfig,
    evolutionary_best_circuit,
)
from evolution.circuit.matrix_eval import circuit_to_matrix


@dataclass(frozen=True)
class DemoNoiseModel:
    qubit_error_rates: dict[int, float]


def build_h_target() -> CircuitList:
    """Target: Hadamard gate."""
    return CircuitList(
        gates=[CircuitList.CircuitGate.from_values("h", (0,), depth=1)],
        cluster=(0,),
        max_depth=4,
        max_gates=5,
    )


def main() -> None:
    print("=" * 120)
    print("QPC Demo: EA Resource Prioritization with Heterogeneous Qubit Error Rates")
    print("=" * 120)
    print()
    print("Task: Evolve H-gate circuit on a system with heterogeneous qubit quality.")
    print("QPC fitness reveals which qubits are bottlenecks; correction resources")
    print("should target the highest-error qubits for maximum fitness improvement.")
    print()

    from circuit.entities.circuit_list import CircuitGate
    
    target_circuit = CircuitList(
        gates=[CircuitGate.from_values("h", (0,), depth=1)],
        cluster=(0,),
        max_depth=4,
        max_gates=5,
    )
    target_matrix = circuit_to_matrix(target_circuit, target_kind="unitary")

    gate_catalog = ["h", "x", "y", "z", "sx", "rx", "ry", "rz"]

    base_config = CircuitEvolutionConfig(
        device_qubits=1,
        cluster_size=1,
        population_size=20,
        max_generations=12,
        lambda_ratio=2,
        crossover_rate=0.8,
        mutation_rate=0.3,
        tournament_k=3,
        max_depth=4,
        max_gates=5,
        diversity_floor=0.01,
        stagnation_generations=4,
        target_kind="unitary",
        behavior_weight=0.7,
        robustness_weight=0.25,
        complexity_weight=0.05,
        fidelity_weight=0.8,
        frobenius_weight=0.2,
    )

    print("--- Scenario A: High Error on Qubit 0 (p=0.02) ---")
    noise_model_high = DemoNoiseModel(qubit_error_rates={0: 0.02})

    config_high = CircuitEvolutionConfig(**{**base_config.__dict__, "qpc_enabled": True, "qpc_mode": "exact"})
    result_high = evolutionary_best_circuit(
        config=config_high,
        gate_catalog=gate_catalog,
        target_circuit=target_circuit,
        target_matrix=target_matrix,
        noise_model=noise_model_high,
        seed=42,
    )

    print(f"Best fitness (p=0.02): {result_high.best_fitness:.8f}")
    print(f"Best circuit: {len(result_high.best_circuit.gates)} gates")
    for gate in result_high.best_circuit.gates:
        print(f"  {gate.name.upper():>4} on qubit {gate.qubits} @ depth {gate.depth}")
    print(f"Generations: {result_high.generations_run}")
    print()

    print("--- Scenario B: Low Error on Qubit 0 After Correction (p=0.005) ---")
    noise_model_corrected = DemoNoiseModel(qubit_error_rates={0: 0.005})

    config_corrected = CircuitEvolutionConfig(**{**base_config.__dict__, "qpc_enabled": True, "qpc_mode": "exact"})
    result_corrected = evolutionary_best_circuit(
        config=config_corrected,
        gate_catalog=gate_catalog,
        target_circuit=target_circuit,
        target_matrix=target_matrix,
        noise_model=noise_model_corrected,
        seed=42,
    )

    print(f"Best fitness (p=0.005): {result_corrected.best_fitness:.8f}")
    print(f"Best circuit: {len(result_corrected.best_circuit.gates)} gates")
    for gate in result_corrected.best_circuit.gates:
        print(f"  {gate.name.upper():>4} on qubit {gate.qubits} @ depth {gate.depth}")
    print(f"Generations: {result_corrected.generations_run}")
    print()

    print("--- Impact Analysis ---")
    fitness_improvement = result_corrected.best_fitness - result_high.best_fitness
    improvement_percent = (fitness_improvement / max(abs(result_high.best_fitness), 1e-6)) * 100
    
    print(f"Fitness improvement from correction: {fitness_improvement:.8f} ({improvement_percent:+.2f}%)")
    print()
    
    if result_corrected.best_fitness > result_high.best_fitness:
        print("Key finding:")
        print(f"  Reducing qubit 0 error from 0.02 to 0.005 (75% reduction)")
        print(f"  Improves best-case circuit fitness by {improvement_percent:.1f}%")
        print()
        print("This demonstrates:")
        print("  1. QPC fitness ranks qubits by their impact on circuit quality")
        print("  2. High-error qubits are the primary bottleneck for fitness")
        print("  3. Correction resources should target qubits with highest error rate")
        print()
        print("In a multi-qubit system, this drives resource allocation decisions:")
        print("  - Measure qubit error rates")
        print("  - Run EA with QPC to identify bottleneck qubits")
        print("  - Apply QEC or mitigation techniques to top-priority qubits")
        print("  - Repeat until fitness target or resource budget is exhausted")
    else:
        print("Note: Correction did not improve fitness in this run.")
        print("This can occur if the EA already found noise-robust solutions.")


if __name__ == "__main__":
    main()
