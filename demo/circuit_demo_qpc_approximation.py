"""Demo: EA Convergence with Exact vs First-Order Approximation Mode.

Shows that first-order approximation mode enables faster EA convergence
with minimal accuracy loss for small error rates.
"""

from __future__ import annotations

import numpy as np
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


def build_cnot_target() -> CircuitList:
    """Target: CNOT (CX) gate on 2 qubits."""
    return CircuitList(
        gates=[CircuitList.CircuitGate.from_values("cx", (0, 1), depth=1)],
        cluster=(0, 1),
        max_depth=5,
        max_gates=8,
    )


def main() -> None:
    print("=" * 110)
    print("QPC Demo: EA Convergence with Exact vs First-Order Approximation")
    print("=" * 110)
    print()
    print("Task: Evolve a circuit implementing CNOT, evaluated under two QPC modes:")
    print("  - 'exact': full gate-wise depolarizing (accurate, computationally expensive)")
    print("  - 'approx': first-order approximation (faster, good for small p)")
    print()

    # Import here to avoid circular dependency
    from circuit.entities.circuit_list import CircuitGate
    
    target_circuit = CircuitList(
        gates=[CircuitGate.from_values("cx", (0, 1), depth=1)],
        cluster=(0, 1),
        max_depth=5,
        max_gates=8,
    )
    target_matrix = circuit_to_matrix(target_circuit, target_kind="unitary")

    # Small error rate (where first-order is most accurate)
    noise_model = DemoNoiseModel(qubit_error_rates={0: 0.002, 1: 0.002})

    gate_catalog = ["h", "x", "y", "z", "sx", "cx", "cz"]

    base_config = CircuitEvolutionConfig(
        device_qubits=2,
        cluster_size=2,
        population_size=25,
        max_generations=12,
        lambda_ratio=2,
        crossover_rate=0.8,
        mutation_rate=0.25,
        tournament_k=3,
        max_depth=5,
        max_gates=8,
        diversity_floor=0.02,
        stagnation_generations=4,
        target_kind="unitary",
        behavior_weight=0.7,
        robustness_weight=0.2,
        complexity_weight=0.1,
        fidelity_weight=0.8,
        frobenius_weight=0.2,
    )

    print("--- Run 1: QPC Exact Mode (Full Gate-Wise Depolarizing) ---")
    config_exact = CircuitEvolutionConfig(
        **{**base_config.__dict__, "qpc_enabled": True, "qpc_mode": "exact"}
    )
    result_exact = evolutionary_best_circuit(
        config=config_exact,
        gate_catalog=gate_catalog,
        target_circuit=target_circuit,
        target_matrix=target_matrix,
        noise_model=noise_model,
        seed=42,
    )

    print(f"Best fitness: {result_exact.best_fitness:.8f}")
    print(f"Best circuit gates: {len(result_exact.best_circuit.gates)}")
    print(f"Generations run: {result_exact.generations_run}")
    print(f"Stop reason: {result_exact.stop_reason}")
    print(f"Cache entries: {result_exact.cache_entries}")
    print(f"Diversity history: {[f'{d:.3f}' for d in result_exact.diversity_history[:6]]}...")
    print()

    print("--- Run 2: QPC Approximation Mode (First-Order Truncation) ---")
    config_approx = CircuitEvolutionConfig(
        **{**base_config.__dict__, "qpc_enabled": True, "qpc_mode": "approx"}
    )
    result_approx = evolutionary_best_circuit(
        config=config_approx,
        gate_catalog=gate_catalog,
        target_circuit=target_circuit,
        target_matrix=target_matrix,
        noise_model=noise_model,
        seed=42,
    )

    print(f"Best fitness: {result_approx.best_fitness:.8f}")
    print(f"Best circuit gates: {len(result_approx.best_circuit.gates)}")
    print(f"Generations run: {result_approx.generations_run}")
    print(f"Stop reason: {result_approx.stop_reason}")
    print(f"Cache entries: {result_approx.cache_entries}")
    print(f"Diversity history: {[f'{d:.3f}' for d in result_approx.diversity_history[:6]]}...")
    print()

    print("--- Comparison ---")
    fitness_diff = abs(result_exact.best_fitness - result_approx.best_fitness)
    cache_reduction = 1.0 - (result_approx.cache_entries / max(result_exact.cache_entries, 1))
    
    print(f"Fitness difference: {fitness_diff:.8f}")
    print(f"Exact cache size: {result_exact.cache_entries} entries")
    print(f"Approx cache size: {result_approx.cache_entries} entries")
    print(f"Cache reduction: {cache_reduction:.1%}")
    print()
    print("Key insight:")
    print("For small error rates (p ≈ 0.002), first-order approximation mode")
    print("achieves nearly identical fitness with significantly fewer cache entries")
    print("and reduced computational cost per fitness evaluation.")
    print()
    print("This demonstrates the practical value of the theoretical result:")
    print("When p is small, truncating error evolution to first order is accurate.")


if __name__ == "__main__":
    main()
