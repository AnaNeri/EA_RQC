"""Demo: X vs HZH—EA discovers noise-robust solutions under QPC.

Evolutionary algorithm search for robust X gate using QPC vs standard fitness.
Shows that QPC-aware fitness guides search toward shorter, noise-resilient circuits.
"""

from __future__ import annotations

from dataclasses import dataclass

from circuit.entities.circuit_list import CircuitGate, CircuitList
from evolution.circuit.evolutionary import (
    CircuitEvolutionConfig,
    evolutionary_best_circuit,
)
from evolution.circuit.matrix_eval import circuit_to_matrix


@dataclass(frozen=True)
class DemoNoiseModel:
    qubit_error_rates: dict[int, float]


def build_x_target() -> CircuitList:
    """Target: single X gate (ideal)."""
    return CircuitList(
        gates=[CircuitGate.from_values("x", (0,), depth=1)],
        cluster=(0,),
        max_depth=4,
        max_gates=5,
    )


def main() -> None:
    print("=" * 100)
    print("QPC Demo: Evolutionary Search for Noise-Robust X Circuit")
    print("=" * 100)
    print()
    print("Task: Evolve a circuit that implements X gate, robust under depolarizing noise.")
    print("QPC fitness considers faulty channels; standard fitness only considers ideal unitary.")
    print()

    # Setup
    target_circuit = build_x_target()
    target_matrix = circuit_to_matrix(target_circuit, target_kind="unitary")
    
    # Noise model: increase to 5% error per gate to make effect visible
    noise_model = DemoNoiseModel(qubit_error_rates={0: 0.05})
    
    gate_catalog = ["h", "x", "y", "z", "sx", "rx", "ry", "rz"]
    
    # Config for both runs
    base_config = CircuitEvolutionConfig(
        device_qubits=1,
        cluster_size=1,
        population_size=20,
        max_generations=15,
        lambda_ratio=2,
        crossover_rate=0.8,
        mutation_rate=0.3,
        tournament_k=3,
        max_depth=4,
        max_gates=5,
        diversity_floor=0.01,
        stagnation_generations=5,
        target_kind="unitary",
        behavior_weight=0.7,
        robustness_weight=0.2,
        complexity_weight=0.1,
        fidelity_weight=0.8,
        frobenius_weight=0.2,
    )
    
    # Build a seeded initial population that contains an H-Z-H sequence
    hzh = CircuitList(
        gates=[
            CircuitGate.from_values("h", (0,), depth=1),
            CircuitGate.from_values("z", (0,), depth=2),
            CircuitGate.from_values("h", (0,), depth=3),
        ],
        cluster=(0,),
        max_depth=4,
        max_gates=5,
    )

    # Show how HZH scores under ideal vs QPC evaluation
    from evolution.circuit.fitness import fitness_circuit

    hzh_ideal_fb = fitness_circuit(
        hzh,
        target=target_circuit,
        target_matrix=target_matrix,
        target_kind="unitary",
        noise_model=noise_model,
        qpc_enabled=False,
        fidelity_weight=base_config.fidelity_weight,
        frobenius_weight=base_config.frobenius_weight,
    )
    hzh_qpc_fb = fitness_circuit(
        hzh,
        target=target_circuit,
        target_matrix=target_matrix,
        target_kind="unitary",
        noise_model=noise_model,
        qpc_enabled=True,
        qpc_mode="exact",
        fidelity_weight=base_config.fidelity_weight,
        frobenius_weight=base_config.frobenius_weight,
    )

    print("HZH fitness (ideal):", f"{hzh_ideal_fb.total_fitness:.6f}")
    print("HZH fitness (QPC)  :", f"{hzh_qpc_fb.total_fitness:.6f}")
    print()

    print("--- Run 1: Standard Fitness (Ideal Unitary Only) ---")
    config_standard = CircuitEvolutionConfig(**{**base_config.__dict__, "qpc_enabled": False})
    result_standard = evolutionary_best_circuit(
        config=config_standard,
        gate_catalog=gate_catalog,
        target_circuit=target_circuit,
        target_matrix=target_matrix,
        noise_model=noise_model,
        seed=42,
        initial_population=[hzh],
    )
    
    print(f"Best circuit found: {len(result_standard.best_circuit.gates)} gates")
    for gate in result_standard.best_circuit.gates:
        print(f"  {gate.name.upper():>4} @ depth {gate.depth}")
    print(f"Best fitness (ideal): {result_standard.best_fitness:.6f}")
    print(f"Generations: {result_standard.generations_run}, Stop reason: {result_standard.stop_reason}")
    print()
    
    print("--- Run 2: QPC Fitness (Faulty Channels Considered) ---")
    config_qpc = CircuitEvolutionConfig(**{**base_config.__dict__, "qpc_enabled": True, "qpc_mode": "exact"})
    result_qpc = evolutionary_best_circuit(
        config=config_qpc,
        gate_catalog=gate_catalog,
        target_circuit=target_circuit,
        target_matrix=target_matrix,
        noise_model=noise_model,
        seed=42,
        initial_population=[hzh],
    )
    
    print(f"Best circuit found: {len(result_qpc.best_circuit.gates)} gates")
    for gate in result_qpc.best_circuit.gates:
        print(f"  {gate.name.upper():>4} @ depth {gate.depth}")
    print(f"Best fitness (QPC): {result_qpc.best_fitness:.6f}")
    print(f"Generations: {result_qpc.generations_run}, Stop reason: {result_qpc.stop_reason}")
    print()
    
    print("--- Comparison ---")
    print(f"Standard found {len(result_standard.best_circuit.gates)}-gate solution")
    print(f"QPC found {len(result_qpc.best_circuit.gates)}-gate solution")
    print()
    print("Key insight:")
    print("QPC-aware fitness biases search toward shorter, noise-resilient circuits")
    print("because it evaluates circuits against faulty channels (with depolarizing noise),")
    print("not just ideal unitaries. Single-gate solutions (like X) are naturally robust.")
    print()
    print(f"Cache entries (QPC): {result_qpc.cache_entries}")
    print(f"Diversity history (QPC): {[f'{d:.3f}' for d in result_qpc.diversity_history[:5]]}...")


if __name__ == "__main__":
    main()
