"""Genetic operators and population generation for cluster EA."""

from __future__ import annotations

import math
from typing import Any, Sequence

from .models import Individual, Population
from .utils import as_rng, cluster_noise_penalty, normalise_cluster, repair_cluster, topology_penalty


def fitness(
    individual: Sequence[int],
    noise_model: Any,
    coupling_graph: Any = None,
    prioritize: str = "qubits",
) -> float:
    cluster = normalise_cluster(individual)
    if not cluster:
        return 0.0

    qubit_penalty = cluster_noise_penalty(cluster, noise_model)
    coupling_penalty = topology_penalty(cluster, coupling_graph)

    mode = prioritize.lower().strip()
    if mode == "coupling":
        penalty = coupling_penalty if coupling_penalty > 0 else qubit_penalty
    elif mode == "both":
        penalty = qubit_penalty + coupling_penalty
    else:
        penalty = qubit_penalty

    return 1.0 / (1.0 + max(penalty, 0.0))


def generate_combinations(population_size: int, device_qubits: int, target_qubits: int, seed: int | None = None) -> Population:
    if population_size <= 0:
        raise ValueError("population_size must be greater than zero")

    rng = as_rng(seed)
    population: Population = []
    seen: set[tuple[int, ...]] = set()
    max_unique = math.comb(device_qubits, target_qubits)
    target_population = min(population_size, max_unique)

    while len(population) < target_population:
        cluster = sorted(rng.sample(range(device_qubits), target_qubits))
        signature = tuple(cluster)
        if signature in seen:
            continue
        seen.add(signature)
        population.append(cluster)

    return population


def crossover(parent1: Sequence[int], parent2: Sequence[int], device_qubits: int, crossover_rate: float = 0.7, seed: int | None = None) -> tuple[Individual, Individual]:
    rng = as_rng(seed)
    target_qubits = len(parent1)
    if target_qubits == 0:
        return [], []
    if rng.random() > crossover_rate:
        return normalise_cluster(parent1), normalise_cluster(parent2)

    parent1 = repair_cluster(parent1, device_qubits, target_qubits, rng)
    parent2 = repair_cluster(parent2, device_qubits, target_qubits, rng)

    shared = [qubit for qubit in parent1 if qubit in parent2]
    parent1_only = [qubit for qubit in parent1 if qubit not in shared]
    parent2_only = [qubit for qubit in parent2 if qubit not in shared]

    def build_child(primary: list[int], secondary: list[int]) -> Individual:
        child = list(shared)

        for qubit in primary:
            if len(child) >= target_qubits:
                break
            if qubit not in child:
                child.append(qubit)

        for qubit in secondary:
            if len(child) >= target_qubits:
                break
            if qubit not in child:
                child.append(qubit)

        return repair_cluster(child, device_qubits, target_qubits, rng)

    child1 = build_child(parent1_only, parent2_only)
    child2 = build_child(parent2_only, parent1_only)

    return child1, child2


def mutation(child: Sequence[int], device_qubits: int, target_qubits: int, mutation_rate: float = 0.05, seed: int | None = None) -> Individual:
    rng = as_rng(seed)
    mutated = list(child)

    if mutated and rng.random() <= mutation_rate:
        index = rng.randrange(len(mutated))
        choices = [qubit for qubit in range(device_qubits) if qubit not in mutated or qubit == mutated[index]]
        if choices:
            mutated[index] = rng.choice(choices)

    return repair_cluster(mutated, device_qubits, target_qubits, rng)
