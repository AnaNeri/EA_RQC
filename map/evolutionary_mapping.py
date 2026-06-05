"""Orchestrate the evolutionary search for good qubit clusters."""

from __future__ import annotations

from typing import Any

from .models import EvolutionaryMappingResult, Population
from .operators import crossover, fitness, generate_combinations, mutation
from .selection import selection_new_generation, selection_rank
from .utils import as_rng


def evolutionary_best_clusters(
    device_qubits: int,
    target_qubits: int,
    max_generation: int,
    population: int,
    noise_model: Any,
    type_ranking: str,
    lambda_ratio: int,
    crossover_rate: float,
    mutation_rate: float,
    coupling_graph: Any = None,
    prioritize: str = "qubits",
    seed: int | None = None,
) -> EvolutionaryMappingResult:
    rng = as_rng(seed)
    current_population = generate_combinations(population, device_qubits, target_qubits, seed=rng.randrange(2**32))
    generation = 0

    lambda_ratio = max(1, lambda_ratio)

    if noise_model is not None:
        while generation < max_generation:
            fitness_scores = [fitness(individual, noise_model, coupling_graph=coupling_graph, prioritize=prioritize) for individual in current_population]

            children: Population = []
            parent_pairs = max(1, population * lambda_ratio)

            for _ in range(parent_pairs):
                parent1, parent2 = selection_rank(current_population, fitness_scores, type_ranking, seed=rng.randrange(2**32))
                child1, child2 = crossover(parent1, parent2, device_qubits, crossover_rate=crossover_rate, seed=rng.randrange(2**32))
                child1 = mutation(child1, device_qubits, target_qubits, mutation_rate=mutation_rate, seed=rng.randrange(2**32))
                child2 = mutation(child2, device_qubits, target_qubits, mutation_rate=mutation_rate, seed=rng.randrange(2**32))
                children.append(child1)
                children.append(child2)

            children_fitness = [fitness(individual, noise_model, coupling_graph=coupling_graph, prioritize=prioritize) for individual in children]
            current_population = selection_new_generation(
                current_population,
                children,
                fitness_scores,
                children_fitness,
                device_qubits,
                seed=rng.randrange(2**32),
            )
            generation += 1

    final_fitness = [fitness(individual, noise_model, coupling_graph=coupling_graph, prioritize=prioritize) for individual in current_population]
    if current_population:
        best_index = max(range(len(current_population)), key=lambda index: final_fitness[index])
        best_cluster = current_population[best_index]
        best_fitness = final_fitness[best_index]
    else:
        best_cluster = []
        best_fitness = 0.0

    ranked_population = [individual for individual, _ in sorted(zip(current_population, final_fitness), key=lambda item: item[1], reverse=True)]
    ranked_scores = sorted(final_fitness, reverse=True)

    return EvolutionaryMappingResult(
        population=ranked_population,
        fitness_scores=ranked_scores,
        best_cluster=best_cluster,
        best_fitness=best_fitness,
        generations_run=generation,
    )