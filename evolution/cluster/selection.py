"""Rank selection and survivor selection for cluster EA."""

from __future__ import annotations

from typing import Sequence

from .models import Individual, Population
from .utils import as_rng, normalise_cluster


def selection_rank(population: Population, fitness_scores: Sequence[float], type_ranking: str, seed: int | None = None) -> tuple[Individual, Individual]:
    if not population:
        raise ValueError("population cannot be empty")
    if len(population) != len(fitness_scores):
        raise ValueError("population and fitness_scores must have the same length")

    ranked = sorted(zip(population, fitness_scores), key=lambda item: item[1], reverse=True)
    ranked_population = [individual for individual, _ in ranked]
    population_size = len(ranked_population)

    ranking = type_ranking.lower().strip()
    if ranking == "deterministic":
        return ranked_population[0], ranked_population[1 if population_size > 1 else 0]

    if ranking == "exponential":
        weights = [2 ** (population_size - index - 1) for index in range(population_size)]
    else:
        weights = [population_size - index for index in range(population_size)]

    rng = as_rng(seed)
    parent1 = rng.choices(ranked_population, weights=weights, k=1)[0]
    parent2 = rng.choices(ranked_population, weights=weights, k=1)[0]

    if population_size > 1:
        attempts = 0
        while parent2 == parent1 and attempts < 5:
            parent2 = rng.choices(ranked_population, weights=weights, k=1)[0]
            attempts += 1

    return parent1, parent2


def selection_new_generation(
    population: Population,
    children: Population,
    fitness_scores: Sequence[float],
    children_fitness: Sequence[float],
    device_qubits: int,
    seed: int | None = None,
) -> Population:
    if len(population) != len(fitness_scores):
        raise ValueError("population and fitness_scores must have the same length")
    if len(children) != len(children_fitness):
        raise ValueError("children and children_fitness must have the same length")

    combined = list(zip(population, fitness_scores)) + list(zip(children, children_fitness))
    ranked = sorted(combined, key=lambda item: item[1], reverse=True)

    survivor_count = len(population)
    rng = as_rng(seed)
    next_generation: Population = []
    seen: set[tuple[int, ...]] = set()

    for individual, _ in ranked:
        signature = tuple(normalise_cluster(individual))
        if signature in seen:
            continue
        seen.add(signature)
        next_generation.append(normalise_cluster(individual))
        if len(next_generation) == survivor_count:
            break

    if next_generation:
        target_qubits = len(next_generation[0])
    elif population:
        target_qubits = len(population[0])
    elif children:
        target_qubits = len(children[0])
    else:
        return []

    while len(next_generation) < survivor_count:
        candidate = sorted(rng.sample(range(device_qubits), target_qubits))
        signature = tuple(candidate)
        if signature not in seen:
            seen.add(signature)
            next_generation.append(candidate)

    return next_generation
