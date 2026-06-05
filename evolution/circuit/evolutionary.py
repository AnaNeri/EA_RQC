"""Evolutionary search for robust circuit structures."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any, Sequence

from circuit.entities.circuit_list import CircuitList
from .fitness import CircuitFitnessBreakdown, fitness_circuit
from .operators import mutate_circuit, probabilistic_anchor_crossover, random_circuit


@dataclass(frozen=True)
class CircuitEvolutionResult:
	population: list[CircuitList]
	fitness_scores: list[CircuitFitnessBreakdown]
	best_circuit: CircuitList
	best_fitness: float
	diversity_history: list[float]
	generations_run: int


@dataclass(frozen=True)
class CircuitEvolutionConfig:
	device_qubits: int
	cluster_size: int
	population_size: int
	max_generations: int
	lambda_ratio: int = 2
	crossover_rate: float = 0.9
	mutation_rate: float = 0.2
	tournament_k: int = 3
	max_depth: int = 8
	max_gates: int = 16
	diversity_floor: float = 0.05


def _as_rng(seed: int | None) -> Random:
	return Random(seed)


def _population_diversity(population: list[CircuitList]) -> float:
	if not population:
		return 0.0
	signatures = {tuple(circuit.signatures()) for circuit in population}
	return len(signatures) / len(population)


def _tournament_pick(population: list[CircuitList], fitness_scores: list[float], k: int, rng: Random) -> CircuitList:
	size = len(population)
	if size == 0:
		raise ValueError("population cannot be empty")
	candidate_count = max(1, min(k, size))
	indices = rng.sample(range(size), candidate_count)
	winner_index = max(indices, key=lambda idx: fitness_scores[idx])
	return population[winner_index]


def _build_initial_population(
	config: CircuitEvolutionConfig,
	gate_catalog: Sequence[str],
	rng: Random,
) -> list[CircuitList]:
	population: list[CircuitList] = []
	for _ in range(config.population_size):
		cluster = tuple(sorted(rng.sample(range(config.device_qubits), config.cluster_size)))
		population.append(
			random_circuit(
				cluster=cluster,
				max_depth=config.max_depth,
				max_gates=config.max_gates,
				gate_catalog=gate_catalog,
				seed=rng.randrange(2**32),
			)
		)
	return population


def evolutionary_best_circuit(
	*,
	config: CircuitEvolutionConfig,
	gate_catalog: Sequence[str],
	target_circuit: CircuitList | None,
	noise_model: Any,
	seed: int | None = None,
) -> CircuitEvolutionResult:
	rng = _as_rng(seed)
	population = _build_initial_population(config, gate_catalog, rng)

	diversity_history: list[float] = []
	generation = 0

	while generation < config.max_generations:
		breakdowns = [
			fitness_circuit(circuit, target=target_circuit, noise_model=noise_model)
			for circuit in population
		]
		scores = [entry.total_fitness for entry in breakdowns]

		diversity = _population_diversity(population)
		diversity_history.append(diversity)
		if diversity <= config.diversity_floor:
			break

		children: list[CircuitList] = []
		offspring_rounds = max(1, config.population_size * max(1, config.lambda_ratio))
		for _ in range(offspring_rounds):
			parent1 = _tournament_pick(population, scores, config.tournament_k, rng)
			parent2 = _tournament_pick(population, scores, config.tournament_k, rng)

			child1, child2 = probabilistic_anchor_crossover(
				parent1,
				parent2,
				crossover_rate=config.crossover_rate,
				seed=rng.randrange(2**32),
			)
			child1 = mutate_circuit(
				child1,
				device_qubits=config.device_qubits,
				mutation_rate=config.mutation_rate,
				gate_catalog=gate_catalog,
				seed=rng.randrange(2**32),
			)
			child2 = mutate_circuit(
				child2,
				device_qubits=config.device_qubits,
				mutation_rate=config.mutation_rate,
				gate_catalog=gate_catalog,
				seed=rng.randrange(2**32),
			)
			children.append(child1)
			children.append(child2)

		child_breakdowns = [
			fitness_circuit(circuit, target=target_circuit, noise_model=noise_model)
			for circuit in children
		]

		ranked_children = sorted(
			zip(children, child_breakdowns),
			key=lambda item: item[1].total_fitness,
			reverse=True,
		)

		population = [entry[0] for entry in ranked_children[: config.population_size]]
		generation += 1

	final_breakdowns = [
		fitness_circuit(circuit, target=target_circuit, noise_model=noise_model)
		for circuit in population
	]
	best_index = max(range(len(population)), key=lambda idx: final_breakdowns[idx].total_fitness)

	ranked = sorted(zip(population, final_breakdowns), key=lambda item: item[1].total_fitness, reverse=True)
	return CircuitEvolutionResult(
		population=[item[0] for item in ranked],
		fitness_scores=[item[1] for item in ranked],
		best_circuit=population[best_index],
		best_fitness=final_breakdowns[best_index].total_fitness,
		diversity_history=diversity_history,
		generations_run=generation,
	)
