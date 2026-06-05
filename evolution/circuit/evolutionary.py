"""Evolutionary search for robust circuit structures."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any, Sequence

import numpy as np

from circuit.entities.circuit_list import CircuitList
from .fitness import CircuitFitnessBreakdown, fitness_circuit
from .operators import mutate_circuit, probabilistic_anchor_crossover, random_circuit
from .targets import load_matrix_target_json


@dataclass(frozen=True)
class CircuitEvolutionResult:
	population: list[CircuitList]
	fitness_scores: list[CircuitFitnessBreakdown]
	best_circuit: CircuitList
	best_fitness: float
	diversity_history: list[float]
	generations_run: int
	stop_reason: str = "unknown"
	cache_entries: int = 0


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
	stagnation_generations: int = 8
	stagnation_min_delta: float = 1e-6
	target_fitness_threshold: float = 0.9999
	target_matrix_json_path: str | None = None
	target_kind: str = "unitary"
	behavior_weight: float = 0.5
	robustness_weight: float = 0.35
	complexity_weight: float = 0.15
	fidelity_weight: float = 0.7
	frobenius_weight: float = 0.3
	# QPC/probabilistic combinator settings
	qpc_enabled: bool = False
	qpc_mode: str = "exact"  # "exact", "approx", or "mixed"


def _as_rng(seed: int | None) -> Random:
	return Random(seed)


def _population_diversity(population: list[CircuitList]) -> float:
	if not population:
		return 0.0
	signatures = {tuple(circuit.signatures()) for circuit in population}
	return len(signatures) / len(population)


def _circuit_cache_key(circuit: CircuitList, qpc_enabled: bool = False, qpc_mode: str = "exact") -> tuple[object, ...]:
	return (
		tuple(circuit.cluster),
		int(circuit.max_depth),
		circuit.max_gates,
		tuple(circuit.signatures(precision=6, include_depth=True)),
		qpc_enabled,
		qpc_mode,
	)


def _evaluate_population(
	population: list[CircuitList],
	*,
	target_circuit: CircuitList | None,
	target_matrix: np.ndarray | None,
	target_kind: str,
	noise_model: Any,
	behavior_weight: float,
	robustness_weight: float,
	complexity_weight: float,
	fidelity_weight: float,
	frobenius_weight: float,
	qpc_enabled: bool = False,
	qpc_mode: str = "exact",
	cache: dict[tuple[object, ...], CircuitFitnessBreakdown] | None = None,
) -> list[CircuitFitnessBreakdown]:
	if cache is None:
		cache = {}
	
	breakdowns: list[CircuitFitnessBreakdown] = []
	for circuit in population:
		key = _circuit_cache_key(circuit, qpc_enabled, qpc_mode)
		if key not in cache:
			cache[key] = fitness_circuit(
				circuit,
				target=target_circuit,
				target_matrix=target_matrix,
				target_kind=target_kind,
				noise_model=noise_model,
				behavior_weight=behavior_weight,
				robustness_weight=robustness_weight,
				complexity_weight=complexity_weight,
				fidelity_weight=fidelity_weight,
				frobenius_weight=frobenius_weight,
				qpc_enabled=qpc_enabled,
				qpc_mode=qpc_mode,
			)
		breakdowns.append(cache[key])
	return breakdowns


def _tournament_pick(population: list[CircuitList], fitness_scores: list[float], k: int, rng: Random) -> CircuitList:
	size = len(population)
	if size == 0:
		raise ValueError("population cannot be empty")
	candidate_count = max(1, min(k, size))
	indices = rng.sample(range(size), candidate_count)
	weights = [max(0.0, float(fitness_scores[idx])) for idx in indices]
	if not any(weights):
		weights = [1.0] * len(indices)
	winner_index = rng.choices(indices, weights=weights, k=1)[0]
	return population[winner_index]


def _select_mu_lambda_survivors(
	children: list[CircuitList],
	child_breakdowns: list[CircuitFitnessBreakdown],
	*,
	mu: int,
) -> list[CircuitList]:
	"""Select the next generation under explicit (mu, lambda) replacement.

	Parents are not carried forward; only the top mu offspring survive.
	"""
	if mu <= 0:
		raise ValueError("mu must be positive")
	if len(children) != len(child_breakdowns):
		raise ValueError("children and child_breakdowns must have the same length")

	ranked_children = sorted(
		zip(children, child_breakdowns),
		key=lambda item: item[1].total_fitness,
		reverse=True,
	)
	return [entry[0] for entry in ranked_children[:mu]]


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
	target_matrix: np.ndarray | None = None,
	noise_model: Any,
	seed: int | None = None,
) -> CircuitEvolutionResult:
	rng = _as_rng(seed)
	population = _build_initial_population(config, gate_catalog, rng)

	effective_target_kind = config.target_kind
	effective_target_matrix = target_matrix
	if config.target_matrix_json_path:
		parsed_target = load_matrix_target_json(config.target_matrix_json_path)
		effective_target_kind = parsed_target.target_kind
		effective_target_matrix = parsed_target.matrix

	diversity_history: list[float] = []
	fitness_cache: dict[tuple[object, ...], CircuitFitnessBreakdown] = {}
	generation = 0
	stop_reason = "max_generations"
	best_seen_fitness = float("-inf")
	stagnant_generations = 0

	while generation < config.max_generations:
		breakdowns = _evaluate_population(
			population,
			target_circuit=target_circuit,
			target_matrix=effective_target_matrix,
			target_kind=effective_target_kind,
			noise_model=noise_model,
			behavior_weight=config.behavior_weight,
			robustness_weight=config.robustness_weight,
			complexity_weight=config.complexity_weight,
			fidelity_weight=config.fidelity_weight,
			frobenius_weight=config.frobenius_weight,
			qpc_enabled=config.qpc_enabled,
			qpc_mode=config.qpc_mode,
			cache=fitness_cache,
		)
		scores = [entry.total_fitness for entry in breakdowns]

		diversity = _population_diversity(population)
		diversity_history.append(diversity)
		best_score = max(scores, default=0.0)
		if best_score >= config.target_fitness_threshold:
			stop_reason = "target_quality_threshold"
			break
		if best_score > best_seen_fitness + config.stagnation_min_delta:
			best_seen_fitness = best_score
			stagnant_generations = 0
		else:
			stagnant_generations += 1
			if config.stagnation_generations > 0 and stagnant_generations >= config.stagnation_generations:
				stop_reason = "stagnation"
				break
		if diversity <= config.diversity_floor:
			stop_reason = "low_diversity"
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

		child_breakdowns = _evaluate_population(
			children,
			target_circuit=target_circuit,
			target_matrix=effective_target_matrix,
			target_kind=effective_target_kind,
			noise_model=noise_model,
			behavior_weight=config.behavior_weight,
			robustness_weight=config.robustness_weight,
			complexity_weight=config.complexity_weight,
			fidelity_weight=config.fidelity_weight,
			frobenius_weight=config.frobenius_weight,
			qpc_enabled=config.qpc_enabled,
			qpc_mode=config.qpc_mode,
			cache=fitness_cache,
		)

		population = _select_mu_lambda_survivors(
			children,
			child_breakdowns,
			mu=config.population_size,
		)
		generation += 1

	final_breakdowns = _evaluate_population(
		population,
		target_circuit=target_circuit,
		target_matrix=effective_target_matrix,
		target_kind=effective_target_kind,
		noise_model=noise_model,
		behavior_weight=config.behavior_weight,
		robustness_weight=config.robustness_weight,
		complexity_weight=config.complexity_weight,
		fidelity_weight=config.fidelity_weight,
		frobenius_weight=config.frobenius_weight,
		qpc_enabled=config.qpc_enabled,
		qpc_mode=config.qpc_mode,
		cache=fitness_cache,
	)
	best_index = max(range(len(population)), key=lambda idx: final_breakdowns[idx].total_fitness)

	ranked = sorted(zip(population, final_breakdowns), key=lambda item: item[1].total_fitness, reverse=True)
	return CircuitEvolutionResult(
		population=[item[0] for item in ranked],
		fitness_scores=[item[1] for item in ranked],
		best_circuit=population[best_index],
		best_fitness=final_breakdowns[best_index].total_fitness,
		diversity_history=diversity_history,
		generations_run=generation,
		stop_reason=stop_reason,
		cache_entries=len(fitness_cache),
	)
