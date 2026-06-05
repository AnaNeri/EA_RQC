"""Deprecated compatibility namespace.

Prefer importing from `evolution.cluster` and `evolution.circuit`.
"""

from .circuit_evolutionary import CircuitEvolutionConfig, CircuitEvolutionResult, evolutionary_best_circuit
from .circuit_fitness import CircuitFitnessBreakdown, fitness_circuit
from .circuit_operators import find_anchor_blocks, mutate_circuit, probabilistic_anchor_crossover, random_circuit
from .evolutionary_mapping import evolutionary_best_clusters
from .models import EvolutionaryMappingResult
from .operators import crossover, fitness, generate_combinations, mutation
from .selection import selection_new_generation, selection_rank

__all__ = [
    "CircuitEvolutionConfig",
    "CircuitEvolutionResult",
    "CircuitFitnessBreakdown",
    "EvolutionaryMappingResult",
    "crossover",
    "evolutionary_best_clusters",
    "evolutionary_best_circuit",
    "find_anchor_blocks",
    "fitness",
    "fitness_circuit",
    "generate_combinations",
    "mutate_circuit",
    "mutation",
    "probabilistic_anchor_crossover",
    "random_circuit",
    "selection_new_generation",
    "selection_rank",
]