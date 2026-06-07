"""Data structures used by the cluster evolutionary search."""

from __future__ import annotations

from dataclasses import dataclass, field


Individual = list[int]
Population = list[Individual]


@dataclass(frozen=True)
class EvolutionaryMappingResult:
    """Result container for the search."""

    population: Population
    fitness_scores: list[float]
    best_cluster: Individual
    best_fitness: float
    generations_run: int
    top_clusters: Population = field(default_factory=list)
    top_fitness_scores: list[float] = field(default_factory=list)
