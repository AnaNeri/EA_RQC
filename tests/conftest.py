"""Shared test fixtures for the map package."""

from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass
class FakeNoiseModel:
    qubit_error_rates: dict[int, float]


@dataclass
class FakeGraph:
    edges: dict[tuple[int, int], float]
    path_lengths: dict[tuple[int, int], int]

    def get_edge_data(self, source: int, target: int, default=None):
        return self.edges.get((source, target), self.edges.get((target, source), default))

    def shortest_path_length(self, source: int, target: int):
        return self.path_lengths.get((source, target), self.path_lengths.get((target, source)))


@pytest.fixture()
def noise_model() -> FakeNoiseModel:
    return FakeNoiseModel({0: 0.10, 1: 0.20, 2: 0.30, 3: 0.40, 4: 0.50})


@pytest.fixture()
def coupling_graph() -> FakeGraph:
    return FakeGraph(
        edges={(0, 1): 0.05, (1, 2): 0.07},
        path_lengths={(0, 2): 2, (2, 3): 3},
    )