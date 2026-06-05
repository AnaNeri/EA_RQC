from __future__ import annotations

import random

import pytest

from map.utils import cluster_noise_penalty, normalise_cluster, repair_cluster, topology_penalty


class TestNormaliseCluster:
    def test_removes_duplicates_and_sorts(self):
        assert normalise_cluster([3, 1, 3, 2, 1]) == [1, 2, 3]


class TestRepairCluster:
    def test_repairs_invalid_values_and_size(self):
        rng = random.Random(7)

        repaired = repair_cluster([-1, 0, 0, 9], device_qubits=5, target_qubits=3, rng=rng)

        assert len(repaired) == 3
        assert repaired == sorted(repaired)
        assert all(0 <= qubit < 5 for qubit in repaired)

    def test_rejects_target_larger_than_device(self):
        rng = random.Random(1)

        with pytest.raises(ValueError):
            repair_cluster([0], device_qubits=2, target_qubits=3, rng=rng)


class TestNoisePenalty:
    def test_averages_qubit_errors_from_mapping(self, noise_model):
        penalty = cluster_noise_penalty([0, 2, 4], noise_model)

        assert penalty == pytest.approx((0.10 + 0.30 + 0.50) / 3)


class TestTopologyPenalty:
    def test_uses_direct_edge_when_available(self, coupling_graph):
        penalty = topology_penalty([0, 1], coupling_graph)

        assert penalty == pytest.approx(0.05)

    def test_zero_error_edge_is_not_treated_as_missing(self):
        class ZeroEdgeGraph:
            def get_edge_data(self, source, target, default=None):
                if (source, target) == (0, 1) or (source, target) == (1, 0):
                    return 0.0
                return default

        penalty = topology_penalty([0, 1], ZeroEdgeGraph())

        assert penalty == pytest.approx(0.0)

    def test_uses_path_penalty_when_no_direct_edge(self, coupling_graph):
        penalty = topology_penalty([0, 2], coupling_graph)

        assert penalty == pytest.approx(1.0)