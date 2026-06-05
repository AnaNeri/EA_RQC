from __future__ import annotations

from evolution.cluster.evolutionary import evolutionary_best_clusters


class TestEvolutionaryBestClusters:
    def test_runs_complete_search_and_returns_result(self, noise_model):
        result = evolutionary_best_clusters(
            device_qubits=6,
            target_qubits=3,
            max_generation=2,
            population=4,
            noise_model=noise_model,
            type_ranking="linear",
            lambda_ratio=2,
            crossover_rate=1.0,
            mutation_rate=0.5,
            prioritize="qubits",
            seed=21,
        )

        assert result.generations_run == 2
        assert len(result.population) == 4
        assert len(result.fitness_scores) == 4
        assert len(result.best_cluster) == 3
        assert all(0 <= qubit < 6 for qubit in result.best_cluster)

    def test_skips_evolution_when_noise_model_is_none(self):
        result = evolutionary_best_clusters(
            device_qubits=6,
            target_qubits=3,
            max_generation=5,
            population=4,
            noise_model=None,
            type_ranking="linear",
            lambda_ratio=2,
            crossover_rate=1.0,
            mutation_rate=0.5,
            seed=21,
        )

        assert result.generations_run == 0
        assert len(result.population) == 4