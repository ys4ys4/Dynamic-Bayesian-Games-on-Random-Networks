"""Small executable check for a correctly installed package."""

from simulations.game_engine import SequentialGame
from simulations.run_experiments import run_sim


def main():
    result = run_sim(
        game_type=SequentialGame,
        graph_type="complete",
        agents=12,
        runs=3,
        seed=42,
        signal_type="bounded",
        q=0.8,
    )

    assert len(result.running_accuracies) == 3
    assert len(result.convergence_metrics) == 3
    assert result.params["seed"] == 42
    print("Smoke test passed: 3 complete-graph runs with 12 agents.")


if __name__ == "__main__":
    main()
