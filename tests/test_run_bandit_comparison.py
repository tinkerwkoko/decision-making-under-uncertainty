
from experiments.run_bandit_comparison import run_comparison


def test_run_comparison_returns_both_algorithms():
    results = run_comparison(
        [0.1, 0.3, 0.5, 0.8],
        n_rounds=50,
        n_simulations=5,
    )

    assert "epsilon_greedy" in results
    assert "thompson_sampling" in results


def test_run_comparison_returns_correct_number_of_rounds():
    n_rounds = 50

    results = run_comparison(
        [0.1, 0.3, 0.5, 0.8],
        n_rounds=n_rounds,
        n_simulations=5,
    )

    assert len(results["epsilon_greedy"]) == n_rounds
    assert len(results["thompson_sampling"]) == n_rounds


def test_average_regret_is_nonnegative():
    results = run_comparison(
        [0.1, 0.3, 0.5, 0.8],
        n_rounds=50,
        n_simulations=5,
    )

    for algorithm_regrets in results.values():
        assert all(regret >= 0 for regret in algorithm_regrets)


def test_average_cumulative_regret_never_decreases():
    results = run_comparison(
        [0.1, 0.3, 0.5, 0.8],
        n_rounds=50,
        n_simulations=5,
    )

    for algorithm_regrets in results.values():
        for previous, current in zip(
            algorithm_regrets,
            algorithm_regrets[1:],
        ):
            assert current >= previous


def test_run_comparison_is_reproducible():
    first_run = run_comparison(
        [0.1, 0.3, 0.5, 0.8],
        n_rounds=50,
        n_simulations=5,
    )

    second_run = run_comparison(
        [0.1, 0.3, 0.5, 0.8],
        n_rounds=50,
        n_simulations=5,
    )

    assert first_run == second_run
