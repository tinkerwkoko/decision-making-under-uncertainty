
from experiments.compare_bandit_algorithms import (
    run_epsilon_greedy,
    run_thompson_sampling,
)


def run_comparison(
    probabilities,
    n_rounds=1000,
    n_simulations=100,
    epsilon=0.1,
):
    """
    Compare Epsilon-Greedy and Thompson Sampling
    across multiple independent simulations.

    Returns average cumulative regret for each algorithm.
    """

    epsilon_greedy_regrets = []
    thompson_sampling_regrets = []

    for simulation in range(n_simulations):

        # Use a different seed for each simulation
        seed = simulation

        # Run Epsilon-Greedy
        epsilon_result = run_epsilon_greedy(
            probabilities,
            n_rounds,
            epsilon,
            seed=seed,
        )

        # Run Thompson Sampling
        thompson_result = run_thompson_sampling(
            probabilities,
            n_rounds,
            seed=seed,
        )

        epsilon_greedy_regrets.append(
            epsilon_result["cumulative_regret"]
        )

        thompson_sampling_regrets.append(
            thompson_result["cumulative_regret"]
        )

    # Calculate average cumulative regret at every round
    average_epsilon_regret = [
        sum(regrets[round_index] for regrets in epsilon_greedy_regrets)
        / n_simulations
        for round_index in range(n_rounds)
    ]

    average_thompson_regret = [
        sum(regrets[round_index] for regrets in thompson_sampling_regrets)
        / n_simulations
        for round_index in range(n_rounds)
    ]

    return {
        "epsilon_greedy": average_epsilon_regret,
        "thompson_sampling": average_thompson_regret,
    }
