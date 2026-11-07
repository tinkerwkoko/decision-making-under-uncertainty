
import random

from algorithms.bandit_algorithms import (
    initialize_action_estimates,
    update_action_estimates,
    epsilon_greedy,
    initialize_beta_parameters,
    update_beta_parameters,
    thompson_sampling,
)

from environments.bandit import BernoulliBandit

from experiments.bandit_metrics import calculate_cumulative_regret


def run_epsilon_greedy(probabilities, n_rounds, epsilon, seed=None):
    """
    Run one Epsilon-Greedy simulation.

    Returns selected arms, observed rewards, and cumulative regret.
    """

    bandit = BernoulliBandit(probabilities, seed=seed)

    action_counts, estimated_values = initialize_action_estimates(
        bandit.n_arms
    )

    rng = random.Random(None if seed is None else seed + 1)

    selected_arms = []
    rewards = []

    for _ in range(n_rounds):
        arm = epsilon_greedy(estimated_values, epsilon, rng)

        reward = bandit.pull(arm)

        update_action_estimates(
            action_counts,
            estimated_values,
            arm,
            reward,
        )

        selected_arms.append(arm)
        rewards.append(reward)

    cumulative_regret = calculate_cumulative_regret(
        probabilities,
        selected_arms,
    )

    return {
        "selected_arms": selected_arms,
        "rewards": rewards,
        "cumulative_regret": cumulative_regret,
    }


def run_thompson_sampling(probabilities, n_rounds, seed=None):
    """
    Run one Thompson Sampling simulation.

    Returns selected arms, observed rewards, and cumulative regret.
    """

    bandit = BernoulliBandit(probabilities, seed=seed)

    alpha, beta = initialize_beta_parameters(bandit.n_arms)

    rng = random.Random(None if seed is None else seed + 1)

    selected_arms = []
    rewards = []

    for _ in range(n_rounds):
        arm = thompson_sampling(alpha, beta, rng)

        reward = bandit.pull(arm)

        update_beta_parameters(alpha, beta, arm, reward)

        selected_arms.append(arm)
        rewards.append(reward)

    cumulative_regret = calculate_cumulative_regret(
        probabilities,
        selected_arms,
    )

    return {
        "selected_arms": selected_arms,
        "rewards": rewards,
        "cumulative_regret": cumulative_regret,
    }
