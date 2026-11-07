
import random


# ==========================================
# THOMPSON SAMPLING
# ==========================================

def initialize_beta_parameters(n_arms):
    """
    Initialise Beta distribution parameters for each arm.

    Every arm starts with alpha=1 and beta=1,
    representing a uniform prior.
    """
    if not isinstance(n_arms, int) or isinstance(n_arms, bool) or n_arms <= 0:
        raise ValueError("n_arms must be a positive integer.")

    alpha = [1.0] * n_arms
    beta = [1.0] * n_arms

    return alpha, beta


def update_beta_parameters(alpha, beta, arm, reward):
    """
    Update the Beta distribution parameters after observing a reward.
    """
    if len(alpha) != len(beta):
        raise ValueError("Alpha and beta must have equal lengths.")

    if not isinstance(arm, int) or isinstance(arm, bool):
        raise ValueError("Arm must be an integer.")

    if not 0 <= arm < len(alpha):
        raise ValueError("Invalid arm index.")

    if reward not in (0, 1):
        raise ValueError("Reward must be either 0 or 1.")

    if any(a <= 0 for a in alpha) or any(b <= 0 for b in beta):
        raise ValueError("Beta distribution parameters must be positive.")

    if reward == 1:
        alpha[arm] += 1
    else:
        beta[arm] += 1

    return alpha, beta


def thompson_sampling(alpha, beta, rng=None):
    """
    Select an arm using Thompson Sampling.

    Sample a possible reward probability from each arm's
    Beta distribution, then select the arm with the
    highest sampled probability.
    """
    if len(alpha) == 0 or len(alpha) != len(beta):
        raise ValueError("Alpha and beta must have equal, non-empty lengths.")

    if any(a <= 0 for a in alpha) or any(b <= 0 for b in beta):
        raise ValueError("Beta distribution parameters must be positive.")

    if rng is None:
        rng = random

    sampled_probabilities = []

    for a, b in zip(alpha, beta):
        sampled_probability = rng.betavariate(a, b)
        sampled_probabilities.append(sampled_probability)

    selected_arm = max(
        range(len(sampled_probabilities)),
        key=lambda arm: sampled_probabilities[arm],
    )

    return selected_arm


# ==========================================
# EPSILON-GREEDY
# ==========================================

def initialize_action_estimates(n_arms):
    """
    Initialize action counts and estimated reward values.

    Each arm starts with:
    - Zero selections.
    - Zero estimated reward.
    """
    if not isinstance(n_arms, int) or isinstance(n_arms, bool) or n_arms <= 0:
        raise ValueError("n_arms must be a positive integer.")

    action_counts = [0] * n_arms
    estimated_values = [0.0] * n_arms

    return action_counts, estimated_values


def update_action_estimates(action_counts, estimated_values, arm, reward):
    """
    Update the estimated reward value of an arm
    using the incremental average formula.
    """
    if len(action_counts) == 0 or len(action_counts) != len(estimated_values):
        raise ValueError(
            "Action counts and estimated values must have equal, non-empty lengths."
        )

    if any(
        not isinstance(count, int) or isinstance(count, bool) or count < 0
        for count in action_counts
    ):
        raise ValueError("Action counts must be non-negative integers.")

    if any(value < 0 or value > 1 for value in estimated_values):
        raise ValueError("Estimated values must be between 0 and 1.")

    if not isinstance(arm, int) or isinstance(arm, bool):
        raise ValueError("Arm must be an integer.")

    if not 0 <= arm < len(action_counts):
        raise ValueError("Invalid arm index.")

    if reward not in (0, 1):
        raise ValueError("Reward must be either 0 or 1.")

    action_counts[arm] += 1

    old_estimate = estimated_values[arm]

    estimated_values[arm] = old_estimate + (
        reward - old_estimate
    ) / action_counts[arm]

    return action_counts, estimated_values


def epsilon_greedy(estimated_values, epsilon, rng=None):
    """
    Select an arm using the Epsilon-Greedy strategy.

    With probability epsilon, explore by selecting
    a random arm.

    With probability 1 - epsilon, exploit by selecting
    an arm with the highest estimated reward.
    """
    if len(estimated_values) == 0:
        raise ValueError("At least one arm is required.")

    if any(value < 0 or value > 1 for value in estimated_values):
        raise ValueError("Estimated values must be between 0 and 1.")

    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool):
        raise ValueError("Epsilon must be a number.")

    if not 0 <= epsilon <= 1:
        raise ValueError("Epsilon must be between 0 and 1.")

    if rng is None:
        rng = random

    if rng.random() < epsilon:
        return rng.randrange(len(estimated_values))

    best_value = max(estimated_values)

    best_arms = [
        arm
        for arm, value in enumerate(estimated_values)
        if value == best_value
    ]

    return rng.choice(best_arms)
