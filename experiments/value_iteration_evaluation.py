"""
Value Iteration evaluation experiments.

Provides reproducible evaluation of Value Iteration on GridWorld.
Value Iteration assumes a known environment model (the GridWorld's
deterministic transitions and rewards) and computes the optimal
state-value function and policy via Bellman optimality updates.

All functions use seeded RNG where randomness is involved
(environment reset is deterministic in GridWorld, but seeding is
accepted for API consistency and future stochastic environments).
"""

from environments.gridworld import GridWorld
from algorithms.value_iteration import value_iteration, extract_policy, follow_policy


def evaluate_value_iteration(
    env,
    gamma=0.9,
    theta=0.000001,
    max_iterations=10000,
    max_steps=100,
    seed=None,
):
    """Run Value Iteration and evaluate the resulting policy.

    Value Iteration computes the optimal state-value function V*(s)
    using Bellman optimality updates, then extracts the optimal
    policy pi*(s) = argmax_a Q(s,a). The policy is followed from
    the start state to measure success, steps, and reward.

    Args:
        env: GridWorld environment (provides the known model).
        gamma: Discount factor.
        theta: Convergence threshold (stop when max change < theta).
        max_iterations: Maximum number of Bellman sweeps.
        max_steps: Maximum steps when following the policy.
        seed: Accepted for API consistency (GridWorld is deterministic).

    Returns:
        dict with keys:
            - values: Optimal state-value function.
            - iterations: Number of iterations until convergence.
            - converged: Whether convergence was reached before max_iterations.
            - policy: Extracted optimal policy (state -> action or None).
            - path: State sequence from start to goal (or termination).
            - success: Whether the policy reached the goal.
            - steps: Number of steps taken.
            - total_reward: Cumulative reward along the path.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1.")
    if theta <= 0:
        raise ValueError("theta must be greater than 0.")
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer.")
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer.")

    values, iterations = value_iteration(
        env, gamma=gamma, theta=theta, max_iterations=max_iterations
    )
    converged = iterations < max_iterations

    policy = extract_policy(env, values, gamma=gamma)

    # Follow the policy on a fresh environment instance
    eval_env = GridWorld(
        size=env.size,
        start=env.start,
        goal=env.goal,
        step_reward=env.step_reward,
    )
    path = follow_policy(eval_env, policy)

    success = path[-1] == eval_env.goal
    steps = len(path) - 1
    total_reward = steps * eval_env.step_reward

    return {
        "values": values,
        "iterations": iterations,
        "converged": converged,
        "policy": policy,
        "path": path,
        "success": success,
        "steps": steps,
        "total_reward": total_reward,
        "gamma": gamma,
        "theta": theta,
    }