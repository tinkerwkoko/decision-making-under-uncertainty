"""
Comparison of Q-Learning and Value Iteration on GridWorld.

Q-Learning is a model-free reinforcement learning algorithm that learns
through interaction with the environment. Value Iteration is a model-based
dynamic programming algorithm that requires full knowledge of the environment
model (transition probabilities and rewards).

This module runs both methods under comparable conditions and reports
their performance, while clearly documenting their different information
requirements. The comparison is NOT presented as a fair horse race:
Value Iteration has access to the complete environment model, whereas
Q-Learning must discover the value function through experience.
"""

from environments.gridworld import GridWorld
from algorithms.q_learning import train_q_learning, choose_action
from algorithms.value_iteration import value_iteration, extract_policy, follow_policy
from experiments.q_learning_evaluation import run_single_q_learning_evaluation
from experiments.value_iteration_evaluation import evaluate_value_iteration


def evaluate_q_learning_policy(
    env,
    episodes=500,
    alpha=0.1,
    gamma=0.9,
    epsilon=0.1,
    epsilon_decay=0.995,
    min_epsilon=0.01,
    max_steps=100,
    n_eval_episodes=100,
    seed=None,
):
    """Train Q-learning and evaluate the learned greedy policy."""
    result = run_single_q_learning_evaluation(
        env,
        episodes=episodes,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
        max_steps=max_steps,
        n_eval_episodes=n_eval_episodes,
        seed=seed,
    )
    return {
        "success": result["eval_success_rate"] > 0.0,
        "success_rate": result["eval_success_rate"],
        "avg_steps": result["eval_avg_steps"],
        "avg_reward": result["eval_avg_reward"],
        "episodes": episodes,
        "epsilon": epsilon,
    }


def compare_q_learning_value_iteration(
    env,
    n_runs=5,
    q_episodes=500,
    alpha=0.1,
    gamma=0.9,
    epsilon=0.1,
    epsilon_decay=0.995,
    min_epsilon=0.01,
    max_steps=100,
    n_eval_episodes=100,
    vi_theta=0.000001,
    vi_max_iterations=10000,
    base_seed=42,
):
    """Compare Q-Learning with Value Iteration on the same GridWorld."""
    if not isinstance(n_runs, int) or n_runs <= 0:
        raise ValueError("n_runs must be a positive integer.")
    if not isinstance(base_seed, int):
        raise ValueError("base_seed must be an integer.")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1.")

    q_results = []
    for run in range(n_runs):
        seed = base_seed + run
        q_result = evaluate_q_learning_policy(
            env,
            episodes=q_episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon,
            max_steps=max_steps,
            n_eval_episodes=n_eval_episodes,
            seed=seed,
        )
        q_results.append(q_result)

    q_success_rates = [r["success_rate"] for r in q_results]
    q_avg_steps = [r["avg_steps"] for r in q_results]
    q_avg_rewards = [r["avg_reward"] for r in q_results]
    n = float(n_runs)
    q_mean_success = sum(q_success_rates) / n
    q_mean_steps = sum(q_avg_steps) / n
    q_mean_reward = sum(q_avg_rewards) / n

    q_aggregated = {
        "n_runs": n_runs,
        "episodes": q_episodes,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon,
        "base_seed": base_seed,
        "mean_success_rate": q_mean_success,
        "std_success_rate": (
            sum((s - q_mean_success) ** 2 for s in q_success_rates) / n
        )
        ** 0.5,
        "mean_avg_steps": q_mean_steps,
        "std_avg_steps": (
            sum((s - q_mean_steps) ** 2 for s in q_avg_steps) / n
        )
        ** 0.5,
        "mean_avg_reward": q_mean_reward,
        "std_avg_reward": (
            sum((r - q_mean_reward) ** 2 for r in q_avg_rewards) / n
        )
        ** 0.5,
        "per_run_results": q_results,
    }

    vi_result = evaluate_value_iteration(
        env,
        gamma=gamma,
        theta=vi_theta,
        max_iterations=vi_max_iterations,
        max_steps=max_steps,
        seed=base_seed,
    )

    info_requirements = {
        "q_learning": (
            "Model-free. Learns Q(s,a) through interaction with the "
            "environment. Requires only: state, action, reward, next_state, "
            "done signal. No prior knowledge of transition probabilities "
            "or reward function needed."
        ),
        "value_iteration": (
            "Model-based. Requires full knowledge of the environment "
            "model: transition probabilities P(s'|s,a) and rewards "
            "R(s,a,s'). Computes optimal V*(s) via Bellman optimality "
            "updates. In GridWorld, the model is known because transitions "
            "are deterministic and rewards are fully specified."
        ),
    }

    return {
        "q_learning": q_aggregated,
        "value_iteration": vi_result,
        "info_requirements": info_requirements,
        "gamma": gamma,
        "max_steps": max_steps,
    }