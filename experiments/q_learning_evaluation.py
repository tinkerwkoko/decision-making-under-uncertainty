"""
Q-Learning evaluation experiments.

Provides reproducible, multi-run evaluation of Q-learning on GridWorld.
Supports configurable epsilon values (including 0.1 and 0.3), multiple
independent runs, episode rewards, episode lengths, goal-reaching success
rates, and learning curves.

All functions use seeded RNG (random.Random(seed)) for reproducibility.
"""

import random

from environments.gridworld import GridWorld
from algorithms.q_learning import train_q_learning, choose_action


def run_single_q_learning_evaluation(
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
    """Run one Q-learning training cycle followed by greedy evaluation."""
    if not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("episodes must be a positive integer.")
    if not isinstance(n_eval_episodes, int) or n_eval_episodes <= 0:
        raise ValueError("n_eval_episodes must be a positive integer.")
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer.")

    q_table, train_rewards, epsilon_history = train_q_learning(
        env,
        episodes=episodes,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
        max_steps=max_steps,
        seed=seed,
    )

    eval_env = GridWorld(
        size=env.size,
        start=env.start,
        goal=env.goal,
        step_reward=env.step_reward,
    )
    eval_seed = None if seed is None else seed + 10000
    eval_rng = random.Random(eval_seed)

    successes = 0
    eval_episode_rewards = []
    eval_episode_steps = []

    for _ in range(n_eval_episodes):
        state = eval_env.reset()
        total_reward = 0
        steps = 0

        for _ in range(max_steps):
            action = choose_action(
                q_table, state, epsilon=0.0, rng=eval_rng
            )
            next_state, reward, done = eval_env.step(action)
            total_reward += reward
            steps += 1
            state = next_state

            if done:
                successes += 1
                break

        eval_episode_rewards.append(total_reward)
        eval_episode_steps.append(steps)

    return {
        "q_table": q_table,
        "train_rewards": train_rewards,
        "epsilon_history": epsilon_history,
        "eval_success_rate": (successes / n_eval_episodes) * 100.0,
        "eval_avg_reward": sum(eval_episode_rewards) / n_eval_episodes,
        "eval_avg_steps": sum(eval_episode_steps) / n_eval_episodes,
        "eval_successes": successes,
    }


def run_q_learning_evaluation(
    env,
    n_runs=5,
    episodes=500,
    alpha=0.1,
    gamma=0.9,
    epsilon=0.1,
    epsilon_decay=0.995,
    min_epsilon=0.01,
    max_steps=100,
    n_eval_episodes=100,
    base_seed=42,
):
    """Run Q-learning evaluation across multiple independent runs."""
    if not isinstance(n_runs, int) or n_runs <= 0:
        raise ValueError("n_runs must be a positive integer.")
    if not isinstance(base_seed, int):
        raise ValueError("base_seed must be an integer.")

    all_results = []

    for run in range(n_runs):
        seed = base_seed + run
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
        all_results.append(result)

    success_rates = [r["eval_success_rate"] for r in all_results]
    avg_rewards = [r["eval_avg_reward"] for r in all_results]
    avg_steps = [r["eval_avg_steps"] for r in all_results]

    n = float(n_runs)
    mean_success = sum(success_rates) / n
    mean_reward = sum(avg_rewards) / n
    mean_steps = sum(avg_steps) / n

    std_success = (
        sum((s - mean_success) ** 2 for s in success_rates) / n
    ) ** 0.5
    std_reward = (
        sum((r - mean_reward) ** 2 for r in avg_rewards) / n
    ) ** 0.5
    std_steps = (
        sum((s - mean_steps) ** 2 for s in avg_steps) / n
    ) ** 0.5

    return {
        "n_runs": n_runs,
        "epsilon": epsilon,
        "episodes": episodes,
        "alpha": alpha,
        "gamma": gamma,
        "base_seed": base_seed,
        "train_rewards_per_run": [r["train_rewards"] for r in all_results],
        "epsilon_history_per_run": [r["epsilon_history"] for r in all_results],
        "success_rates": success_rates,
        "avg_rewards": avg_rewards,
        "avg_steps": avg_steps,
        "mean_success_rate": mean_success,
        "std_success_rate": std_success,
        "mean_avg_reward": mean_reward,
        "std_avg_reward": std_reward,
        "mean_avg_steps": mean_steps,
        "std_avg_steps": std_steps,
    }


def evaluate_q_learning_across_epsilons(
    env,
    n_runs=5,
    episodes=500,
    alpha=0.1,
    gamma=0.9,
    epsilons=(0.1, 0.3),
    epsilon_decay=0.995,
    min_epsilon=0.01,
    max_steps=100,
    n_eval_episodes=100,
    base_seed=42,
):
    """Evaluate Q-learning across multiple epsilon values."""
    results = {}
    for epsilon in epsilons:
        results[epsilon] = run_q_learning_evaluation(
            env,
            n_runs=n_runs,
            episodes=episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon,
            max_steps=max_steps,
            n_eval_episodes=n_eval_episodes,
            base_seed=base_seed,
        )
    return results