"""
Transfer Learning Experiment for GridWorld.

Compares Q-learning with transferred knowledge (from a source GridWorld task)
against Q-learning learning from scratch (on a target GridWorld task).

Experimental design:

    Source task: GridWorld(4x4, start=(0,0), goal=(3,3))
    Target task: GridWorld(4x4, start=(0,0), goal=(3,2))  # different goal

    State correspondence: Direct mapping — state (r, c) in source maps to
    state (r, c) in target. Since both grids are the same size, every state
    has a corresponding state.

    Transferred knowledge: The converged Q-table from the source task is used
    to initialise the Q-table for the target task. States that exist in both
    tasks inherit the source Q-values; this represents "prior knowledge" about
    which actions are valuable in each state.

    Comparison:
      1. Transfer learning: Train on source, transfer Q-values, then train on target.
      2. From scratch: Train on target with zero-initialised Q-table.

    Metrics compared across multiple independent runs:
      - Learning curves (episode rewards during target training)
      - Goal-reaching success rate after training
      - Average reward per episode
      - Average steps per episode

This is a defensible transfer approach because:
  - The state spaces are identical (same grid size), so no state-space
    transformation is needed.
  - Only compatible Q-values are transferred (same state-action pairs).
  - The target task is related but different (different goal), so transferred
    knowledge provides a head start but still requires adaptation.
  - We do not claim this is meta-Reinforcement Learning; it is a simple,
    explainable knowledge-transfer experiment.

All functions use seeded RNG (random.Random(seed)) for reproducibility.
"""

import random

from environments.gridworld import GridWorld
from algorithms.q_learning import (
    train_q_learning,
    choose_action,
    extract_q_policy,
)


def _make_source_gridworld():
    """Create the source GridWorld environment."""
    return GridWorld(size=4, start=(0, 0), goal=(3, 3), step_reward=-1)


def _make_target_gridworld():
    """Create the target GridWorld environment."""
    return GridWorld(size=4, start=(0, 0), goal=(3, 2), step_reward=-1)


def _map_source_q_table_to_target(source_q_table, target_env):
    """Transfer Q-values from source task to target task via direct state mapping."""
    target_q_table = {}

    for row in range(target_env.size):
        for col in range(target_env.size):
            state = (row, col)
            target_q_table[state] = {}
            for action in target_env.actions:
                if state in source_q_table and action in source_q_table[state]:
                    target_q_table[state][action] = source_q_table[state][action]
                else:
                    target_q_table[state][action] = 0.0

    return target_q_table


def _evaluate_q_table_on_target(q_table, env, n_episodes=100, max_steps=100, seed=42):
    """Evaluate a Q-table greedily on the target GridWorld.

    Uses epsilon=0.0 (pure exploitation) to measure the quality of the
    learned or transferred policy.
    """
    rng = random.Random(seed)

    successes = 0
    episode_rewards = []
    episode_steps = []

    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        for _ in range(max_steps):
            action = choose_action(q_table, state, epsilon=0.0, rng=rng)
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            state = next_state

            if done:
                successes += 1
                break

        episode_rewards.append(total_reward)
        episode_steps.append(steps)

    return {
        "success_rate": (successes / n_episodes) * 100.0,
        "avg_reward": sum(episode_rewards) / n_episodes,
        "avg_steps": sum(episode_steps) / n_episodes,
        "successes": successes,
    }


def run_transfer_learning_experiment(
    n_runs=5,
    source_episodes=2000,
    target_episodes=1000,
    alpha=0.1,
    gamma=0.9,
    epsilon=1.0,
    epsilon_decay=0.995,
    min_epsilon=0.01,
    max_steps=100,
    n_eval_episodes=100,
    base_seed=42,
):
    """Run transfer learning vs from-scratch comparison on GridWorld."""
    if not isinstance(n_runs, int) or n_runs <= 0:
        raise ValueError("n_runs must be a positive integer.")
    if not isinstance(base_seed, int):
        raise ValueError("base_seed must be an integer.")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be greater than 0 and at most 1.")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1.")
    if not isinstance(source_episodes, int) or source_episodes <= 0:
        raise ValueError("source_episodes must be a positive integer.")
    if not isinstance(target_episodes, int) or target_episodes <= 0:
        raise ValueError("target_episodes must be a positive integer.")

    transfer_results = []
    from_scratch_results = []
    source_q_table = None

    for run in range(n_runs):
        seed = base_seed + run

        source_env = _make_source_gridworld()
        source_q, source_rewards, source_eps = train_q_learning(
            source_env,
            episodes=source_episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon,
            max_steps=max_steps,
            seed=seed,
        )
        source_q_table = source_q

        target_env = _make_target_gridworld()
        transferred_q_table = _map_source_q_table_to_target(source_q_table, target_env)

        transfer_q_table, transfer_rewards, _ = _train_from_q_table(
            target_env,
            transferred_q_table,
            episodes=target_episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon,
            max_steps=max_steps,
            seed=seed + 1000,
        )

        from_scratch_q_table, from_scratch_rewards, _ = train_q_learning(
            target_env,
            episodes=target_episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon,
            max_steps=max_steps,
            seed=seed + 2000,
        )

        transfer_eval = _evaluate_q_table_on_target(
            transfer_q_table, target_env, n_episodes=n_eval_episodes, seed=seed + 3000
        )
        from_scratch_eval = _evaluate_q_table_on_target(
            from_scratch_q_table, target_env, n_episodes=n_eval_episodes, seed=seed + 3000
        )

        transfer_results.append({
            "train_rewards": transfer_rewards,
            "eval_success_rate": transfer_eval["success_rate"],
            "eval_avg_reward": transfer_eval["avg_reward"],
            "eval_avg_steps": transfer_eval["avg_steps"],
            "eval_successes": transfer_eval["successes"],
        })

        from_scratch_results.append({
            "train_rewards": from_scratch_rewards,
            "eval_success_rate": from_scratch_eval["success_rate"],
            "eval_avg_reward": from_scratch_eval["avg_reward"],
            "eval_avg_steps": from_scratch_eval["avg_steps"],
            "eval_successes": from_scratch_eval["successes"],
        })

    return _aggregate_results(transfer_results, from_scratch_results, source_q_table,
                              n_runs, source_episodes, target_episodes, base_seed)


def _train_from_q_table(
    env,
    q_table,
    episodes=1000,
    alpha=0.1,
    gamma=0.9,
    epsilon=1.0,
    epsilon_decay=0.995,
    min_epsilon=0.01,
    max_steps=100,
    seed=None,
):
    """Train Q-learning starting from a given Q-table (not zeros).

    Used for transfer learning: target Q-table initialised with transferred
    source Q-values, then further trained on the target.
    """
    if not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("episodes must be a positive integer.")
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer.")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be greater than 0 and at most 1.")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1.")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be between 0 and 1.")
    if not 0.0 < epsilon_decay <= 1.0:
        raise ValueError("epsilon_decay must be greater than 0 and at most 1.")
    if not 0.0 <= min_epsilon <= 1.0:
        raise ValueError("min_epsilon must be between 0 and 1.")

    rng = random.Random(seed)
    q_table = {s: dict(actions) for s, actions in q_table.items()}

    episode_rewards = []
    epsilon_history = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        epsilon_history.append(epsilon)

        for _ in range(max_steps):
            action = choose_action(q_table, state, epsilon, rng)
            next_state, reward, done = env.step(action)

            current_q = q_table[state][action]
            if done:
                future_q = 0.0
            else:
                future_q = max(q_table[next_state].values())

            target = reward + gamma * future_q
            new_q = current_q + alpha * (target - current_q)
            q_table[state][action] = new_q

            total_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    return q_table, episode_rewards, epsilon_history


def _aggregate_results(transfer_results, from_scratch_results, source_q_table,
                       n_runs, source_episodes, target_episodes, base_seed):
    """Aggregate per-run results into means and standard deviations."""
    n = float(n_runs)

    def mean(values):
        return sum(values) / n

    def std(values, m):
        return (sum((v - m) ** 2 for v in values) / n) ** 0.5

    t_success = [r["eval_success_rate"] for r in transfer_results]
    t_reward = [r["eval_avg_reward"] for r in transfer_results]
    t_steps = [r["eval_avg_steps"] for r in transfer_results]
    t_mean_success = mean(t_success)
    t_mean_reward = mean(t_reward)
    t_mean_steps = mean(t_steps)

    f_success = [r["eval_success_rate"] for r in from_scratch_results]
    f_reward = [r["eval_avg_reward"] for r in from_scratch_results]
    f_steps = [r["eval_avg_steps"] for r in from_scratch_results]
    f_mean_success = mean(f_success)
    f_mean_reward = mean(f_reward)
    f_mean_steps = mean(f_steps)

    return {
        "n_runs": n_runs,
        "source_episodes": source_episodes,
        "target_episodes": target_episodes,
        "base_seed": base_seed,
        "source_q_table": source_q_table,
        "transfer": {
            "mean_success_rate": t_mean_success,
            "std_success_rate": std(t_success, t_mean_success),
            "mean_avg_reward": t_mean_reward,
            "std_avg_reward": std(t_reward, t_mean_reward),
            "mean_avg_steps": t_mean_steps,
            "std_avg_steps": std(t_steps, t_mean_steps),
            "success_rates": t_success,
            "avg_rewards": t_reward,
            "avg_steps": t_steps,
            "train_rewards_per_run": [r["train_rewards"] for r in transfer_results],
        },
        "from_scratch": {
            "mean_success_rate": f_mean_success,
            "std_success_rate": std(f_success, f_mean_success),
            "mean_avg_reward": f_mean_reward,
            "std_avg_reward": std(f_reward, f_mean_reward),
            "mean_avg_steps": f_mean_steps,
            "std_avg_steps": std(f_steps, f_mean_steps),
            "success_rates": f_success,
            "avg_rewards": f_reward,
            "avg_steps": f_steps,
            "train_rewards_per_run": [r["train_rewards"] for r in from_scratch_results],
        },
        "transfer_minus_from_scratch": {
            "success_rate_diff": t_mean_success - f_mean_success,
            "avg_reward_diff": t_mean_reward - f_mean_reward,
            "avg_steps_diff": t_mean_steps - f_mean_steps,
        },
    }