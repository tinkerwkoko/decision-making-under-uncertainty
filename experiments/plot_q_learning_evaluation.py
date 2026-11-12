"""
Visualise Q-learning evaluation results across epsilon values.

Plots average training reward learning curves for epsilon=0.1 and epsilon=0.3,
showing how different exploration rates affect learning speed.
"""

import matplotlib.pyplot as plt

from environments.gridworld import GridWorld
from experiments.q_learning_evaluation import evaluate_q_learning_across_epsilons


def compute_windowed_average(rewards, window_size=50):
    """Compute average reward over non-overlapping windows."""
    averages = []
    for i in range(0, len(rewards), window_size):
        window = rewards[i:i + window_size]
        if window:
            averages.append(sum(window) / len(window))
    return averages


def main():
    env = GridWorld()
    epsilons = (0.1, 0.3)

    results = evaluate_q_learning_across_epsilons(
        env,
        n_runs=5,
        episodes=1000,
        alpha=0.1,
        gamma=0.9,
        epsilons=epsilons,
        base_seed=42,
        n_eval_episodes=100,
    )

    plt.figure(figsize=(10, 6))

    for epsilon in epsilons:
        # Average training rewards across runs at each episode
        all_rewards = results[epsilon]["train_rewards_per_run"]
        n_episodes = len(all_rewards[0])

        # Compute mean reward per episode across runs
        mean_per_episode = []
        for ep in range(n_episodes):
            values = [run[ep] for run in all_rewards]
            mean_per_episode.append(sum(values) / len(values))

        windowed = compute_windowed_average(mean_per_episode, window_size=50)
        x = [i * 50 + 50 for i in range(len(windowed))]

        plt.plot(
            x,
            windowed,
            label=f"Epsilon-Greedy (epsilon={epsilon})",
            linewidth=2,
        )

    plt.xlabel("Episode")
    plt.ylabel("Average Reward (50-episode window)")
    plt.title("Q-Learning: Learning Curves Across Epsilon Values")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiments/q_learning_learning_curves.png", dpi=300)
    plt.close()

    # Print summary
    for epsilon in epsilons:
        r = results[epsilon]
        print(f"Epsilon={epsilon}:")
        print(f"  Mean success rate: {r['mean_success_rate']:.2f}% "
              f"(+/- {r['std_success_rate']:.2f})")
        print(f"  Mean avg reward: {r['mean_avg_reward']:.2f} "
              f"(+/- {r['std_avg_reward']:.2f})")
        print(f"  Mean avg steps: {r['mean_avg_steps']:.2f} "
              f"(+/- {r['std_avg_steps']:.2f})")

    print("\nSaved: experiments/q_learning_learning_curves.png")


if __name__ == "__main__":
    main()