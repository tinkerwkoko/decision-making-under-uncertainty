"""
Visualise Q-Learning vs Value Iteration comparison.

Creates a bar chart comparing success rate, average steps, and average
reward between Q-learning (aggregated across runs) and Value Iteration
(single deterministic run). Also annotates the information requirement
difference between the two methods.
"""

import matplotlib.pyplot as plt

from environments.gridworld import GridWorld
from experiments.compare_q_learning_value_iteration import compare_q_learning_value_iteration


def main():
    env = GridWorld()
    comparison = compare_q_learning_value_iteration(
        env,
        n_runs=5,
        q_episodes=1000,
        epsilon=0.1,
        base_seed=42,
        n_eval_episodes=100,
    )

    ql = comparison["q_learning"]
    vi = comparison["value_iteration"]

    metrics = ["Success Rate (%)", "Avg Steps", "Avg Reward"]
    ql_values = [ql["mean_success_rate"], ql["mean_avg_steps"], ql["mean_avg_reward"]]
    vi_values = [100.0 if vi["success"] else 0.0, vi["steps"], vi["total_reward"] / max(vi["steps"], 1)]

    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        [i - width / 2 for i in x],
        ql_values,
        width,
        label="Q-Learning (model-free, 5 runs, mean)",
        color="#1f77b4",
        alpha=0.8,
    )
    ax.bar(
        [i + width / 2 for i in x],
        vi_values,
        width,
        label="Value Iteration (model-based, single run)",
        color="#ff7f0e",
        alpha=0.8,
    )

    ax.set_ylabel("Value")
    ax.set_title("Q-Learning vs Value Iteration on GridWorld")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("experiments/q_learning_vs_value_iteration.png", dpi=300)
    plt.close()

    # Print summary
    print("Q-Learning vs Value Iteration Comparison")
    print("=" * 50)
    print(f"Q-Learning success rate: {ql['mean_success_rate']:.2f}% "
          f"(+/- {ql['std_success_rate']:.2f})")
    print(f"Value Iteration success: {vi['success']}")
    print(f"Q-Learning avg steps: {ql['mean_avg_steps']:.2f} "
          f"(+/- {ql['std_avg_steps']:.2f})")
    print(f"Value Iteration steps: {vi['steps']}")
    print(f"Q-Learning avg reward: {ql['mean_avg_reward']:.2f}")
    print(f"Value Iteration total reward: {vi['total_reward']:.2f}")
    print()
    print("Information requirements:")
    print(f"  Q-Learning: {comparison['info_requirements']['q_learning']}")
    print(f"  Value Iteration: {comparison['info_requirements']['value_iteration']}")
    print()
    print("Saved: experiments/q_learning_vs_value_iteration.png")


if __name__ == "__main__":
    main()