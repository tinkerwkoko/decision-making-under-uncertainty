"""
Visualise Transfer Learning results.

Plots learning curves comparing transfer learning vs from-scratch Q-learning
on the target GridWorld task. Also prints a summary table of evaluation
metrics (success rate, average reward, average steps).
"""

import matplotlib.pyplot as plt

from experiments.transfer_learning import run_transfer_learning_experiment


def compute_windowed_average(rewards, window_size=50):
    """Compute average reward over non-overlapping windows."""
    averages = []
    for i in range(0, len(rewards), window_size):
        window = rewards[i:i + window_size]
        if window:
            averages.append(sum(window) / len(window))
    return averages


def main():
    results = run_transfer_learning_experiment(
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
    )

    transfer = results["transfer"]
    from_scratch = results["from_scratch"]
    diff = results["transfer_minus_from_scratch"]

    # --- Plot learning curves on target task ---
    plt.figure(figsize=(10, 6))

    # Average training reward per episode across runs, for transfer
    t_all = transfer["train_rewards_per_run"]
    t_n_episodes = len(t_all[0])
    t_mean_per_episode = []
    for ep in range(t_n_episodes):
        values = [run[ep] for run in t_all]
        t_mean_per_episode.append(sum(values) / len(values))

    # Average training reward per episode across runs, for from-scratch
    f_all = from_scratch["train_rewards_per_run"]
    f_n_episodes = len(f_all[0])
    f_mean_per_episode = []
    for ep in range(f_n_episodes):
        values = [run[ep] for run in f_all]
        f_mean_per_episode.append(sum(values) / len(values))

    t_windowed = compute_windowed_average(t_mean_per_episode, window_size=50)
    f_windowed = compute_windowed_average(f_mean_per_episode, window_size=50)
    x = [i * 50 + 50 for i in range(len(t_windowed))]

    plt.plot(x, t_windowed, label="Transfer Learning", linewidth=2, color="#1f77b4")
    plt.plot(x, f_windowed, label="From Scratch", linewidth=2, color="#ff7f0e")

    plt.xlabel("Episode (on target task)")
    plt.ylabel("Average Reward (50-episode window)")
    plt.title("Transfer Learning vs From-Scratch on Target GridWorld")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiments/transfer_learning_curves.png", dpi=300)
    plt.close()

    # --- Print summary ---
    print("Transfer Learning Experiment Results")
    print("=" * 50)
    print(f"Source task:  GridWorld 4x4, goal (3,3), {results['source_episodes']} episodes")
    print(f"Target task:  GridWorld 4x4, goal (3,2), {results['target_episodes']} episodes")
    print(f"Independent runs: {results['n_runs']}, base seed: {results['base_seed']}")
    print()
    print(f"{'Metric':<25} {'Transfer':>12} {'From Scratch':>12} {'Diff':>10}")
    print("-" * 50)
    print(f"{'Success rate (%)':<25} "
          f"{transfer['mean_success_rate']:>11.2f} "
          f"{from_scratch['mean_success_rate']:>11.2f} "
          f"{diff['success_rate_diff']:>9.2f}")
    print(f"{'Avg reward':<25} "
          f"{transfer['mean_avg_reward']:>11.2f} "
          f"{from_scratch['mean_avg_reward']:>11.2f} "
          f"{diff['avg_reward_diff']:>9.2f}")
    print(f"{'Avg steps':<25} "
          f"{transfer['mean_avg_steps']:>11.2f} "
          f"{from_scratch['mean_avg_steps']:>11.2f} "
          f"{diff['avg_steps_diff']:>9.2f}")
    print()
    print(f"Transfer success rate (+/- {transfer['std_success_rate']:.2f}): "
          f"{transfer['mean_success_rate']:.2f}%")
    print(f"From-scratch success rate (+/- {from_scratch['std_success_rate']:.2f}): "
          f"{from_scratch['mean_success_rate']:.2f}%")
    print()
    print("Interpretation:")
    if diff["success_rate_diff"] > 0:
        print("  Transfer learning achieved a HIGHER success rate than from-scratch.")
        print("  Transferred Q-values from the source task provided a head start.")
    elif diff["success_rate_diff"] < 0:
        print("  Transfer learning achieved a LOWER success rate than from-scratch.")
        print("  Transferred knowledge may have been misleading for the target goal.")
    else:
        print("  Transfer learning and from-scratch achieved similar success rates.")
    print()
    print("Saved: experiments/transfer_learning_curves.png")
    print()
    print("Limitations:")
    print("  - Same grid size means direct state correspondence; different grid sizes")
    print("    would require a state-mapping function.")
    print("  - This is simple Q-value transfer, not meta-Reinforcement Learning.")
    print("  - Results are specific to this pair of source/target GridWorlds.")


if __name__ == "__main__":
    main()