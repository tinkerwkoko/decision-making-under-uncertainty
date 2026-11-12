"""
Visualise POMDP belief updates over time.

Plots how the agent's belief distribution over hidden states evolves
as it receives observations, demonstrating how partial observability
affects the agent's certainty about the true state.
"""

import matplotlib.pyplot as plt

from experiments.pomdp_demonstration import demonstrate_belief_updates, format_belief


def main():
    result = demonstrate_belief_updates(n_steps=10, seed=42)

    states = sorted(result["initial_belief"].keys())
    n_steps = result["n_steps"]

    # Collect belief values for each state over time
    # Time 0 = initial belief, then one entry per step
    belief_history = {s: [] for s in states}

    # Initial belief
    for s in states:
        belief_history[s].append(result["initial_belief"][s])

    # After each step
    for step in result["steps"]:
        for s in states:
            belief_history[s].append(step["belief_after"][s])

    # Plot
    plt.figure(figsize=(10, 6))
    x = range(n_steps + 1)  # initial + n_steps

    for s in states:
        plt.plot(
            x,
            belief_history[s],
            marker="o",
            label=f"State {s}",
            linewidth=2,
        )

    plt.xlabel("Time Step")
    plt.ylabel("Belief Probability")
    plt.title("POMDP Belief Distribution Over Time (seed=42)")
    plt.xticks(list(x))
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("experiments/pomdp_belief_evolution.png", dpi=300)
    plt.close()

    # Print summary
    print("POMDP Belief Evolution")
    print("=" * 40)
    print(f"Initial belief: {format_belief(result['initial_belief'])}")
    print(f"Final belief:   {format_belief(result['final_belief'])}")
    print()
    for step in result["steps"]:
        print(
            f"Step {step['step_index']+1}: action={step['action']}, "
            f"obs={step['observation']}, true_state={step['true_state']}, "
            f"belief={format_belief(step['belief_after'])}"
        )
    print()
    print("Saved: experiments/pomdp_belief_evolution.png")


if __name__ == "__main__":
    main()