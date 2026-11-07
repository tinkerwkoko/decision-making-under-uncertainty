
import matplotlib.pyplot as plt

from experiments.run_bandit_comparison import run_comparison


PROBABILITIES = [0.1, 0.3, 0.5, 0.8]
N_ROUNDS = 1000
N_SIMULATIONS = 100


def main():
    results = run_comparison(
        probabilities=PROBABILITIES,
        n_rounds=N_ROUNDS,
        n_simulations=N_SIMULATIONS,
    )

    rounds = range(1, N_ROUNDS + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(
        rounds,
        results["epsilon_greedy"],
        label="Epsilon-Greedy (epsilon=0.1)",
    )

    plt.plot(
        rounds,
        results["thompson_sampling"],
        label="Thompson Sampling",
    )

    plt.xlabel("Number of Rounds")
    plt.ylabel("Average Cumulative Expected Regret")
    plt.title("Epsilon-Greedy vs Thompson Sampling")

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        "experiments/bandit_regret_comparison.png",
        dpi=300,
    )

    plt.show()

    print(
        "Epsilon-Greedy final regret:",
        round(results["epsilon_greedy"][-1], 3),
    )

    print(
        "Thompson Sampling final regret:",
        round(results["thompson_sampling"][-1], 3),
    )


if __name__ == "__main__":
    main()
