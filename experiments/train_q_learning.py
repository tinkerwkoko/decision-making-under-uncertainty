import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from environments.gridworld import GridWorld
from algorithms.q_learning import train_q_learning


def main():
    # Create the environment
    env = GridWorld()

    # Train the Q-learning agent
    q_table, rewards, epsilon_history = train_q_learning(
        env,
        episodes=5000,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01,
        seed=42,
    )

    # Calculate average reward over windows of 100 episodes
    window_size = 100

    average_rewards = [
        sum(rewards[i:i + window_size]) / window_size
        for i in range(0, len(rewards), window_size)
    ]

    # Plot episode rewards
    plt.figure(figsize=(10, 5))
    plt.plot(
        range(window_size, len(rewards) + 1, window_size),
        average_rewards,
    )

    plt.title("Q-learning: Average Reward During Training")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward (100-episode window)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("experiments/q_learning_rewards.png")
    plt.show()

    # Plot epsilon decay
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(epsilon_history) + 1), epsilon_history)

    plt.title("Q-learning: Epsilon Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("experiments/q_learning_epsilon.png")
    plt.show()

    print("Training completed.")
    print(f"Total episodes: {len(rewards)}")
    print(f"Final epsilon: {epsilon_history[-1]:.4f}")
    print(f"Final average reward: {average_rewards[-1]:.2f}")


if __name__ == "__main__":
    main()