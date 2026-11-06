import sys
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.gridworld import GridWorld
from algorithms.q_learning import (
    train_q_learning,
    choose_action,
)


def evaluate_agent(q_table, episodes=100, seed=42, max_steps=100):
    env = GridWorld()
    rng = random.Random(seed)

    successes = 0
    episode_steps = []
    episode_rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        for step in range(max_steps):
            # Greedy action selection: no epsilon exploration
            action = choose_action(
                q_table,
                state,
                epsilon=0.0,
                rng=rng,
            )

            next_state, reward, done = env.step(action)

            total_reward += reward
            steps += 1
            state = next_state

            if done:
                successes += 1
                break

        episode_steps.append(steps)
        episode_rewards.append(total_reward)

    success_rate = (successes / episodes) * 100
    average_steps = sum(episode_steps) / episodes
    average_reward = sum(episode_rewards) / episodes

    return success_rate, average_steps, average_reward


def main():
    env = GridWorld()

    # Train the agent
    q_table, rewards, epsilon_history = train_q_learning(
        env,
        episodes=5000,
        seed=42,
    )

    # Evaluate the learned Q-table
    success_rate, average_steps, average_reward = evaluate_agent(
        q_table,
        episodes=100,
        seed=42,
    )

    print("\nQ-learning Evaluation Results")
    print("-----------------------------")
    print("Evaluation episodes: 100")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Average steps: {average_steps:.2f}")
    print(f"Average reward: {average_reward:.2f}")


if __name__ == "__main__":
    main()