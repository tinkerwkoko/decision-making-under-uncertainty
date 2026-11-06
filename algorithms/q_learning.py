
import random


def initialize_q_table(env):
    """Create a Q-table with zero values for every state-action pair."""
    q_table = {}

    for row in range(env.size):
        for col in range(env.size):
            state = (row, col)

            q_table[state] = {
                action: 0.0 for action in env.actions
            }

    return q_table


def choose_action(q_table, state, epsilon=0.1, rng=None):
    """
    Select an action using epsilon-greedy exploration.

    Explore randomly with probability epsilon.
    Otherwise, choose an action with the highest Q-value.
    Randomly break ties between equally valued actions.
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be between 0 and 1.")

    if rng is None:
        rng = random

    actions = list(q_table[state].keys())

    if rng.random() < epsilon:
        return rng.choice(actions)

    best_value = max(q_table[state].values())

    best_actions = [
        action
        for action, value in q_table[state].items()
        if value == best_value
    ]

    return rng.choice(best_actions)


def update_q_value(
    q_table,
    state,
    action,
    reward,
    next_state,
    alpha=0.1,
    gamma=0.9,
    done=False,
):
    """
    Update a Q-value using the Q-learning update equation.
    """

    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be greater than 0 and at most 1.")

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1.")

    current_q = q_table[state][action]

    if done:
        future_q = 0.0
    else:
        future_q = max(q_table[next_state].values())

    target = reward + gamma * future_q

    new_q = current_q + alpha * (target - current_q)

    q_table[state][action] = new_q

    return new_q


def train_q_learning(
    env,
    episodes=5000,
    alpha=0.1,
    gamma=0.9,
    epsilon=1.0,
    epsilon_decay=0.995,
    min_epsilon=0.01,
    max_steps=100,
    seed=None,
):
    """
    Train an agent using Q-learning with epsilon decay.

    Returns:
        q_table: Learned state-action values.
        episode_rewards: Total reward received in each episode.
        epsilon_history: Exploration rate used in each episode.
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

    if min_epsilon > epsilon:
        raise ValueError("min_epsilon cannot be greater than initial epsilon.")

    rng = random.Random(seed)

    q_table = initialize_q_table(env)

    episode_rewards = []
    epsilon_history = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        epsilon_history.append(epsilon)

        for step in range(max_steps):
            action = choose_action(
                q_table, state, epsilon, rng
            )

            next_state, reward, done = env.step(action)

            update_q_value(
                q_table,
                state,
                action,
                reward,
                next_state,
                alpha,
                gamma,
                done,
            )

            total_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)

        epsilon = max(
            min_epsilon,
            epsilon * epsilon_decay
        )

    return q_table, episode_rewards, epsilon_history


def extract_q_policy(env, q_table, rng=None):
    """
    Extract a policy from the learned Q-table.

    Randomly break ties between actions with equal Q-values.
    """
    if rng is None:
        rng = random

    policy = {}

    for row in range(env.size):
        for col in range(env.size):
            state = (row, col)

            if state == env.goal:
                policy[state] = None
                continue

            best_value = max(q_table[state].values())

            best_actions = [
                action
                for action, value in q_table[state].items()
                if value == best_value
            ]

            policy[state] = rng.choice(best_actions)

    return policy
