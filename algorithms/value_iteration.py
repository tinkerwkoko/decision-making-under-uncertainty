
def get_all_states(env):
    states = []

    for row in range(env.size):
        for col in range(env.size):
            states.append((row, col))

    return states

def initialize_values(states):
    values = {}

    for state in states:
        values[state] = 0.0

    return values


def calculate_action_value(env, values, state, action, gamma=0.9):
    next_state = env.get_next_position(state, action)

    reward = env.step_reward

    if next_state == env.goal:
        future_value = 0.0
    else:
        future_value = values[next_state]

    return reward + gamma * future_value


def calculate_state_value(env, values, state, gamma=0.9):
    if state == env.goal:
        return 0.0

    action_values = []

    for action in env.actions:
        value = calculate_action_value(
            env, values, state, action, gamma
        )
        action_values.append(value)

    return max(action_values)


def value_iteration(env, gamma=0.9, theta=0.000001, max_iterations=10000):
    states = get_all_states(env)
    values = initialize_values(states)

    for iteration in range(1, max_iterations + 1):
        old_values = values.copy()
        new_values = old_values.copy()

        delta = 0.0

        for state in states:
            new_value = calculate_state_value(
                env, old_values, state, gamma
            )

            new_values[state] = new_value

            change = abs(new_value - old_values[state])
            delta = max(delta, change)

        values = new_values

        if delta < theta:
            return values, iteration

    return values, max_iterations

def extract_policy(env, values, gamma=0.9):
    policy = {}

    for state in get_all_states(env):

        if state == env.goal:
            policy[state] = None
            continue

        action_values = {}

        for action in env.actions:
            action_values[action] = calculate_action_value(
                env, values, state, action, gamma
            )

        best_action = max(
            action_values,
            key=action_values.get
        )

        policy[state] = best_action

    return policy

def follow_policy(env, policy):
    state = env.reset()
    path = [state]

    while state != env.goal:
        action = policy[state]

        next_state, reward, done = env.step(action)

        state = next_state
        path.append(state)

        if done:
            break

    return path