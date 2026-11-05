
from environments.gridworld import GridWorld

from algorithms.value_iteration import (
    get_all_states,
    initialize_values,
    value_iteration,
    extract_policy,
    follow_policy,
)


def test_get_all_states():
    env = GridWorld()

    states = get_all_states(env)

    assert len(states) == 16
    assert (0, 0) in states
    assert (3, 3) in states


def test_initialize_values():
    env = GridWorld()

    states = get_all_states(env)
    values = initialize_values(states)

    assert len(values) == 16
    assert all(value == 0.0 for value in values.values())


def test_goal_value_is_zero():
    env = GridWorld()

    values, iterations = value_iteration(env)

    assert values[env.goal] == 0.0


def test_start_value_is_correct():
    env = GridWorld()

    values, iterations = value_iteration(env)

    assert round(values[env.start], 5) == -4.68559


def test_states_closer_to_goal_have_higher_values():
    env = GridWorld()

    values, iterations = value_iteration(env)

    assert values[(3, 2)] > values[(0, 0)]


def test_policy_selects_action_toward_goal():
    env = GridWorld()

    values, iterations = value_iteration(env)
    policy = extract_policy(env, values)

    assert policy[(3, 2)] == "right"
    assert policy[env.goal] is None


def test_follow_policy_reaches_goal():
    env = GridWorld()

    values, iterations = value_iteration(env)
    policy = extract_policy(env, values)

    path = follow_policy(env, policy)

    assert path[0] == env.start
    assert path[-1] == env.goal
    assert len(path) - 1 == 6
