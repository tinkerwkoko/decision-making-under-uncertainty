from environments.gridworld import GridWorld

from algorithms.value_iteration import (
    get_all_states,
    initialize_values,
    value_iteration,
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
    hi hi i