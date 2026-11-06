
import pytest
import random

from environments.gridworld import GridWorld

from algorithms.q_learning import (
    initialize_q_table,
    choose_action,
    update_q_value,
    train_q_learning,
    extract_q_policy,
)


def test_initialize_q_table():
    env = GridWorld()

    q_table = initialize_q_table(env)

    assert len(q_table) == 16
    assert (0, 0) in q_table
    assert (3, 3) in q_table


def test_all_q_values_start_at_zero():
    env = GridWorld()

    q_table = initialize_q_table(env)

    for state in q_table:
        for action in q_table[state]:
            assert q_table[state][action] == 0.0


def test_q_table_contains_all_actions():
    env = GridWorld()

    q_table = initialize_q_table(env)

    for state in q_table:
        assert list(q_table[state].keys()) == env.actions


def test_choose_action_returns_valid_action():
    env = GridWorld()
    q_table = initialize_q_table(env)

    action = choose_action(
        q_table,
        (0, 0),
        epsilon=0.5,
        rng=random.Random(42),
    )

    assert action in env.actions


def test_greedy_action_selects_highest_q_value():
    env = GridWorld()
    q_table = initialize_q_table(env)

    q_table[(0, 0)] = {
        "up": -5.0,
        "down": 2.0,
        "left": -3.0,
        "right": 1.0,
    }

    action = choose_action(
        q_table,
        (0, 0),
        epsilon=0.0,
        rng=random.Random(42),
    )

    assert action == "down"


def test_choose_action_rejects_invalid_epsilon():
    env = GridWorld()
    q_table = initialize_q_table(env)

    with pytest.raises(ValueError):
        choose_action(q_table, (0, 0), epsilon=1.5)


def test_q_value_update():
    env = GridWorld()
    q_table = initialize_q_table(env)

    q_table[(0, 0)]["right"] = 2.0
    q_table[(0, 1)]["down"] = 5.0

    new_value = update_q_value(
        q_table,
        state=(0, 0),
        action="right",
        reward=-1,
        next_state=(0, 1),
        alpha=0.1,
        gamma=0.9,
    )

    assert new_value == pytest.approx(2.15)


def test_terminal_update_has_no_future_value():
    env = GridWorld()
    q_table = initialize_q_table(env)

    new_value = update_q_value(
        q_table,
        state=(3, 2),
        action="right",
        reward=-1,
        next_state=(3, 3),
        alpha=0.1,
        gamma=0.9,
        done=True,
    )

    assert new_value == pytest.approx(-0.1)


def test_update_rejects_invalid_learning_rate():
    env = GridWorld()
    q_table = initialize_q_table(env)

    with pytest.raises(ValueError):
        update_q_value(
            q_table,
            (0, 0),
            "right",
            -1,
            (0, 1),
            alpha=1.5,
        )


def test_update_rejects_invalid_discount_factor():
    env = GridWorld()
    q_table = initialize_q_table(env)

    with pytest.raises(ValueError):
        update_q_value(
            q_table,
            (0, 0),
            "right",
            -1,
            (0, 1),
            gamma=1.5,
        )


def test_training_returns_expected_outputs():
    env = GridWorld()

    q_table, rewards, epsilon_history = train_q_learning(
        env,
        episodes=100,
        seed=42,
    )

    assert len(q_table) == 16
    assert len(rewards) == 100
    assert len(epsilon_history) == 100


def test_training_is_reproducible():
    env = GridWorld()

    q_table_1, rewards_1, epsilon_1 = train_q_learning(
        env,
        episodes=100,
        seed=42,
    )

    q_table_2, rewards_2, epsilon_2 = train_q_learning(
        env,
        episodes=100,
        seed=42,
    )

    assert q_table_1 == q_table_2
    assert rewards_1 == rewards_2
    assert epsilon_1 == epsilon_2


def test_extract_q_policy():
    env = GridWorld()
    q_table = initialize_q_table(env)

    q_table[(3, 2)]["right"] = 10.0

    policy = extract_q_policy(
        env,
        q_table,
        rng=random.Random(42),
    )

    assert policy[(3, 2)] == "right"
    assert policy[env.goal] is None
