
import random
import pytest

from algorithms.bandit_algorithms import (
    initialize_action_estimates,
    update_action_estimates,
    epsilon_greedy,
)


def test_initialize_action_estimates():
    counts, values = initialize_action_estimates(4)

    assert counts == [0, 0, 0, 0]
    assert values == [0.0, 0.0, 0.0, 0.0]


def test_initialize_rejects_zero_arms():
    with pytest.raises(ValueError):
        initialize_action_estimates(0)


def test_initialize_rejects_negative_arms():
    with pytest.raises(ValueError):
        initialize_action_estimates(-2)


def test_update_action_estimates():
    counts = [0, 0]
    values = [0.0, 0.0]

    counts, values = update_action_estimates(
        counts, values, 0, 1
    )

    assert counts == [1, 0]
    assert values == [1.0, 0.0]


def test_incremental_average():
    counts = [0]
    values = [0.0]

    for reward in [1, 0, 1, 0]:
        counts, values = update_action_estimates(
            counts, values, 0, reward
        )

    assert counts == [4]
    assert values[0] == pytest.approx(0.5)


def test_update_rejects_invalid_arm():
    with pytest.raises(ValueError):
        update_action_estimates([0, 0], [0.0, 0.0], 2, 1)


def test_update_rejects_invalid_reward():
    with pytest.raises(ValueError):
        update_action_estimates([0], [0.0], 0, 2)


def test_update_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        update_action_estimates([0, 0], [0.0], 0, 1)


def test_epsilon_greedy_exploits_when_epsilon_is_zero():
    values = [0.2, 0.8, 0.4]

    selected_arm = epsilon_greedy(
        values,
        epsilon=0,
        rng=random.Random(42),
    )

    assert selected_arm == 1


def test_epsilon_greedy_explores_when_epsilon_is_one():
    values = [0.2, 0.8, 0.4]

    selected_arm = epsilon_greedy(
        values,
        epsilon=1,
        rng=random.Random(42),
    )

    assert selected_arm in [0, 1, 2]


def test_epsilon_greedy_returns_valid_arm():
    values = [0.2, 0.8, 0.4]

    selected_arm = epsilon_greedy(
        values,
        epsilon=0.1,
        rng=random.Random(42),
    )

    assert 0 <= selected_arm < len(values)


def test_epsilon_greedy_rejects_empty_values():
    with pytest.raises(ValueError):
        epsilon_greedy([], epsilon=0.1)


def test_epsilon_greedy_rejects_invalid_epsilon():
    with pytest.raises(ValueError):
        epsilon_greedy([0.2, 0.8], epsilon=1.5)


def test_epsilon_greedy_rejects_negative_epsilon():
    with pytest.raises(ValueError):
        epsilon_greedy([0.2, 0.8], epsilon=-0.1)


def test_epsilon_greedy_reproducibility():
    values = [0.2, 0.8, 0.4]

    result_1 = epsilon_greedy(
        values, epsilon=0.5, rng=random.Random(42)
    )

    result_2 = epsilon_greedy(
        values, epsilon=0.5, rng=random.Random(42)
    )

    assert result_1 == result_2
