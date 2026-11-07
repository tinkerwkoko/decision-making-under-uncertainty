import pytest

from environments.bandit import BernoulliBandit


def test_bandit_initialization():
    bandit = BernoulliBandit([0.2, 0.5, 0.7, 0.9])

    assert bandit.n_arms == 4


def test_bandit_identifies_best_arm():
    bandit = BernoulliBandit([0.2, 0.5, 0.7, 0.9])

    assert bandit.best_arm == 3
    assert bandit.best_probability == 0.9


def test_bandit_returns_binary_reward():
    bandit = BernoulliBandit([0.2, 0.5, 0.7, 0.9], seed=42)

    reward = bandit.pull(3)

    assert reward in [0, 1]


def test_arm_with_zero_probability_never_rewards():
    bandit = BernoulliBandit([0.0], seed=42)

    rewards = [bandit.pull(0) for _ in range(100)]

    assert all(reward == 0 for reward in rewards)


def test_arm_with_one_probability_always_rewards():
    bandit = BernoulliBandit([1.0], seed=42)

    rewards = [bandit.pull(0) for _ in range(100)]

    assert all(reward == 1 for reward in rewards)


def test_invalid_arm_index():
    bandit = BernoulliBandit([0.2, 0.5, 0.9])

    with pytest.raises(ValueError):
        bandit.pull(3)


def test_negative_arm_index():
    bandit = BernoulliBandit([0.2, 0.5, 0.9])

    with pytest.raises(ValueError):
        bandit.pull(-1)


def test_invalid_probability():
    with pytest.raises(ValueError):
        BernoulliBandit([0.2, 1.5, 0.9])


def test_negative_probability():
    with pytest.raises(ValueError):
        BernoulliBandit([0.2, -0.1, 0.9])


def test_empty_probabilities():
    with pytest.raises(ValueError):
        BernoulliBandit([])


def test_reproducible_rewards():
    bandit_1 = BernoulliBandit([0.2, 0.5, 0.9], seed=42)
    bandit_2 = BernoulliBandit([0.2, 0.5, 0.9], seed=42)

    rewards_1 = [bandit_1.pull(2) for _ in range(20)]
    rewards_2 = [bandit_2.pull(2) for _ in range(20)]

    assert rewards_1 == rewards_2