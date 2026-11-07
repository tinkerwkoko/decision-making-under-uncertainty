import pytest
import random

from algorithms.bandit_algorithms import (
    initialize_beta_parameters,
    update_beta_parameters,
    thompson_sampling,
)


def test_initialize_beta_parameters():
    alpha, beta = initialize_beta_parameters(4)

    assert alpha == [1.0, 1.0, 1.0, 1.0]
    assert beta == [1.0, 1.0, 1.0, 1.0]


def test_initialize_rejects_zero_arms():
    with pytest.raises(ValueError):
        initialize_beta_parameters(0)


def test_initialize_rejects_negative_arms():
    with pytest.raises(ValueError):
        initialize_beta_parameters(-2)


def test_success_increases_alpha():
    alpha, beta = initialize_beta_parameters(3)

    update_beta_parameters(alpha, beta, 1, 1)

    assert alpha == [1.0, 2.0, 1.0]
    assert beta == [1.0, 1.0, 1.0]


def test_failure_increases_beta():
    alpha, beta = initialize_beta_parameters(3)

    update_beta_parameters(alpha, beta, 1, 0)

    assert alpha == [1.0, 1.0, 1.0]
    assert beta == [1.0, 2.0, 1.0]


def test_multiple_observations_update_correctly():
    alpha, beta = initialize_beta_parameters(2)

    update_beta_parameters(alpha, beta, 0, 1)
    update_beta_parameters(alpha, beta, 0, 1)
    update_beta_parameters(alpha, beta, 0, 0)

    assert alpha[0] == 3.0
    assert beta[0] == 2.0


def test_update_rejects_invalid_arm():
    alpha, beta = initialize_beta_parameters(2)

    with pytest.raises(ValueError):
        update_beta_parameters(alpha, beta, 2, 1)


def test_update_rejects_invalid_reward():
    alpha, beta = initialize_beta_parameters(2)

    with pytest.raises(ValueError):
        update_beta_parameters(alpha, beta, 0, 2)


def test_thompson_sampling_returns_valid_arm():
    alpha, beta = initialize_beta_parameters(4)

    selected_arm = thompson_sampling(
        alpha,
        beta,
        rng=random.Random(42),
    )

    assert selected_arm in range(4)


def test_thompson_sampling_prefers_strongly_supported_arm():
    alpha = [1.0, 100.0]
    beta = [100.0, 1.0]

    selected_arm = thompson_sampling(
        alpha,
        beta,
        rng=random.Random(42),
    )

    assert selected_arm == 1


def test_thompson_sampling_rejects_empty_parameters():
    with pytest.raises(ValueError):
        thompson_sampling([], [])


def test_thompson_sampling_rejects_mismatched_parameters():
    with pytest.raises(ValueError):
        thompson_sampling([1.0, 2.0], [1.0])


def test_thompson_sampling_rejects_nonpositive_parameters():
    with pytest.raises(ValueError):
        thompson_sampling([0.0], [1.0])