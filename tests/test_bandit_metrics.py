import pytest

from experiments.bandit_metrics import calculate_cumulative_regret


def test_calculate_cumulative_regret():
    probabilities = [0.2, 0.5, 0.9]
    selected_arms = [0, 1, 2, 1, 0]

    result = calculate_cumulative_regret(probabilities, selected_arms)

    assert result == pytest.approx([0.7, 1.1, 1.1, 1.5, 2.2])


def test_optimal_arm_has_zero_regret():
    probabilities = [0.2, 0.5, 0.9]
    selected_arms = [2, 2, 2]

    result = calculate_cumulative_regret(probabilities, selected_arms)

    assert result == [0.0, 0.0, 0.0]


def test_empty_action_history():
    probabilities = [0.2, 0.5, 0.9]
    selected_arms = []

    result = calculate_cumulative_regret(probabilities, selected_arms)

    assert result == []


def test_empty_probabilities_raises_error():
    with pytest.raises(ValueError):
        calculate_cumulative_regret([], [0, 1])

def test_probability_above_one_raises_error():
    with pytest.raises(ValueError):
        calculate_cumulative_regret([0.2, 1.5], [0])


def test_negative_probability_raises_error():
    with pytest.raises(ValueError):
        calculate_cumulative_regret([-0.2, 0.5], [0])


def test_invalid_arm_index_raises_error():
    with pytest.raises(ValueError):
        calculate_cumulative_regret([0.2, 0.5], [2])


def test_negative_arm_index_raises_error():
    with pytest.raises(ValueError):
        calculate_cumulative_regret([0.2, 0.5], [-1])


def test_boolean_arm_index_raises_error():
    with pytest.raises(ValueError):
        calculate_cumulative_regret([0.2, 0.5], [True])