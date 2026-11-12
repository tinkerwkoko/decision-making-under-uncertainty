import pytest

from environments.gridworld import GridWorld
from experiments.value_iteration_evaluation import evaluate_value_iteration


def test_vi_evaluation_returns_expected_keys():
    env = GridWorld()
    result = evaluate_value_iteration(env, gamma=0.9, max_steps=100)

    assert "values" in result
    assert "iterations" in result
    assert "converged" in result
    assert "policy" in result
    assert "path" in result
    assert "success" in result
    assert "steps" in result
    assert "total_reward" in result
    assert "gamma" in result
    assert "theta" in result


def test_vi_reaches_goal():
    env = GridWorld()
    result = evaluate_value_iteration(env, gamma=0.9, max_steps=100)
    assert result["success"] is True
    assert result["path"][0] == env.start
    assert result["path"][-1] == env.goal


def test_vi_converges():
    env = GridWorld()
    result = evaluate_value_iteration(env, gamma=0.9, max_iterations=10000)
    assert result["converged"] is True


def test_vi_path_is_optimal_length():
    env = GridWorld()
    result = evaluate_value_iteration(env, gamma=0.9, max_steps=100)
    # From (0,0) to (3,3) in a 4x4 grid with no obstacles: 6 steps minimum
    assert result["steps"] == 6


def test_vi_rejects_invalid_gamma():
    env = GridWorld()
    with pytest.raises(ValueError):
        evaluate_value_iteration(env, gamma=1.5)
    with pytest.raises(ValueError):
        evaluate_value_iteration(env, gamma=-0.1)


def test_vi_rejects_invalid_theta():
    env = GridWorld()
    with pytest.raises(ValueError):
        evaluate_value_iteration(env, theta=0.0)


def test_vi_rejects_invalid_max_steps():
    env = GridWorld()
    with pytest.raises(ValueError):
        evaluate_value_iteration(env, max_steps=0)


def test_vi_policy_has_all_states():
    env = GridWorld()
    result = evaluate_value_iteration(env, gamma=0.9, max_steps=100)
    assert len(result["policy"]) == 16
    assert result["policy"][env.goal] is None


def test_vi_values_goal_is_zero():
    env = GridWorld()
    result = evaluate_value_iteration(env, gamma=0.9, max_steps=100)
    assert result["values"][env.goal] == 0.0