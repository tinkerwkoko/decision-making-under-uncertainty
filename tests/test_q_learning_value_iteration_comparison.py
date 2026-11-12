import pytest

from environments.gridworld import GridWorld
from experiments.compare_q_learning_value_iteration import (
    evaluate_q_learning_policy,
    compare_q_learning_value_iteration,
)


def test_q_learning_policy_evaluation_returns_expected_keys():
    env = GridWorld()
    result = evaluate_q_learning_policy(env, episodes=50, epsilon=0.1, seed=42, n_eval_episodes=20)

    assert "success" in result
    assert "success_rate" in result
    assert "avg_steps" in result
    assert "avg_reward" in result
    assert "episodes" in result
    assert "epsilon" in result


def test_q_learning_policy_success_rate_range():
    env = GridWorld()
    result = evaluate_q_learning_policy(env, episodes=50, epsilon=0.1, seed=42, n_eval_episodes=20)
    assert 0.0 <= result["success_rate"] <= 100.0


def test_comparison_returns_both_methods():
    env = GridWorld()
    comparison = compare_q_learning_value_iteration(
        env, n_runs=3, q_episodes=50, epsilon=0.1, base_seed=42, n_eval_episodes=20
    )
    assert "q_learning" in comparison
    assert "value_iteration" in comparison
    assert "info_requirements" in comparison


def test_comparison_has_info_requirements():
    env = GridWorld()
    comparison = compare_q_learning_value_iteration(
        env, n_runs=3, q_episodes=50, epsilon=0.1, base_seed=42, n_eval_episodes=20
    )
    assert "q_learning" in comparison["info_requirements"]
    assert "value_iteration" in comparison["info_requirements"]
    assert "model-free" in comparison["info_requirements"]["q_learning"].lower()
    assert "model-based" in comparison["info_requirements"]["value_iteration"].lower()


def test_comparison_q_learning_has_aggregation():
    env = GridWorld()
    comparison = compare_q_learning_value_iteration(
        env, n_runs=3, q_episodes=50, epsilon=0.1, base_seed=42, n_eval_episodes=20
    )
    ql = comparison["q_learning"]
    assert ql["n_runs"] == 3
    assert "mean_success_rate" in ql
    assert "std_success_rate" in ql
    assert "mean_avg_steps" in ql
    assert "mean_avg_reward" in ql


def test_comparison_vi_has_convergence():
    env = GridWorld()
    comparison = compare_q_learning_value_iteration(
        env, n_runs=3, q_episodes=50, epsilon=0.1, base_seed=42, n_eval_episodes=20
    )
    vi = comparison["value_iteration"]
    assert vi["converged"] is True
    assert vi["success"] is True


def test_comparison_rejects_invalid_n_runs():
    env = GridWorld()
    with pytest.raises(ValueError):
        compare_q_learning_value_iteration(env, n_runs=0)


def test_comparison_rejects_invalid_gamma():
    env = GridWorld()
    with pytest.raises(ValueError):
        compare_q_learning_value_iteration(env, gamma=1.5)