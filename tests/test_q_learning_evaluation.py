import pytest
import random

from environments.gridworld import GridWorld
from experiments.q_learning_evaluation import (
    run_single_q_learning_evaluation,
    run_q_learning_evaluation,
    evaluate_q_learning_across_epsilons,
)


def test_single_evaluation_returns_expected_keys():
    env = GridWorld()
    result = run_single_q_learning_evaluation(
        env, episodes=50, epsilon=0.1, seed=42, n_eval_episodes=20
    )

    assert "q_table" in result
    assert "train_rewards" in result
    assert "epsilon_history" in result
    assert "eval_success_rate" in result
    assert "eval_avg_reward" in result
    assert "eval_avg_steps" in result
    assert "eval_successes" in result


def test_single_evaluation_reproducible():
    env1 = GridWorld()
    env2 = GridWorld()
    r1 = run_single_q_learning_evaluation(env1, episodes=50, epsilon=0.1, seed=42, n_eval_episodes=20)
    r2 = run_single_q_learning_evaluation(env2, episodes=50, epsilon=0.1, seed=42, n_eval_episodes=20)

    assert r1["train_rewards"] == r2["train_rewards"]
    assert r1["eval_success_rate"] == r2["eval_success_rate"]
    assert r1["eval_avg_reward"] == r2["eval_avg_reward"]


def test_success_rate_between_zero_and_hundred():
    env = GridWorld()
    result = run_single_q_learning_evaluation(env, episodes=50, epsilon=0.1, seed=42, n_eval_episodes=20)
    assert 0.0 <= result["eval_success_rate"] <= 100.0


def test_eval_avg_steps_positive():
    env = GridWorld()
    result = run_single_q_learning_evaluation(env, episodes=50, epsilon=0.1, seed=42, n_eval_episodes=20)
    assert result["eval_avg_steps"] > 0


def test_q_learning_across_epsilons():
    env = GridWorld()
    results = evaluate_q_learning_across_epsilons(
        env, n_runs=3, episodes=50, epsilons=(0.1, 0.3), base_seed=42, n_eval_episodes=20
    )
    assert 0.1 in results
    assert 0.3 in results
    assert "mean_success_rate" in results[0.1]
    assert "mean_success_rate" in results[0.3]


def test_q_learning_across_epsilons_reproducible():
    env1 = GridWorld()
    env2 = GridWorld()
    r1 = evaluate_q_learning_across_epsilons(
        env1, n_runs=3, episodes=50, epsilons=(0.1, 0.3), base_seed=42, n_eval_episodes=20
    )
    r2 = evaluate_q_learning_across_epsilons(
        env2, n_runs=3, episodes=50, epsilons=(0.1, 0.3), base_seed=42, n_eval_episodes=20
    )
    assert r1[0.1]["mean_success_rate"] == r2[0.1]["mean_success_rate"]
    assert r1[0.3]["mean_success_rate"] == r2[0.3]["mean_success_rate"]


def test_run_q_learning_evaluation_aggregation():
    env = GridWorld()
    result = run_q_learning_evaluation(
        env, n_runs=3, episodes=50, epsilon=0.1, base_seed=42, n_eval_episodes=20
    )
    assert result["n_runs"] == 3
    assert result["epsilon"] == 0.1
    assert len(result["success_rates"]) == 3
    assert len(result["avg_rewards"]) == 3
    assert len(result["avg_steps"]) == 3
    assert 0.0 <= result["mean_success_rate"] <= 100.0
    assert result["std_success_rate"] >= 0.0


def test_run_q_learning_evaluation_rejects_invalid_n_runs():
    env = GridWorld()
    with pytest.raises(ValueError):
        run_q_learning_evaluation(env, n_runs=0, episodes=50, epsilon=0.1)


def test_run_q_learning_evaluation_rejects_invalid_base_seed():
    env = GridWorld()
    with pytest.raises(ValueError):
        run_q_learning_evaluation(env, n_runs=3, episodes=50, epsilon=0.1, base_seed="not_int")


def test_single_evaluation_rejects_invalid_episodes():
    env = GridWorld()
    with pytest.raises(ValueError):
        run_single_q_learning_evaluation(env, episodes=0, epsilon=0.1)