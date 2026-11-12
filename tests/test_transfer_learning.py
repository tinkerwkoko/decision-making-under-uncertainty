import pytest

from environments.gridworld import GridWorld
from experiments.transfer_learning import (
    run_transfer_learning_experiment,
    _make_source_gridworld,
    _make_target_gridworld,
    _map_source_q_table_to_target,
    _evaluate_q_table_on_target,
)
from algorithms.q_learning import train_q_learning, initialize_q_table


def test_source_gridworld():
    env = _make_source_gridworld()
    assert env.size == 4
    assert env.start == (0, 0)
    assert env.goal == (3, 3)


def test_target_gridworld():
    env = _make_target_gridworld()
    assert env.size == 4
    assert env.start == (0, 0)
    assert env.goal == (3, 2)


def test_q_table_transfer_maps_states():
    source_env = _make_source_gridworld()
    source_q = initialize_q_table(source_env)
    # Set a known value
    source_q[(0, 0)]["right"] = 5.0

    target_env = _make_target_gridworld()
    target_q = _map_source_q_table_to_target(source_q, target_env)

    # All target states should exist
    assert len(target_q) == 16
    # Transferred value should be present
    assert target_q[(0, 0)]["right"] == 5.0
    # Non-transferred values should be zero
    assert target_q[(0, 0)]["up"] == 0.0


def test_q_table_transfer_preserves_all_actions():
    source_env = _make_source_gridworld()
    source_q = initialize_q_table(source_env)
    target_env = _make_target_gridworld()
    target_q = _map_source_q_table_to_target(source_q, target_env)

    for state in target_q:
        assert set(target_q[state].keys()) == set(target_env.actions)


def test_evaluate_q_table_returns_expected_keys():
    env = GridWorld()
    q_table = initialize_q_table(env)
    result = _evaluate_q_table_on_target(q_table, env, n_episodes=10, seed=42)

    assert "success_rate" in result
    assert "avg_reward" in result
    assert "avg_steps" in result
    assert "successes" in result
    assert 0.0 <= result["success_rate"] <= 100.0


def test_transfer_experiment_returns_expected_keys():
    result = run_transfer_learning_experiment(
        n_runs=2, source_episodes=100, target_episodes=100, base_seed=42, n_eval_episodes=20
    )

    assert "n_runs" in result
    assert "source_episodes" in result
    assert "target_episodes" in result
    assert "base_seed" in result
    assert "source_q_table" in result
    assert "transfer" in result
    assert "from_scratch" in result
    assert "transfer_minus_from_scratch" in result


def test_transfer_experiment_has_transfer_and_from_scratch():
    result = run_transfer_learning_experiment(
        n_runs=2, source_episodes=100, target_episodes=100, base_seed=42, n_eval_episodes=20
    )

    transfer = result["transfer"]
    from_scratch = result["from_scratch"]

    for method in (transfer, from_scratch):
        assert "mean_success_rate" in method
        assert "std_success_rate" in method
        assert "mean_avg_reward" in method
        assert "std_avg_reward" in method
        assert "mean_avg_steps" in method
        assert "std_avg_steps" in method
        assert "success_rates" in method
        assert "avg_rewards" in method
        assert "avg_steps" in method
        assert "train_rewards_per_run" in method


def test_transfer_experiment_reproducible():
    result1 = run_transfer_learning_experiment(
        n_runs=2, source_episodes=100, target_episodes=100, base_seed=42, n_eval_episodes=20
    )
    result2 = run_transfer_learning_experiment(
        n_runs=2, source_episodes=100, target_episodes=100, base_seed=42, n_eval_episodes=20
    )

    assert result1["transfer"]["mean_success_rate"] == result2["transfer"]["mean_success_rate"]
    assert result1["from_scratch"]["mean_success_rate"] == result2["from_scratch"]["mean_success_rate"]


def test_transfer_experiment_rejects_invalid_n_runs():
    with pytest.raises(ValueError):
        run_transfer_learning_experiment(n_runs=0)


def test_transfer_experiment_rejects_invalid_base_seed():
    with pytest.raises(ValueError):
        run_transfer_learning_experiment(base_seed="not_int")


def test_transfer_experiment_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        run_transfer_learning_experiment(n_runs=1, alpha=1.5)


def test_transfer_experiment_rejects_invalid_gamma():
    with pytest.raises(ValueError):
        run_transfer_learning_experiment(n_runs=1, gamma=-0.1)


def test_transfer_experiment_rejects_invalid_episodes():
    with pytest.raises(ValueError):
        run_transfer_learning_experiment(n_runs=1, source_episodes=0)
    with pytest.raises(ValueError):
        run_transfer_learning_experiment(n_runs=1, target_episodes=0)


def test_transfer_diff_key_exists():
    result = run_transfer_learning_experiment(
        n_runs=2, source_episodes=100, target_episodes=100, base_seed=42, n_eval_episodes=20
    )
    diff = result["transfer_minus_from_scratch"]
    assert "success_rate_diff" in diff
    assert "avg_reward_diff" in diff
    assert "avg_steps_diff" in diff


def test_transfer_source_q_table_not_none():
    result = run_transfer_learning_experiment(
        n_runs=2, source_episodes=100, target_episodes=100, base_seed=42, n_eval_episodes=20
    )
    assert result["source_q_table"] is not None
    assert len(result["source_q_table"]) == 16