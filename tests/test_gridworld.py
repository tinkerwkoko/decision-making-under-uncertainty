
import pytest

from environments.gridworld import GridWorld


def test_default_initialization():
    env = GridWorld()

    assert env.size == 4
    assert env.start == (0, 0)
    assert env.goal == (3, 3)
    assert env.step_reward == -1
    assert env.agent_position == (0, 0)
    assert env.actions == ["up", "down", "left", "right"]
    assert env.done is False


@pytest.mark.parametrize(
    "action, expected",
    [
        ("up", (0, 2)),
        ("down", (2, 2)),
        ("left", (1, 1)),
        ("right", (1, 3)),
    ],
)
def test_movement(action, expected):
    env = GridWorld()

    result = env.get_next_position((1, 2), action)

    assert result == expected


@pytest.mark.parametrize(
    "state, action, expected",
    [
        ((0, 0), "up", (0, 0)),
        ((0, 0), "left", (0, 0)),
        ((3, 3), "down", (3, 3)),
        ((3, 3), "right", (3, 3)),
    ],
)
def test_boundary_restrictions(state, action, expected):
    env = GridWorld()

    result = env.get_next_position(state, action)

    assert result == expected


def test_invalid_action():
    env = GridWorld()

    with pytest.raises(ValueError):
        env.step("jump")


def test_step_returns_correct_reward():
    env = GridWorld()

    next_state, reward, done = env.step("right")

    assert next_state == (0, 1)
    assert reward == -1
    assert done is False


def test_goal_detection():
    env = GridWorld()
    env.agent_position = (3, 2)

    next_state, reward, done = env.step("right")

    assert next_state == (3, 3)
    assert reward == -1
    assert done is True
    assert env.done is True


def test_cannot_move_after_episode_ends():
    env = GridWorld()
    env.agent_position = (3, 2)

    env.step("right")

    with pytest.raises(RuntimeError):
        env.step("left")


def test_reset_starts_new_episode():
    env = GridWorld()
    env.agent_position = (3, 2)

    env.step("right")

    state = env.reset()

    assert state == (0, 0)
    assert env.agent_position == (0, 0)
    assert env.done is False
