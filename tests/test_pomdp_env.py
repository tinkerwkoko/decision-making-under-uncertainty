import pytest

from environments.pomdp import POMDP


def make_simple_pomdp():
    """Create a simple 3-state, 2-action, 2-observation POMDP for testing."""
    states = ["A", "B", "C"]
    actions = ["stay", "move"]
    observations = ["obs_x", "obs_y"]

    transition_probs = {}
    for s in states:
        transition_probs[(s, s, "stay")] = 1.0
    transition_probs[("B", "A", "move")] = 0.8
    transition_probs[("C", "A", "move")] = 0.2
    transition_probs[("A", "B", "move")] = 0.7
    transition_probs[("C", "B", "move")] = 0.3
    transition_probs[("A", "C", "move")] = 0.4
    transition_probs[("B", "C", "move")] = 0.6

    observation_probs = {}
    for s in states:
        for a in actions:
            observation_probs[("obs_x", s, a)] = 0.0
            observation_probs[("obs_y", s, a)] = 0.0
    observation_probs[("obs_x", "A", "stay")] = 0.8
    observation_probs[("obs_y", "A", "stay")] = 0.2
    observation_probs[("obs_x", "A", "move")] = 0.8
    observation_probs[("obs_y", "A", "move")] = 0.2
    observation_probs[("obs_x", "B", "stay")] = 0.2
    observation_probs[("obs_y", "B", "stay")] = 0.8
    observation_probs[("obs_x", "B", "move")] = 0.2
    observation_probs[("obs_y", "B", "move")] = 0.8
    observation_probs[("obs_x", "C", "stay")] = 0.7
    observation_probs[("obs_y", "C", "stay")] = 0.3
    observation_probs[("obs_x", "C", "move")] = 0.7
    observation_probs[("obs_y", "C", "move")] = 0.3

    rewards = {}
    for s in states:
        for a in actions:
            for s2 in states:
                rewards[(s, a, s2)] = 0.0
    rewards[("C", "stay", "C")] = 1.0

    return POMDP(
        states=states,
        actions=actions,
        observations=observations,
        transition_probs=transition_probs,
        observation_probs=observation_probs,
        rewards=rewards,
        start_state="A",
        seed=42,
    )


def test_pomdp_initialization():
    pomdp = make_simple_pomdp()
    assert pomdp.states == ["A", "B", "C"]
    assert pomdp.actions == ["stay", "move"]
    assert pomdp.observations == ["obs_x", "obs_y"]
    assert pomdp.start_state == "A"
    assert pomdp.true_state == "A"


def test_pomdp_initial_belief_uniform():
    pomdp = make_simple_pomdp()
    belief = pomdp.belief
    assert abs(belief["A"] - 1.0 / 3.0) < 1e-9
    assert abs(belief["B"] - 1.0 / 3.0) < 1e-9
    assert abs(belief["C"] - 1.0 / 3.0) < 1e-9


def test_pomdp_start_belief_custom():
    pomdp = POMDP(
        states=["A", "B"],
        actions=["stay"],
        observations=["obs_x"],
        transition_probs={("A", "A", "stay"): 1.0, ("B", "B", "stay"): 1.0},
        observation_probs={("obs_x", "A", "stay"): 1.0, ("obs_x", "B", "stay"): 1.0},
        rewards={},
        start_state="A",
        start_belief={"A": 0.7, "B": 0.3},
    )
    assert pomdp.belief == {"A": 0.7, "B": 0.3}


def test_pomdp_transition_sums_to_one():
    pomdp = make_simple_pomdp()
    for s in pomdp.states:
        for a in pomdp.actions:
            total = sum(
                pomdp.transition_probs.get((s2, s, a), 0.0) for s2 in pomdp.states
            )
            assert abs(total - 1.0) < 1e-9


def test_pomdp_observation_sums_to_one():
    pomdp = make_simple_pomdp()
    for s in pomdp.states:
        for a in pomdp.actions:
            total = sum(
                pomdp.observation_probs.get((o, s, a), 0.0) for o in pomdp.observations
            )
            assert abs(total - 1.0) < 1e-9


def test_pomdp_step_returns_observation_reward_done():
    pomdp = make_simple_pomdp()
    obs, rew, done = pomdp.step("stay")
    assert obs in pomdp.observations
    assert isinstance(rew, (int, float))
    assert done is False


def test_pomdp_step_invalid_action():
    pomdp = make_simple_pomdp()
    with pytest.raises(ValueError):
        pomdp.step("invalid_action")


def test_pomdp_reset():
    pomdp = make_simple_pomdp()
    pomdp.step("move")
    obs = pomdp.reset(seed=42)
    assert pomdp.true_state == "A"
    assert obs in pomdp.observations


def test_pomdp_belief_property_returns_copy():
    pomdp = make_simple_pomdp()
    belief1 = pomdp.belief
    belief2 = pomdp.belief
    assert belief1 == belief2
    assert belief1 is not belief2


def test_pomdp_set_belief_validates_sum_to_one():
    pomdp = make_simple_pomdp()
    with pytest.raises(ValueError):
        pomdp.set_belief({"A": 0.5, "B": 0.6, "C": 0.0})


def test_pomdp_invalid_states_raises():
    with pytest.raises(ValueError):
        POMDP(
            states=[],
            actions=["a"],
            observations=["o"],
            transition_probs={},
            observation_probs={},
            rewards={},
            start_state="A",
        )


def test_pomdp_invalid_start_state_raises():
    with pytest.raises(ValueError):
        POMDP(
            states=["A", "B"],
            actions=["a"],
            observations=["o"],
            transition_probs={},
            observation_probs={},
            rewards={},
            start_state="C",
        )