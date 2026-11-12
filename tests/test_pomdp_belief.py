import pytest

from environments.pomdp import POMDP
from algorithms.pomdp import (
    predict_belief,
    bayesian_belief_update,
    normalise_belief,
    update_belief,
)


def make_simple_pomdp():
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


def test_normalise_belief():
    belief = {"A": 0.2, "B": 0.3, "C": 0.5}
    result = normalise_belief(belief)
    assert abs(result["A"] - 0.2) < 1e-9
    assert abs(result["B"] - 0.3) < 1e-9
    assert abs(result["C"] - 0.5) < 1e-9
    assert abs(sum(result.values()) - 1.0) < 1e-9


def test_normalise_belief_rescales():
    belief = {"A": 1.0, "B": 2.0, "C": 3.0}
    result = normalise_belief(belief)
    assert abs(result["A"] - 1.0 / 6.0) < 1e-9
    assert abs(result["B"] - 2.0 / 6.0) < 1e-9
    assert abs(result["C"] - 3.0 / 6.0) < 1e-9
    assert abs(sum(result.values()) - 1.0) < 1e-9


def test_normalise_belief_rejects_all_zero():
    with pytest.raises(ValueError):
        normalise_belief({"A": 0.0, "B": 0.0})


def test_normalise_belief_rejects_negative():
    with pytest.raises(ValueError):
        normalise_belief({"A": -0.1, "B": 1.1})


def test_predict_belief_stay_keeps_same():
    pomdp = make_simple_pomdp()
    belief = {"A": 0.5, "B": 0.3, "C": 0.2}
    predicted = predict_belief(pomdp, belief, "stay")
    # stay: A->A, B->B, C->C with prob 1.0
    assert abs(predicted["A"] - 0.5) < 1e-9
    assert abs(predicted["B"] - 0.3) < 1e-9
    assert abs(predicted["C"] - 0.2) < 1e-9


def test_predict_belief_move_spreads():
    pomdp = make_simple_pomdp()
    belief = {"A": 1.0, "B": 0.0, "C": 0.0}  # certain in A
    predicted = predict_belief(pomdp, belief, "move")
    # move from A: B->0.8, C->0.2
    assert abs(predicted["A"] - 0.0) < 1e-9
    assert abs(predicted["B"] - 0.8) < 1e-9
    assert abs(predicted["C"] - 0.2) < 1e-9
    assert abs(sum(predicted.values()) - 1.0) < 1e-9


def test_predict_belief_invalid_action():
    pomdp = make_simple_pomdp()
    belief = {"A": 1.0, "B": 0.0, "C": 0.0}
    with pytest.raises(ValueError):
        predict_belief(pomdp, belief, "invalid")


def test_bayesian_update_concentrates_on_likely_state():
    pomdp = make_simple_pomdp()
    # Start uniform, predict after "stay" (stays same), then observe obs_x
    belief = {"A": 1.0 / 3.0, "B": 1.0 / 3.0, "C": 1.0 / 3.0}
    predicted = predict_belief(pomdp, belief, "stay")
    updated = bayesian_belief_update(pomdp, predicted, "obs_x", "stay")
    # obs_x is most likely in A (0.8) and C (0.7), less in B (0.2)
    # After update, A and C should have higher probability than B
    assert updated["A"] > updated["B"]
    assert updated["C"] > updated["B"]
    assert abs(sum(updated.values()) - 1.0) < 1e-9


def test_bayesian_update_invalid_action():
    pomdp = make_simple_pomdp()
    belief = {"A": 1.0 / 3.0, "B": 1.0 / 3.0, "C": 1.0 / 3.0}
    with pytest.raises(ValueError):
        bayesian_belief_update(pomdp, belief, "obs_x", "invalid")


def test_bayesian_update_invalid_observation():
    pomdp = make_simple_pomdp()
    belief = {"A": 1.0 / 3.0, "B": 1.0 / 3.0, "C": 1.0 / 3.0}
    with pytest.raises(ValueError):
        bayesian_belief_update(pomdp, belief, "invalid_obs", "stay")


def test_bayesian_update_zero_likelihood():
    states = ["A", "B"]
    actions = ["stay"]
    observations = ["obs_x", "obs_y"]
    # obs_x has 0 probability for all states, obs_y has 1.0 for all states
    # This sums to 1.0 per (state, action) so POMDP construction succeeds
    transition_probs = {("A", "A", "stay"): 1.0, ("B", "B", "stay"): 1.0}
    observation_probs = {
        ("obs_x", "A", "stay"): 0.0,
        ("obs_y", "A", "stay"): 1.0,
        ("obs_x", "B", "stay"): 0.0,
        ("obs_y", "B", "stay"): 1.0,
    }
    rewards = {("A", "stay", "A"): 0.0, ("B", "stay", "B"): 0.0}
    pomdp2 = POMDP(states, actions, observations, transition_probs, observation_probs, rewards, "A")
    belief = {"A": 0.5, "B": 0.5}
    # Observing obs_x has zero likelihood under all states -> should raise
    with pytest.raises(ValueError):
        bayesian_belief_update(pomdp2, belief, "obs_x", "stay")


def test_update_belief_combined_steps():
    pomdp = make_simple_pomdp()
    belief = {"A": 1.0, "B": 0.0, "C": 0.0}  # certain in A
    updated = update_belief(pomdp, belief, "move", "obs_x")
    # After move from A: predicted B=0.8, C=0.2
    # After obs_x: B likelihood 0.2, C likelihood 0.7
    # unnormalised: B=0.8*0.2=0.16, C=0.2*0.7=0.14
    # normalised: B=0.16/0.30=0.533, C=0.14/0.30=0.467
    assert abs(updated["A"] - 0.0) < 1e-9
    assert abs(updated["B"] - 0.16 / 0.30) < 1e-6
    assert abs(updated["C"] - 0.14 / 0.30) < 1e-6
    assert abs(sum(updated.values()) - 1.0) < 1e-9


def test_update_belief_preserves_validity():
    pomdp = make_simple_pomdp()
    belief = {"A": 0.4, "B": 0.3, "C": 0.3}
    for action in pomdp.actions:
        for obs in pomdp.observations:
            updated = update_belief(pomdp, belief, action, obs)
            assert all(p >= 0 for p in updated.values())
            assert abs(sum(updated.values()) - 1.0) < 1e-9


def test_update_belief_invalid_action():
    pomdp = make_simple_pomdp()
    belief = {"A": 1.0 / 3.0, "B": 1.0 / 3.0, "C": 1.0 / 3.0}
    with pytest.raises(ValueError):
        update_belief(pomdp, belief, "invalid", "obs_x")


def test_belief_remains_valid_after_multiple_updates():
    pomdp = make_simple_pomdp()
    belief = {"A": 1.0 / 3.0, "B": 1.0 / 3.0, "C": 1.0 / 3.0}
    sequence = [("move", "obs_x"), ("stay", "obs_y"), ("move", "obs_x")]
    for action, obs in sequence:
        belief = update_belief(pomdp, belief, action, obs)
        assert all(p >= 0 for p in belief.values())
        assert abs(sum(belief.values()) - 1.0) < 1e-9