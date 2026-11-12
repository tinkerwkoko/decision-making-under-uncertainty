"""
POMDP Belief-State Algorithms.

Implements belief prediction, Bayesian belief updates, and normalisation
for a POMDP. These are the core mathematical operations that allow an
agent to reason under partial observability.

Mathematical model:

    Belief prediction (prior update):
        b'(s') = sum over s of T(s'|s,a) * b(s)

    Bayesian belief update (posterior after observation o):
        b''(s') = O(o|s',a) * b'(s') / normalisation
        where normalisation = sum over s'' of O(o|s'',a) * b'(s'')

The belief distribution b(s) represents the agent's probability that the
true hidden state is s. It is always a valid probability distribution
(non-negative, sums to 1).

This is an educational implementation: it is NOT a general-purpose
optimal POMDP solver (solving a POMDP optimally is PSPACE-complete).
"""

from environments.pomdp import POMDP


def predict_belief(pomdp, belief, action):
    """Compute the predicted belief after taking an action (before observation).

    This is the "prediction" step of Bayesian filtering. It propagates
    the current belief through the transition model:

        b'(s') = sum over s of T(s'|s,a) * b(s)

    Args:
        pomdp: A POMDP environment providing transition probabilities.
        belief: dict state -> probability (current belief, must sum to 1).
        action: The action being taken.

    Returns:
        dict state -> probability: the predicted belief b'(s').

    Raises:
        ValueError: If belief is invalid or action is unknown.
    """
    if action not in pomdp.actions:
        raise ValueError(f"Invalid action: {action}")

    states = pomdp.states
    predicted = {}

    for next_s in states:
        total = 0.0
        for s in states:
            trans_prob = pomdp.transition_probs.get((next_s, s, action), 0.0)
            total += trans_prob * belief[s]
        predicted[next_s] = total

    return normalise_belief(predicted)


def bayesian_belief_update(pomdp, predicted_belief, observation, action):
    """Apply a Bayesian belief update after receiving an observation.

    This is the "correction" step of Bayesian filtering. It combines the
    predicted belief with the observation likelihood:

        b''(s') ∝ O(o|s',a) * b'(s')

    followed by normalisation so the result is a valid distribution:

        b''(s') = O(o|s',a) * b'(s') / sum over s'' of O(o|s'',a) * b'(s'')

    Args:
        pomdp: A POMDP environment providing observation probabilities.
        predicted_belief: dict state -> probability (belief after prediction).
        observation: The observation received.
        action: The action that preceded the observation.

    Returns:
        dict state -> probability: the updated belief b''(s').

    Raises:
        ValueError: If observation or action is unknown, or if the
            observation has zero likelihood under all states (no update
            possible).
    """
    if action not in pomdp.actions:
        raise ValueError(f"Invalid action: {action}")
    if observation not in pomdp.observations:
        raise ValueError(f"Invalid observation: {observation}")

    states = pomdp.states
    unnormalised = {}
    total = 0.0

    for next_s in states:
        obs_prob = pomdp.observation_probs.get(
            (observation, next_s, action), 0.0
        )
        value = obs_prob * predicted_belief[next_s]
        unnormalised[next_s] = value
        total += value

    if total <= 0:
        raise ValueError(
            f"Observation '{observation}' has zero likelihood under all "
            f"states after action '{action}'. Bayesian update cannot proceed."
        )

    return {s: unnormalised[s] / total for s in states}


def normalise_belief(belief):
    """Normalise a belief distribution so it sums to 1.

    Args:
        belief: dict state -> probability (may not sum to 1).

    Returns:
        dict state -> probability: the normalised distribution.

    Raises:
        ValueError: If all probabilities are zero or any are negative.
    """
    total = sum(belief.values())
    if total <= 0:
        raise ValueError(
            "Cannot normalise a belief distribution with total probability <= 0."
        )
    if any(p < 0 for p in belief.values()):
        raise ValueError("Belief probabilities must be non-negative.")

    return {s: p / total for s, p in belief.items()}


def update_belief(pomdp, belief, action, observation):
    """Perform a complete belief update: prediction then Bayesian correction.

    This combines predict_belief and bayesian_belief_update into one step:

        1. Predict:  b'(s')  = sum over s of T(s'|s,a) * b(s)
        2. Update:   b''(s') = O(o|s',a) * b'(s') / normalisation

    Args:
        pomdp: A POMDP environment.
        belief: dict state -> probability (current belief before action).
        action: The action taken.
        observation: The observation received after the action.

    Returns:
        dict state -> probability: the updated belief after both steps.
    """
    predicted = predict_belief(pomdp, belief, action)
    return bayesian_belief_update(pomdp, predicted, observation, action)