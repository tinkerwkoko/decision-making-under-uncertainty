"""
POMDP Environment for Belief-State Reasoning.

A Partially Observable Markov Decision Process (POMDP) extends an MDP by
assuming the agent cannot directly observe the true state. Instead, the
agent receives observations that are probabilistically related to the
hidden state, and maintains a belief distribution over states.

This is an educational implementation: it is NOT a general-purpose
optimal POMDP solver. It demonstrates belief prediction, Bayesian belief
updates, and normalisation under partial observability.
"""

import random


class POMDP:
    """A discrete POMDP with hidden states, actions, and observations."""

    def __init__(
        self,
        states,
        actions,
        observations,
        transition_probs,
        observation_probs,
        rewards,
        start_state,
        start_belief=None,
        seed=None,
    ):
        """Initialise the POMDP."""
        if not states:
            raise ValueError("At least one state is required.")
        if not actions:
            raise ValueError("At least one action is required.")
        if not observations:
            raise ValueError("At least one observation is required.")
        if start_state not in states:
            raise ValueError("start_state must be one of the states.")

        self.states = list(states)
        self.actions = list(actions)
        self.observations = list(observations)
        self.transition_probs = dict(transition_probs)
        self.observation_probs = dict(observation_probs)
        self.rewards = dict(rewards)
        self.start_state = start_state
        self.rng = random.Random(seed)

        self._validate_transitions()
        self._validate_observations()

        if start_belief is None:
            uniform = 1.0 / len(self.states)
            self.start_belief = {s: uniform for s in self.states}
        else:
            self._validate_belief(start_belief)
            self.start_belief = dict(start_belief)

        self.current_state = start_state
        self.current_belief = dict(self.start_belief)

    def _validate_transitions(self):
        """Validate that transition probabilities sum to 1 for each (s,a)."""
        for state in self.states:
            for action in self.actions:
                total = sum(
                    self.transition_probs.get((next_s, state, action), 0.0)
                    for next_s in self.states
                )
                if abs(total - 1.0) > 1e-9:
                    raise ValueError(
                        f"Transition probabilities from state {state} "
                        f"under action {action} must sum to 1.0, got {total}."
                    )

    def _validate_observations(self):
        """Validate that observation probabilities sum to 1 for each (s',a)."""
        for next_s in self.states:
            for action in self.actions:
                total = sum(
                    self.observation_probs.get((obs, next_s, action), 0.0)
                    for obs in self.observations
                )
                if abs(total - 1.0) > 1e-9:
                    raise ValueError(
                        f"Observation probabilities for state {next_s} "
                        f"under action {action} must sum to 1.0, got {total}."
                    )

    def _validate_belief(self, belief):
        """Validate that belief is a valid probability distribution."""
        if set(belief.keys()) != set(self.states):
            raise ValueError("Belief must have a probability for every state.")
        if any(p < 0 for p in belief.values()):
            raise ValueError("Belief probabilities must be non-negative.")
        if abs(sum(belief.values()) - 1.0) > 1e-9:
            raise ValueError("Belief probabilities must sum to 1.0.")

    def reset(self, seed=None):
        """Reset the POMDP to the start state and belief.

        Args:
            seed: Optional new random seed.

        Returns:
            The initial observation (sampled from the start state).
        """
        if seed is not None:
            self.rng = random.Random(seed)
        self.current_state = self.start_state
        self.current_belief = dict(self.start_belief)
        return self._sample_observation(self.current_state, None)

    def transition(self, state, action):
        """Sample the next state given current state and action.

        Args:
            state: Current hidden state.
            action: Action taken.

        Returns:
            The next hidden state, sampled from P(s'|s,a).
        """
        probs = [
            self.transition_probs.get((next_s, state, action), 0.0)
            for next_s in self.states
        ]
        return self._sample_from_distribution(self.states, probs)

    def observe(self, next_state, action):
        """Sample an observation given the next state and action.

        Args:
            next_state: The resulting hidden state.
            action: The action that led to this state.

        Returns:
            An observation, sampled from P(o|s',a).
        """
        probs = [
            self.observation_probs.get((obs, next_state, action), 0.0)
            for obs in self.observations
        ]
        return self._sample_from_distribution(self.observations, probs)

    def reward(self, state, action, next_state):
        """Return the reward for transitioning state -> next_state via action."""
        return self.rewards.get((state, action, next_state), 0.0)

    def step(self, action):
        """Advance the POMDP by one step.

        The true state transitions, and an observation is sampled. The
        agent's belief is NOT updated here — that is done by the belief
        update functions in algorithms/pomdp.py.

        Args:
            action: The action to execute.

        Returns:
            (observation, reward, done) where done is always False for
            this continuing POMDP (episodes are externally bounded).
        """
        if action not in self.actions:
            raise ValueError(f"Invalid action: {action}")

        next_state = self.transition(self.current_state, action)
        rew = self.reward(self.current_state, action, next_state)
        obs = self.observe(next_state, action)

        self.current_state = next_state
        return obs, rew, False

    def _sample_from_distribution(self, items, probs):
        """Sample an item from a categorical distribution defined by probs."""
        total = sum(probs)
        if total <= 0:
            raise ValueError("Distribution probabilities must sum to > 0.")
        normalised = [p / total for p in probs]
        r = self.rng.random()
        cumulative = 0.0
        for item, p in zip(items, normalised):
            cumulative += p
            if r <= cumulative:
                return item
        return items[-1]

    def _sample_observation(self, state, action):
        """Sample an initial observation (action=None treated specially)."""
        if action is None:
            probs = [1.0 / len(self.observations)] * len(self.observations)
        else:
            probs = [
                self.observation_probs.get((obs, state, action), 0.0)
                for obs in self.observations
            ]
        return self._sample_from_distribution(self.observations, probs)

    @property
    def true_state(self):
        """The true hidden state (not observable to the agent)."""
        return self.current_state

    @property
    def belief(self):
        """The agent's current belief distribution over hidden states."""
        return dict(self.current_belief)

    def set_belief(self, belief):
        """Set the agent's belief distribution (validated)."""
        self._validate_belief(belief)
        self.current_belief = dict(belief)