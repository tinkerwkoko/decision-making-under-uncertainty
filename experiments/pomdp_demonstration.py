"""
POMDP Belief-State Demonstration.

Demonstrates how an agent reasons under partial observability by
maintaining and updating a belief distribution over hidden states.

The demonstration uses a simple 3-state POMDP:
- States: A, B, C (hidden from the agent)
- Actions: stay, move
- Observations: obs_x, obs_y (noisy signals about the true state)

The agent starts with a uniform belief, then receives a sequence of
actions and observations, updating its belief after each step using:
  1. Belief prediction:  b'(s') = sum over s of T(s'|s,a) * b(s)
  2. Bayesian update:    b''(s') = O(o|s',a) * b'(s') / normalisation

This shows how incomplete information affects the agent's certainty
about the true state.
"""

from environments.pomdp import POMDP
from algorithms.pomdp import predict_belief, bayesian_belief_update, update_belief


def make_demonstration_pomdp(seed=42):
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
        seed=seed,
    )


def demonstrate_belief_updates(n_steps=5, seed=42):
    """Run a step-by-step belief update demonstration."""
    pomdp = make_demonstration_pomdp(seed=seed)
    pomdp.reset(seed=seed)

    belief = dict(pomdp.belief)
    initial_belief = dict(belief)

    steps = []

    for step_index in range(n_steps):
        belief_before = dict(belief)
        action = "stay" if step_index % 2 == 0 else "move"
        observation, reward, done = pomdp.step(action)
        true_state = pomdp.true_state
        belief = update_belief(pomdp, belief_before, action, observation)

        steps.append({
            "step_index": step_index,
            "action": action,
            "observation": observation,
            "true_state": true_state,
            "reward": reward,
            "belief_before": belief_before,
            "belief_after": dict(belief),
        })

    return {
        "steps": steps,
        "initial_belief": initial_belief,
        "final_belief": dict(belief),
        "n_steps": n_steps,
        "seed": seed,
    }


def format_belief(belief):
    parts = [f"{s}: {p:.3f}" for s, p in sorted(belief.items())]
    return ", ".join(parts)


def print_demonstration(result):
    print("POMDP Belief-State Demonstration")
    print("=" * 50)
    print(f"Initial belief: {format_belief(result['initial_belief'])}")
    print()
    for step in result["steps"]:
        print(f"Step {step['step_index'] + 1}:")
        print(f"  Action: {step['action']}")
        print(f"  Observation: {step['observation']}")
        print(f"  True state: {step['true_state']} (hidden from agent)")
        print(f"  Belief before: {format_belief(step['belief_before'])}")
        print(f"  Belief after:  {format_belief(step['belief_after'])}")
        print()
    print(f"Final belief: {format_belief(result['final_belief'])}")
    print()
    print("Note: The agent never observes the true state directly.")
    print("It maintains a belief distribution that is updated via")
    print("belief prediction (transition model) and Bayesian update")
    print("(observation model) after each action-observation pair.")


def run_comparison_demonstration():
    """Compare how different observation sequences affect belief."""
    result1 = demonstrate_belief_updates(n_steps=5, seed=42)
    result2 = demonstrate_belief_updates(n_steps=5, seed=99)

    print("Comparison of Two Demonstration Runs")
    print("=" * 50)
    print(f"Seed 42  final belief: {format_belief(result1['final_belief'])}")
    print(f"Seed 99  final belief: {format_belief(result2['final_belief'])}")
    print()
    print("Different random seeds produce different observation sequences,")
    print("which lead to different belief distributions. This illustrates")
    print("how partial observability makes the agent's knowledge dependent on")
    print("the specific observations it receives.")


if __name__ == "__main__":
    result = demonstrate_belief_updates(n_steps=5, seed=42)
    print_demonstration(result)
    print()
    run_comparison_demonstration()