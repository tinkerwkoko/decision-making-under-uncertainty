
# Epsilon-Greedy vs Thompson Sampling

## Overview

This experiment compares two approaches to solving the multi-armed bandit problem:

- Epsilon-Greedy
- Thompson Sampling

The objective is to evaluate how effectively each algorithm learns to select the best-performing arm while minimizing cumulative regret.

## Environment

The experiment uses a Bernoulli multi-armed bandit with four arms.

| Arm | Reward Probability |
|---|---:|
| 0 | 0.1 |
| 1 | 0.3 |
| 2 | 0.5 |
| 3 | 0.8 |

Each arm produces a binary reward:
- 1 represents success.
- 0 represents failure.

Arm 3 is the optimal arm because it has the highest reward probability.

## Algorithms

### Epsilon-Greedy

Epsilon-Greedy uses an exploration probability of 0.1.

- With probability 0.1, it selects a random arm.
- Otherwise, it selects an arm with the highest estimated reward.

### Thompson Sampling

Thompson Sampling maintains Beta distributions for each arm.

It samples from these distributions and selects the arm with the highest sampled value.

After observing a reward, it updates the corresponding arm's Beta parameters.

## Experimental Setup

- Number of arms: 4
- Number of rounds per simulation: 1,000
- Number of independent simulations: 100
- Epsilon: 0.1
- Reward distribution: Bernoulli
- Evaluation metric: Average cumulative expected regret

## Results

| Algorithm | Final Average Cumulative Regret |
|---|---:|
| Epsilon-Greedy | 54.276 |
| Thompson Sampling | 12.598 |

Thompson Sampling achieved approximately 76.8% lower cumulative regret than Epsilon-Greedy in this experiment.

The regret curves show that Epsilon-Greedy accumulated regret more steadily, while Thompson Sampling's regret growth slowed considerably as it learned to favour the optimal arm.

## Interpretation

The results suggest that Thompson Sampling was more effective at balancing exploration and exploitation in this experimental setting.

Epsilon-Greedy continued to explore randomly at a fixed probability, even after accumulating evidence about the best arm.

Thompson Sampling adapted its exploration through its posterior distributions, allowing it to increasingly favour promising arms.

These findings apply to the chosen environment and experimental configuration. They should not be interpreted as proof that Thompson Sampling always outperforms Epsilon-Greedy.

## Reproducing the Experiment

Run the following command from the project root:

```bash
python -c "from experiments.run_bandit_comparison import run_comparison; results = run_comparison([0.1, 0.3, 0.5, 0.8], n_rounds=1000, n_simulations=100); print('Epsilon-Greedy:', round(results['epsilon_greedy'][-1], 3)); print('Thompson Sampling:', round(results['thompson_sampling'][-1], 3))"
```

## Implementation

- `algorithms/bandit_algorithms.py`
- `environments/bandit.py`
- `experiments/bandit_metrics.py`
- `experiments/compare_bandit_algorithms.py`
- `experiments/run_bandit_comparison.py`
- `tests/test_run_bandit_comparison.py`
