import random


class BernoulliBandit:
    def __init__(self, probabilities, seed=None):
        if not probabilities:
            raise ValueError("At least one arm is required.")

        if any(not 0.0 <= p <= 1.0 for p in probabilities):
            raise ValueError(
                "All probabilities must be between 0 and 1."
            )

        self.probabilities = probabilities
        self.n_arms = len(probabilities)
        self.rng = random.Random(seed)

        self.best_probability = max(probabilities)
        self.best_arm = probabilities.index(self.best_probability)

    def pull(self, arm):
        if not isinstance(arm, int) or isinstance(arm, bool):
            raise ValueError("Arm must be an integer.")

        if not 0 <= arm < self.n_arms:
            raise ValueError("Invalid arm index.")

        probability = self.probabilities[arm]

        reward = 1 if self.rng.random() < probability else 0

        return reward

    def reset(self, seed=None):
        self.rng = random.Random(seed)