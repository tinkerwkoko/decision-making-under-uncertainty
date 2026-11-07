
def calculate_cumulative_regret(probabilities, selected_arms):
    """
    Calculate cumulative expected regret over a sequence of arm selections.

    Args:
        probabilities: True reward probabilities of all arms.
        selected_arms: Sequence of arms selected by an algorithm.

    Returns:
        A list containing cumulative regret after each selection.
    """
    if not probabilities:
        raise ValueError("At least one arm is required.")

    if any(
        not isinstance(p, (int, float))
        or isinstance(p, bool)
        or not 0 <= p <= 1
        for p in probabilities
    ):
        raise ValueError("All probabilities must be between 0 and 1.")

    if any(
        not isinstance(arm, int)
        or isinstance(arm, bool)
        or not 0 <= arm < len(probabilities)
        for arm in selected_arms
    ):
        raise ValueError("Selected arms must be valid arm indices.")

    best_probability = max(probabilities)

    cumulative_regret = []
    total_regret = 0.0

    for arm in selected_arms:
        regret = best_probability - probabilities[arm]
        total_regret += regret
        cumulative_regret.append(total_regret)

    return cumulative_regret
