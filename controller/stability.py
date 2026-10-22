def hysteresis_decision(
    utilization,
    current_state,
    increase_threshold=0.80,
    decrease_threshold=0.50,
):
    """
    Decide whether a slice should increase,
    decrease, or keep its current allocation.
    """

    if current_state == "increase":
        if utilization < decrease_threshold:
            return "decrease"
        return "increase"

    if current_state == "decrease":
        if utilization > increase_threshold:
            return "increase"
        return "decrease"

    if utilization > increase_threshold:
        return "increase"

    if utilization < decrease_threshold:
        return "decrease"

    return "hold"


def count_oscillations(allocation_history):
    """
    Count back-and-forth allocation changes.

    Example:
    30 -> 35 -> 30 -> 35
    """

    if len(allocation_history) < 3:
        return 0

    directions = []

    for previous, current in zip(
        allocation_history[:-1],
        allocation_history[1:],
    ):

        if current > previous:
            directions.append("up")

        elif current < previous:
            directions.append("down")

    oscillations = 0

    for previous, current in zip(
        directions[:-1],
        directions[1:],
    ):

        if previous != current:
            oscillations += 1

    return oscillations