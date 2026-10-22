def calculate_reconfiguration_cost(
    old_allocation,
    new_allocation,
):
    """
    Calculate how much resource moved during
    a reconfiguration.
    """

    moved = sum(
        abs(
            new_allocation[name]
            - old_allocation[name]
        )
        for name in old_allocation
    )

    # Divide by two because a resource moved
    # from one slice to another appears twice.
    return moved / 2


def count_reconfigurations(allocation_history):
    """
    Count allocation changes.
    """

    count = 0

    for previous, current in zip(
        allocation_history[:-1],
        allocation_history[1:],
    ):

        if previous != current:
            count += 1

    return count


def total_reconfiguration_cost(cost_history):
    """
    Total amount of resources moved.
    """

    return sum(cost_history)