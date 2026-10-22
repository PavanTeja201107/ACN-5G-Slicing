from copy import deepcopy


MIN_ALLOCATION = {
    "embb": 20.0,
    "urllc": 20.0,
    "iot": 10.0,
}

MAX_ALLOCATION = {
    "embb": 70.0,
    "urllc": 60.0,
    "iot": 40.0,
}

# Controller parameters
STEP = 5.0
COOLDOWN = 10
INCREASE_THRESHOLD = 0.80
DECREASE_THRESHOLD = 0.50


def normalize_allocation(allocation):
    """Ensure allocations add up to 100%."""

    total = sum(allocation.values())

    if total == 0:
        return allocation

    return {
        name: value * 100.0 / total
        for name, value in allocation.items()
    }


def predict_demand(history, window=5):
    """
    Simple short-term prediction using a moving average.
    """

    if not history:
        return 0.0

    recent = history[-window:]

    return sum(recent) / len(recent)


def calculate_utilization(demand, allocation):
    """Calculate slice utilization."""

    return {
        name: (
            demand[name] / allocation[name]
            if allocation[name] > 0
            else 0.0
        )
        for name in allocation
    }


def fairness_index(allocation):
    """Calculate Jain's fairness index."""

    values = list(allocation.values())

    numerator = sum(values) ** 2
    denominator = len(values) * sum(
        value ** 2 for value in values
    )

    if denominator == 0:
        return 1.0

    return numerator / denominator


def find_donor(demand, allocation):
    """
    Find a slice that can release resources.
    """

    utilization = calculate_utilization(
        demand,
        allocation
    )

    candidates = [
        name
        for name in allocation
        if utilization[name] < DECREASE_THRESHOLD
        and allocation[name] > MIN_ALLOCATION[name]
    ]

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda name: utilization[name]
    )


def dynamic_reallocation(
    demand,
    current_allocation,
    step=STEP,
):
    """
    Perform one resource reallocation.
    """

    allocation = deepcopy(current_allocation)

    utilization = calculate_utilization(
        demand,
        allocation
    )

    # Find the most overloaded slice
    overloaded = max(
        utilization,
        key=utilization.get
    )

    # No reallocation if demand is acceptable
    if utilization[overloaded] <= INCREASE_THRESHOLD:
        return allocation, False, None, None

    donor = find_donor(
        demand,
        allocation
    )

    if donor is None:
        return allocation, False, overloaded, None

    transferable = min(
        step,
        allocation[donor] - MIN_ALLOCATION[donor]
    )

    transferable = min(
        transferable,
        MAX_ALLOCATION[overloaded]
        - allocation[overloaded]
    )

    if transferable <= 0:
        return allocation, False, overloaded, donor

    allocation[donor] -= transferable
    allocation[overloaded] += transferable

    allocation = normalize_allocation(
        allocation
    )

    return allocation, True, overloaded, donor