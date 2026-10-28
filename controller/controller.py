from copy import deepcopy
import math
from numbers import Real

from controller.thresholds import demand_pressure


ALLOCATION_TOTAL = 100.0
ALLOCATION_TOLERANCE = 1e-8

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


def validate_allocation(allocation):
    """Validate a complete, bounded allocation expressed as percentages."""

    expected = set(MIN_ALLOCATION)
    if not isinstance(allocation, dict) or set(allocation) != expected:
        raise ValueError(
            "Allocation must contain exactly embb, urllc, and iot"
        )

    for name, value in allocation.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"Allocation for {name} must be numeric")
        if not math.isfinite(value):
            raise ValueError(f"Allocation for {name} must be finite")
        if value < 0:
            raise ValueError(f"Allocation for {name} must be non-negative")
        if value < MIN_ALLOCATION[name] - ALLOCATION_TOLERANCE:
            raise ValueError(
                f"Allocation for {name} is below minimum {MIN_ALLOCATION[name]}"
            )
        if value > MAX_ALLOCATION[name] + ALLOCATION_TOLERANCE:
            raise ValueError(
                f"Allocation for {name} exceeds maximum {MAX_ALLOCATION[name]}"
            )

    total = sum(allocation.values())
    if not math.isclose(
        total,
        ALLOCATION_TOTAL,
        rel_tol=0.0,
        abs_tol=ALLOCATION_TOLERANCE,
    ):
        raise ValueError(
            f"Allocation total must be {ALLOCATION_TOTAL}, got {total}"
        )


def normalize_allocation(allocation):
    """Normalize a valid allocation without repairing invalid input."""

    validate_allocation(allocation)

    total = sum(allocation.values())
    normalized = {
        name: value * 100.0 / total
        for name, value in allocation.items()
    }
    validate_allocation(normalized)
    return normalized


def predict_demand(history, window=5):
    """
    Simple short-term prediction using a moving average.
    """

    if not history:
        return 0.0

    recent = history[-window:]

    return sum(recent) / len(recent)


def calculate_utilization(demand, allocation):
    """Calculate slice utilization.

    Precondition: callers must provide a validated allocation containing all
    expected slices. The public dynamic reallocation path validates first.
    """

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
    """Find a slice that can release resources.

    Precondition: callers must provide a validated allocation. This helper is
    internal to the controller and is called by dynamic_reallocation().
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
    demand_pressures=None,
):
    """
    Perform one resource reallocation.
    """

    validate_allocation(current_allocation)
    allocation = deepcopy(current_allocation)

    utilization = calculate_utilization(
        demand,
        allocation
    )
    if demand_pressures is None:
        demand_pressures = {
            name: demand_pressure(demand[name], name)
            for name in allocation
        }

    # Find the most overloaded slice
    overloaded = max(
        utilization,
        key=utilization.get
    )
    pressure_overloaded = [
        name for name in allocation
        if demand_pressures[name] > 1.0
    ]

    # No reallocation if demand is acceptable
    if (
        utilization[overloaded] <= INCREASE_THRESHOLD
        and not pressure_overloaded
    ):
        return allocation, False, None, None
    if pressure_overloaded:
        overloaded = max(
            pressure_overloaded,
            key=lambda name: (demand_pressures[name], utilization[name])
        )

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