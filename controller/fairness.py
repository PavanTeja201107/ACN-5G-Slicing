import numpy as np


def jain_fairness(values):
    """
    Calculate Jain's fairness index.

    Returns a value between 0 and 1.

    1.0 = perfectly fair
    Lower values = less fair
    """

    values = np.asarray(values, dtype=float)

    if np.all(values == 0):
        return 1.0

    numerator = np.sum(values) ** 2
    denominator = len(values) * np.sum(values ** 2)

    if denominator == 0:
        return 1.0

    return numerator / denominator


def allocation_fairness(allocation):
    """
    Calculate fairness of slice resource allocation.

    allocation example:
        {
            "embb": 50,
            "urllc": 30,
            "iot": 20
        }
    """

    values = list(allocation.values())

    return jain_fairness(values)