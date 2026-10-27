"""
Feature extraction for the Random Forest traffic predictor.

This module is the SINGLE source of truth for the feature vector. It is
used by the offline training pipeline (ml/) and by the online predictor
(controller/ml_predictor.py), so training and inference always use the
same features in the same order.

Leakage rule: build_features() only looks at the demand history that is
passed to it. The caller must pass history up to and including time t
(never beyond t). The prediction target (demand at t + horizon) is never
visible to this module.

Units: all demand values are in Mbps (the unit of the project's traffic
dataset columns embb_mbps / urllc_mbps / iot_mbps).
"""

import math

import numpy as np

SLICES = ("embb", "urllc", "iot")

# Number of lagged demand values given to the model (lag_0 = current).
NUM_LAGS = 10

# Same window as the existing moving-average predictor in
# experiments/dynamic.py (PREDICTION_WINDOW = 5).
MA_WINDOW = 5

EWMA_ALPHA = 0.5

FEATURE_NAMES = (
    ["slice_embb", "slice_urllc", "slice_iot"]
    + [f"lag_{k}" for k in range(NUM_LAGS)]
    + [
        "ma5",
        "ma10",
        "ewma",
        "std5",
        "diff1",
        "diff5",
        "range5",
        "embb_now",
        "urllc_now",
        "iot_now",
    ]
)


def _as_array(history, name):
    values = np.asarray(history, dtype=float)

    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            f"Demand history for slice '{name}' must be a "
            f"non-empty 1-D sequence."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"Demand history for slice '{name}' contains "
            f"non-finite values."
        )

    return values


def build_features(slice_name, histories):
    """
    Build the feature vector for one slice at the current time t.

    Parameters
    ----------
    slice_name : str
        One of "embb", "urllc", "iot".
    histories : dict
        {"embb": [...], "urllc": [...], "iot": [...]} with demand
        history in Mbps, oldest first, newest (time t) last.

    Returns
    -------
    list of float, ordered exactly as FEATURE_NAMES.
    """

    if slice_name not in SLICES:
        raise ValueError(f"Unknown slice: {slice_name!r}")

    arrays = {name: _as_array(histories[name], name) for name in SLICES}

    own = arrays[slice_name]

    # lag_k = demand k intervals ago. Before enough history exists the
    # oldest available value is repeated (edge padding).
    lags = [own[-1 - k] if len(own) > k else own[0] for k in range(NUM_LAGS)]

    recent5 = own[-MA_WINDOW:]
    recent10 = own[-NUM_LAGS:]

    # Moving averages use only the available history, exactly like the
    # existing predict_demand() in experiments/dynamic.py.
    ma5 = float(np.mean(recent5))
    ma10 = float(np.mean(recent10))

    # EWMA over the padded lags, oldest -> newest.
    ewma = lags[-1]
    for value in reversed(lags[:-1]):
        ewma = EWMA_ALPHA * value + (1.0 - EWMA_ALPHA) * ewma

    std5 = float(np.std(recent5))
    diff1 = lags[0] - lags[1]
    diff5 = lags[0] - lags[5]
    range5 = float(np.max(recent5) - np.min(recent5))

    features = (
        [1.0 if slice_name == name else 0.0 for name in SLICES]
        + [float(v) for v in lags]
        + [ma5, ma10, float(ewma), std5, float(diff1), float(diff5), range5]
        + [float(arrays[name][-1]) for name in SLICES]
    )

    if len(features) != len(FEATURE_NAMES):
        raise RuntimeError("Feature vector length mismatch.")

    if not all(math.isfinite(v) for v in features):
        raise ValueError("Non-finite value in feature vector.")

    return features
