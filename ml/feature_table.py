"""
Turn traffic runs into a supervised-learning table (no leakage).

For every run, slice and time index t:

    inputs  = features computed from demand[0..t]   (past + present only)
    target  = demand[t + horizon]                    (the future)

Rows with t + horizon beyond the end of the trace have no target and are
DROPPED (never padded or filled), so the end of each trace is handled
correctly. Runs are kept separate: no feature window ever crosses from one
run into another.
"""

import numpy as np
import pandas as pd

from controller.ml_features import FEATURE_NAMES, SLICES, build_features

SLICE_TO_COLUMN = {
    "embb": "embb_mbps",
    "urllc": "urllc_mbps",
    "iot": "iot_mbps",
}


def build_run_table(run_df, horizon=1):
    """Feature table for ONE run (dataframe ordered by time)."""

    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    series = {
        name: run_df[SLICE_TO_COLUMN[name]].to_numpy(dtype=float)
        for name in SLICES
    }
    scenarios = run_df["scenario"].to_numpy()
    times = run_df["time"].to_numpy()
    n = len(run_df)

    rows, meta = [], []

    for t in range(n - horizon):                      # last rows dropped
        histories = {name: series[name][: t + 1] for name in SLICES}

        for name in SLICES:
            rows.append(build_features(name, histories))
            meta.append(
                (
                    name,
                    t,
                    times[t],
                    scenarios[t],
                    series[name][t],
                    series[name][t + horizon],
                )
            )

    table = pd.DataFrame(rows, columns=FEATURE_NAMES)
    meta = pd.DataFrame(
        meta,
        columns=[
            "slice",
            "index",
            "time",
            "scenario",
            "current_mbps",
            "target_mbps",
        ],
    )

    return pd.concat([meta, table], axis=1)


def build_table(runs_df, horizon=1):
    """Feature table for all runs in a long dataframe with run_id."""

    parts = []

    for run_id, run in runs_df.groupby("run_id", sort=False):
        part = build_run_table(run.reset_index(drop=True), horizon)
        part.insert(0, "run_id", run_id)
        part.insert(1, "split", run["split"].iloc[0])
        parts.append(part)

    return pd.concat(parts, ignore_index=True)


def validate_table(table):
    """Raise ValueError on missing/invalid values."""

    numeric = table[list(FEATURE_NAMES) + ["current_mbps", "target_mbps"]]

    if numeric.isna().any().any():
        raise ValueError("Dataset contains missing values.")

    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains non-finite values.")

    if (table["target_mbps"] < 0).any() or (table["current_mbps"] < 0).any():
        raise ValueError("Dataset contains negative demand values.")
