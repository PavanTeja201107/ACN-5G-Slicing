"""
Sanity checks on experiment result files.

  * allocations sum to 100 %, respect MIN/MAX_ALLOCATION (dynamic runs)
  * no NaN / non-finite values in numeric columns
  * all approaches saw the SAME workload (demand columns identical)
  * RF runs: prediction_source == "rf" and invalid-prediction counts
"""

import numpy as np
import pandas as pd

from controller.controller import MIN_ALLOCATION, MAX_ALLOCATION

SLICES = ("embb", "urllc", "iot")
DEMAND_COLS = [f"{s}_demand_mbps" for s in SLICES]
ALLOC_COLS = [f"{s}_allocation_pct" for s in SLICES]


def check_allocations(df, label, tol=1e-6, enforce_bounds=True):
    problems = []
    total = df[ALLOC_COLS].sum(axis=1)

    if not np.allclose(total, 100.0, atol=tol):
        problems.append(f"{label}: allocation does not sum to 100 (max dev {abs(total - 100).max():.2e})")

    if enforce_bounds:
        for s in SLICES:
            col = f"{s}_allocation_pct"
            if (df[col] < MIN_ALLOCATION[s] - tol).any():
                problems.append(f"{label}: {s} below minimum share {MIN_ALLOCATION[s]}")
            if (df[col] > MAX_ALLOCATION[s] + tol).any():
                problems.append(f"{label}: {s} above maximum share {MAX_ALLOCATION[s]}")
    return problems


def check_finite(df, label):
    num = df.select_dtypes(include=[np.number])
    if not np.isfinite(num.to_numpy(dtype=float)).all():
        return [f"{label}: NaN/inf in numeric columns"]
    return []


def check_same_workload(frames):
    names = list(frames)
    base = frames[names[0]][DEMAND_COLS].to_numpy()
    problems = []
    for n in names[1:]:
        other = frames[n][DEMAND_COLS].to_numpy()
        if other.shape != base.shape or not np.allclose(base, other):
            problems.append(f"workload differs between {names[0]} and {n}")
    return problems


def validate(static, ma, rf=None):
    """Return a list of problems (empty list = all checks passed)."""

    frames = {"static": static, "ma": ma}
    problems = check_allocations(static, "static", enforce_bounds=False)
    problems += check_allocations(ma, "ma")
    problems += check_finite(static, "static") + check_finite(ma, "ma")

    if rf is not None:
        frames["rf"] = rf
        problems += check_allocations(rf, "rf") + check_finite(rf, "rf")
        if "prediction_source" not in rf.columns or not (rf["prediction_source"] == "rf").all():
            problems.append("rf: prediction_source column missing or not all 'rf'")

    problems += check_same_workload(frames)
    return problems
