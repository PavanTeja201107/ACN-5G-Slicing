"""Helpers shared by compare_three.py and run_multiseed.py."""

import pandas as pd

from controller.stability import count_oscillations
from experiments.compare import calculate_metrics


def approach_metrics(df, dynamic=True):
    """Original project metrics + stability metrics for one result frame."""

    m = calculate_metrics(df)

    if dynamic:
        m["reconfigurations"] = int(df["reconfigured"].sum())
        m["total_reconfiguration_cost_pct"] = float(df["reconfiguration_cost"].sum())
        # same URLLC-allocation direction-change logic as experiments/compare.py
        m["oscillations"] = count_oscillations(df["urllc_allocation_pct"].tolist())
    else:
        m["reconfigurations"] = 0
        m["total_reconfiguration_cost_pct"] = 0.0
        m["oscillations"] = 0

    return m
