"""
Inspect the generated dataset and run basic validation checks.

    python -m ml.inspect_dataset [--export-features data/ml/rf_features.csv]
"""

import argparse
import os

from ml.feature_table import build_table, validate_table
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default=os.path.join("data", "ml", "ml_traffic_runs.csv"))
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--export-features", default=None)
    args = parser.parse_args()

    if not os.path.exists(args.runs):
        raise FileNotFoundError(
            f"{args.runs} not found. Run: python -m ml.generate_dataset"
        )

    runs = pd.read_csv(args.runs)

    print("Runs per split:")
    print(runs.groupby("split")["run_id"].nunique().to_string())
    print("\nSeeds are disjoint across splits:",
          runs.groupby("seed")["split"].nunique().max() == 1)

    print("\nDemand statistics (Mbps):")
    print(runs[["embb_mbps", "urllc_mbps", "iot_mbps"]].describe().round(2).to_string())

    table = build_table(runs, args.horizon)
    validate_table(table)
    print(f"\nFeature table: {len(table)} rows x {table.shape[1]} columns "
          f"(horizon={args.horizon}); no missing / non-finite / negative values.")

    expected = (len(runs) - runs["run_id"].nunique() * args.horizon) * 3
    print(f"Rows dropped at trace ends: {len(runs) * 3 - len(table)} "
          f"(expected {runs['run_id'].nunique() * args.horizon * 3})")
    assert len(table) == expected

    print("\nRows per split and slice:")
    print(table.groupby(["split", "slice"]).size().unstack().to_string())

    print("\nFirst rows (selected columns):")
    print(table[["run_id", "slice", "time", "scenario", "current_mbps",
                 "lag_1", "ma5", "target_mbps"]].head(6).to_string(index=False))

    if args.export_features:
        os.makedirs(os.path.dirname(args.export_features) or ".", exist_ok=True)
        table.to_csv(args.export_features, index=False)
        print(f"\nFeature table exported to {args.export_features}")


if __name__ == "__main__":
    main()
