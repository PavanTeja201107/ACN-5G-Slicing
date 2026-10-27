"""
Train the Random Forest traffic-demand predictor.

    python -m ml.train_rf
    python -m ml.train_rf --n-estimators 200 --seed 7 --horizon 1

Pipeline: load runs -> validate -> build leakage-free features/targets ->
train on split=="train" runs -> report on held-out "val" and "test" runs
(different seeds, never used for fitting) -> save model + metadata +
metrics.

Outputs
  models/rf_traffic_predictor.joblib
  models/rf_traffic_predictor_metadata.json
  results/ml/training_metrics.csv
"""

import argparse
import json
import os
import platform
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
import joblib

from controller.ml_features import FEATURE_NAMES, NUM_LAGS
from controller.ml_predictor import DEFAULT_MODEL_PATH, metadata_path_for, file_sha256
from ml import evaluation as ev
from ml.feature_table import build_table, validate_table


def features_and_target(table, target_mode):
    X = table[list(FEATURE_NAMES)].to_numpy(dtype=float)
    y = table["target_mbps"].to_numpy(dtype=float)
    if target_mode == "delta":
        y = y - table["current_mbps"].to_numpy(dtype=float)
    return X, y


def predict_next_demand(model, table, target_mode):
    """Model output converted back to predicted demand in Mbps."""
    X = table[list(FEATURE_NAMES)].to_numpy(dtype=float)
    out = model.predict(X)
    if target_mode == "delta":
        out = out + table["current_mbps"].to_numpy(dtype=float)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default=os.path.join("data", "ml", "ml_traffic_runs.csv"))
    p.add_argument("--model-out", default=DEFAULT_MODEL_PATH)
    p.add_argument("--metrics-out", default=os.path.join("results", "ml", "training_metrics.csv"))
    p.add_argument("--horizon", type=int, default=1, help="intervals ahead to predict")
    p.add_argument("--target-mode", choices=["absolute", "delta"], default="delta",
                   help="delta: model learns (next - current); output is converted back to Mbps")
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--min-samples-leaf", type=int, default=20)
    p.add_argument("--max-features", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=1, help="CPU threads (1 = reproducible & light)")
    args = p.parse_args()

    if not os.path.exists(args.runs):
        raise FileNotFoundError(f"{args.runs} not found. Run: python -m ml.generate_dataset")

    runs = pd.read_csv(args.runs)
    table = build_table(runs, args.horizon)
    validate_table(table)

    splits = {s: table[table["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    for s, t in splits.items():
        if t.empty:
            raise ValueError(f"No rows in split '{s}'.")

    print(f"Rows: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])} "
          f"(horizon={args.horizon}, target={args.target_mode})")

    X, y = features_and_target(splits["train"], args.target_mode)

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )

    start = time.time()
    model.fit(X, y)
    fit_seconds = time.time() - start
    print(f"Trained RandomForestRegressor in {fit_seconds:.1f}s on CPU")

    rows = []
    for name in ("val", "test"):
        t = splits[name]
        preds = {"rf": predict_next_demand(model, t, args.target_mode), **ev.baseline_predictions(t)}
        m = ev.metrics_table(t, preds, by=("overall", "slice"))
        m.insert(0, "split", name)
        rows.append(m)

    metrics = pd.concat(rows, ignore_index=True)
    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    metrics.to_csv(args.metrics_out, index=False)

    print("\nHeld-out accuracy (Mbps):")
    print(metrics.round(3).to_string(index=False))

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(model, args.model_out, compress=3)

    importances = dict(zip(FEATURE_NAMES, [float(v) for v in model.feature_importances_]))
    metadata = {
        "model_type": "sklearn.ensemble.RandomForestRegressor",
        "feature_names": list(FEATURE_NAMES),
        "num_lags": NUM_LAGS,
        "horizon": args.horizon,
        "target_mode": args.target_mode,
        "target_unit": "Mbps",
        "hyperparameters": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "random_state": args.seed,
        },
        "training_runs": sorted(runs.loc[runs["split"] == "train", "seed"].unique().tolist()),
        "validation_runs": sorted(runs.loc[runs["split"] == "val", "seed"].unique().tolist()),
        "test_runs": sorted(runs.loc[runs["split"] == "test", "seed"].unique().tolist()),
        "training_rows": int(len(splits["train"])),
        "dataset_file": args.runs,
        "dataset_sha256": file_sha256(args.runs),
        "data_origin": "synthetic, derived from the project's modelled traffic scenarios",
        "feature_importances": importances,
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    with open(metadata_path_for(args.model_out), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    size_mb = os.path.getsize(args.model_out) / 1e6
    print(f"\nModel saved:    {args.model_out} ({size_mb:.1f} MB)")
    print(f"Metadata saved: {metadata_path_for(args.model_out)}")
    print(f"Metrics saved:  {args.metrics_out}")
    top = sorted(importances.items(), key=lambda kv: -kv[1])[:5]
    print("Top features:", ", ".join(f"{k}={v:.3f}" for k, v in top))


if __name__ == "__main__":
    main()
