"""
Full prediction-accuracy evaluation of RF vs the existing moving average.

    python -m ml.evaluate_predictors

Evaluated on (1) held-out test runs (seeds never used for training) and
(2) the project's original canonical stream data/raw/traffic_dataset.csv.
Writes tables to results/ml/ and plots to plots/ml/.
"""

import argparse
import json
import os

import joblib
import pandas as pd

from controller.ml_predictor import DEFAULT_MODEL_PATH, RandomForestPredictor, metadata_path_for
from ml import evaluation as ev
from ml.feature_table import build_table, validate_table
from ml.train_rf import predict_next_demand


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default=os.path.join("data", "ml", "ml_traffic_runs.csv"))
    p.add_argument("--canonical", default=os.path.join("data", "raw", "traffic_dataset.csv"))
    p.add_argument("--model", default=DEFAULT_MODEL_PATH)
    p.add_argument("--out-dir", default=os.path.join("results", "ml"))
    p.add_argument("--plots-dir", default=os.path.join("plots", "ml"))
    args = p.parse_args()

    predictor = RandomForestPredictor.load(args.model)      # clear error if missing
    horizon = int(predictor.metadata["horizon"])
    mode = predictor.metadata["target_mode"]

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    runs = pd.read_csv(args.runs)
    test_runs = runs[runs["split"] == "test"]
    canonical = pd.read_csv(args.canonical)
    canonical["run_id"], canonical["split"] = "canonical", "canonical"

    sets = {
        "heldout_test_runs": build_table(test_runs, horizon).reset_index(drop=True),
        "canonical_stream": build_table(canonical, horizon).reset_index(drop=True),
    }

    all_metrics = []
    for name, table in sets.items():
        validate_table(table)
        preds = {"rf": predict_next_demand(predictor.model, table, mode), **ev.baseline_predictions(table)}
        m = ev.metrics_table(table, preds)
        m.insert(0, "dataset", name)
        all_metrics.append(m)

        first_run = table["run_id"].iloc[0]
        ev.plot_actual_vs_predicted(table, preds, os.path.join(args.plots_dir, f"actual_vs_predicted_{name}.png"),
                                    f"Next-interval demand: actual vs predicted ({name})", run_id=first_run)
        ev.plot_scatter(table, preds, os.path.join(args.plots_dir, f"scatter_{name}.png"),
                        f"Predicted vs actual ({name})")

    result = pd.concat(all_metrics, ignore_index=True)
    out = os.path.join(args.out_dir, "prediction_accuracy.csv")
    result.to_csv(out, index=False)

    print(f"Prediction target: demand at t+{horizon} (Mbps). Errors in Mbps.")
    for name in sets:
        print(f"\n== {name} ==")
        sub = result[(result.dataset == name) & (result.group_type != "scenario")]
        print(sub.drop(columns="dataset").round(3).to_string(index=False))
    print(f"\nSaved: {out}\nPlots: {args.plots_dir}")


if __name__ == "__main__":
    main()
