"""
Stricter generalisation check: leave one traffic scenario out.

    python -m ml.leave_scenario_out

For each of the six scenarios, train an RF on the train-split rows of the
OTHER five scenarios and test on the held-out scenario (test-split runs).
Uses the same hyperparameters as the saved model but a lighter forest
(--n-estimators, default 40) so it finishes quickly.
"""

import argparse
import os

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from controller.ml_predictor import DEFAULT_MODEL_PATH, metadata_path_for
from ml import evaluation as ev
from ml.feature_table import build_table
from ml.train_rf import features_and_target, predict_next_demand
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default=os.path.join("data", "ml", "ml_traffic_runs.csv"))
    p.add_argument("--model", default=DEFAULT_MODEL_PATH)
    p.add_argument("--n-estimators", type=int, default=40)
    p.add_argument("--out", default=os.path.join("results", "ml", "leave_scenario_out.csv"))
    args = p.parse_args()

    with open(metadata_path_for(args.model), encoding="utf-8") as h:
        meta = json.load(h)
    hp, horizon, mode = meta["hyperparameters"], meta["horizon"], meta["target_mode"]

    table = build_table(pd.read_csv(args.runs), horizon)
    train_all = table[table.split == "train"]
    test_all = table[table.split == "test"]

    rows = []
    for scenario in table["scenario"].unique():
        train = train_all[train_all.scenario != scenario].reset_index(drop=True)
        test = test_all[test_all.scenario == scenario].reset_index(drop=True)
        X, y = features_and_target(train, mode)
        model = RandomForestRegressor(
            n_estimators=args.n_estimators, max_depth=hp["max_depth"],
            min_samples_leaf=hp["min_samples_leaf"], max_features=hp["max_features"],
            random_state=hp["random_state"], n_jobs=1).fit(X, y)
        preds = {"rf": predict_next_demand(model, test, mode), **ev.baseline_predictions(test)}
        m = ev.metrics_table(test, preds, by=("overall",))
        m.insert(0, "held_out_scenario", scenario)
        rows.append(m)
        print(m[["held_out_scenario", "predictor", "mae_mbps", "rmse_mbps"]].round(3).to_string(index=False, header=(len(rows) == 1)))

    result = pd.concat(rows, ignore_index=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
