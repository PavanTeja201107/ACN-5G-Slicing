"""
Metrics and plots for traffic-demand predictors.

All errors are in Mbps (same unit as the traffic dataset).

  MAE  = mean(|prediction - actual|)           average miss, same units
  RMSE = sqrt(mean((prediction - actual)^2))   penalises large misses

Predictors compared on exactly the same rows:
  rf           - Random Forest (this extension)
  ma           - the project's existing moving-average predictor
                 (window 5) = the 'ma5' feature, which is computed the
                 same way as predict_demand() in experiments/dynamic.py
  persistence  - "next demand = current demand" (a sanity reference,
                 not part of the original project)
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SLICE_LABEL = {"embb": "eMBB", "urllc": "URLLC", "iot": "IoT/mMTC"}


def mae(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.mean(np.abs(predicted - actual)))


def rmse(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def baseline_predictions(table):
    return {
        "ma": table["ma5"].to_numpy(dtype=float),
        "persistence": table["current_mbps"].to_numpy(dtype=float),
    }


def metrics_table(table, predictions, by=("overall", "slice", "scenario")):
    """Rows: predictor x group. `predictions` = {name: array aligned to table}."""

    actual = table["target_mbps"].to_numpy(dtype=float)
    rows = []

    for name, pred in predictions.items():
        pred = np.asarray(pred, dtype=float)

        if "overall" in by:
            rows.append(("overall", "all", name, len(actual), mae(actual, pred), rmse(actual, pred)))

        for key in ("slice", "scenario"):
            if key not in by:
                continue
            for value, idx in table.groupby(key).indices.items():
                rows.append((key, value, name, len(idx), mae(actual[idx], pred[idx]), rmse(actual[idx], pred[idx])))

    return pd.DataFrame(
        rows, columns=["group_type", "group", "predictor", "n", "mae_mbps", "rmse_mbps"]
    )


def plot_actual_vs_predicted(table, predictions, path, title, run_id=None):
    """
    Actual vs predicted for each slice, in time order.

    `predictions` arrays are aligned POSITIONALLY with the rows of `table`.
    """

    slices = table["slice"].to_numpy()
    runs = table["run_id"].to_numpy()
    index = table["index"].to_numpy()
    actual = table["target_mbps"].to_numpy(dtype=float)

    if run_id is None:
        run_id = runs[0]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    styles = {"rf": ("tab:red", "-"), "ma": ("tab:blue", "--"), "persistence": ("tab:gray", ":")}

    for ax, name in zip(axes, ("embb", "urllc", "iot")):
        pos = np.where((slices == name) & (runs == run_id))[0]
        x = index[pos] + 1                      # interval the target refers to
        ax.plot(x, actual[pos], color="black", lw=1.6, label="actual")

        for pname, pred in predictions.items():
            color, ls = styles.get(pname, ("tab:green", "-"))
            ax.plot(x, np.asarray(pred, dtype=float)[pos], color=color, ls=ls, lw=1.0, label=pname)

        ax.set_ylabel(f"{SLICE_LABEL[name]} (Mbps)")
        ax.grid(alpha=0.3)

    axes[0].legend(ncol=4, loc="upper left")
    axes[-1].set_xlabel("Observation interval (concatenated 6-scenario stream)")
    fig.suptitle(f"{title}  [{run_id}]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter(table, predictions, path, title):
    """Predicted-vs-actual scatter for each predictor."""

    names = list(predictions)
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.5), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    actual = table["target_mbps"].to_numpy(dtype=float)
    top = max(actual.max(), max(np.max(p) for p in predictions.values()))

    for ax, name in zip(axes, names):
        ax.scatter(actual, predictions[name], s=3, alpha=0.3)
        ax.plot([0, top], [0, top], "k--", lw=1)
        ax.set_title(f"{name}  (MAE {mae(actual, predictions[name]):.2f}, RMSE {rmse(actual, predictions[name]):.2f})")
        ax.set_xlabel("actual next-interval demand (Mbps)")
    axes[0].set_ylabel("predicted (Mbps)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
