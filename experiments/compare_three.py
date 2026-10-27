"""
Compare Static vs Moving-Average dynamic vs Random-Forest dynamic.

    python -m experiments.compare_three

Reads   data/processed/static_results.csv
        data/processed/dynamic_results.csv        (moving average)
        data/processed/dynamic_rf_results.csv     (Random Forest)
Writes  results/comparison_three_way.csv
        results/prediction_accuracy_in_loop.csv
        plots/three_way_*.png
The original experiments/compare.py (static vs dynamic) is untouched.
"""

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.metrics_extra import approach_metrics
from experiments.validate_results import validate
from ml.evaluation import mae, rmse

FILES = {
    "Static": "data/processed/static_results.csv",
    "Dynamic-MA": "data/processed/dynamic_results.csv",
    "Dynamic-RF": "data/processed/dynamic_rf_results.csv",
}
SLICES = ("embb", "urllc", "iot")
LABELS = ("eMBB", "URLLC", "IoT")
COLORS = {"Static": "tab:gray", "Dynamic-MA": "tab:blue", "Dynamic-RF": "tab:red"}


def bar_plot(metrics, keys, title, ylabel, filename, scale=1.0, plots_dir="plots"):
    x = np.arange(len(keys))
    w = 0.27
    plt.figure(figsize=(9, 5))
    for i, (name, m) in enumerate(metrics.items()):
        plt.bar(x + (i - 1) * w, [m[k] * scale for k in keys], w, label=name, color=COLORS[name])
    plt.xticks(x, [k.split("_")[0] if "_" in k else k for k in keys])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, filename)
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved plot: {path}")


def in_loop_accuracy(ma_df, rf_df):
    """MAE/RMSE of the predictions the controller actually used (vs next-interval demand)."""
    rows = []
    for name, df, col in (("Dynamic-MA", ma_df, "prediction_mbps"), ("Dynamic-RF", rf_df, "prediction_mbps")):
        for s in SLICES:
            pred = df[f"{s}_{col}"].to_numpy()[:-1]
            actual = df[f"{s}_demand_mbps"].to_numpy()[1:]      # t+1; last row has no target
            rows.append((name, s, mae(actual, pred), rmse(actual, pred)))
        pred = np.concatenate([df[f"{s}_{col}"].to_numpy()[:-1] for s in SLICES])
        actual = np.concatenate([df[f"{s}_demand_mbps"].to_numpy()[1:] for s in SLICES])
        rows.append((name, "overall", mae(actual, pred), rmse(actual, pred)))
    return pd.DataFrame(rows, columns=["approach", "slice", "mae_mbps", "rmse_mbps"])


def main():
    for name, path in FILES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} results not found: {path}\n"
                f"Run the experiments first (see README: static, dynamic, dynamic --predictor rf)."
            )

    frames = {n: pd.read_csv(p) for n, p in FILES.items()}

    problems = validate(frames["Static"], frames["Dynamic-MA"], frames["Dynamic-RF"])
    print("Result validation:", "PASSED" if not problems else "FAILED")
    for p in problems:
        print("  -", p)
    if problems:
        raise SystemExit(1)

    metrics = {n: approach_metrics(df, dynamic=(n != "Static")) for n, df in frames.items()}
    table = pd.DataFrame({"Metric": list(metrics["Static"])})
    for n in metrics:
        table[n] = list(metrics[n].values())

    os.makedirs("results", exist_ok=True)
    table.to_csv("results/comparison_three_way.csv", index=False)
    print("\nThree-way comparison")
    print("-" * 70)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    acc = in_loop_accuracy(frames["Dynamic-MA"], frames["Dynamic-RF"])
    acc.to_csv("results/prediction_accuracy_in_loop.csv", index=False)
    print("\nPrediction accuracy of the values the controller used (Mbps, vs next-interval demand)")
    print(acc.round(3).to_string(index=False))

    keys = lambda suffix: [f"{l}_{suffix}" for l in ("eMBB", "URLLC", "IoT")]
    bar_plot(metrics, keys("avg_latency_ms"), "Average latency", "ms", "three_way_latency.png")
    bar_plot(metrics, keys("p95_latency_ms"), "95th-percentile latency", "ms", "three_way_p95_latency.png")
    bar_plot(metrics, keys("avg_throughput_mbps"), "Average throughput", "Mbps", "three_way_throughput.png")
    bar_plot(metrics, keys("avg_packet_loss"), "Average packet loss", "%", "three_way_packet_loss.png", scale=100)
    bar_plot(metrics, keys("qos_violations"), "QoS violations", "count", "three_way_qos_violations.png")
    bar_plot(metrics, keys("avg_utilization"), "Average utilization", "demand / allocation", "three_way_utilization.png")
    bar_plot(metrics, ["reconfigurations", "oscillations"], "Controller activity", "count", "three_way_stability.png")

    # allocation traces
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, s, l in zip(axes, SLICES, LABELS):
        for n, df in frames.items():
            ax.plot(df[f"{s}_allocation_pct"].to_numpy(), label=n, color=COLORS[n], lw=1.2)
        ax.set_ylabel(f"{l} share (%)")
        ax.grid(alpha=0.3)
    axes[0].legend(ncol=3)
    axes[-1].set_xlabel("Observation interval")
    fig.suptitle("Modelled allocation percentage over the 6-scenario stream")
    fig.tight_layout()
    fig.savefig("plots/three_way_allocation.png", dpi=150)
    plt.close(fig)
    print("Saved plot: plots/three_way_allocation.png")


if __name__ == "__main__":
    main()
