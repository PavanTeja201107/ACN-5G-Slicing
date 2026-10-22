import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

STATIC_FILE = "data/processed/static_results.csv"
DYNAMIC_FILE = "data/processed/dynamic_results.csv"

SUMMARY_FILE = "results/comparison_summary.csv"

PLOTS_DIR = "plots"


# ============================================================
# Helper functions
# ============================================================

def calculate_metrics(data):
    """
    Calculate the main QoS metrics for one experiment.
    """

    metrics = {}

    # --------------------------------------------------------
    # Average latency
    # --------------------------------------------------------

    metrics["eMBB_avg_latency_ms"] = (
        data["embb_latency_ms"].mean()
    )

    metrics["URLLC_avg_latency_ms"] = (
        data["urllc_latency_ms"].mean()
    )

    metrics["IoT_avg_latency_ms"] = (
        data["iot_latency_ms"].mean()
    )

    # --------------------------------------------------------
    # 95th percentile latency
    # --------------------------------------------------------

    metrics["eMBB_p95_latency_ms"] = (
        data["embb_latency_ms"].quantile(0.95)
    )

    metrics["URLLC_p95_latency_ms"] = (
        data["urllc_latency_ms"].quantile(0.95)
    )

    metrics["IoT_p95_latency_ms"] = (
        data["iot_latency_ms"].quantile(0.95)
    )

    # --------------------------------------------------------
    # Average throughput
    # --------------------------------------------------------

    metrics["eMBB_avg_throughput_mbps"] = (
        data["embb_throughput_mbps"].mean()
    )

    metrics["URLLC_avg_throughput_mbps"] = (
        data["urllc_throughput_mbps"].mean()
    )

    metrics["IoT_avg_throughput_mbps"] = (
        data["iot_throughput_mbps"].mean()
    )

    # --------------------------------------------------------
    # Average packet loss
    # --------------------------------------------------------

    metrics["eMBB_avg_packet_loss"] = (
        data["embb_packet_loss"].mean()
    )

    metrics["URLLC_avg_packet_loss"] = (
        data["urllc_packet_loss"].mean()
    )

    metrics["IoT_avg_packet_loss"] = (
        data["iot_packet_loss"].mean()
    )

    # --------------------------------------------------------
    # QoS violations
    # --------------------------------------------------------

    metrics["eMBB_qos_violations"] = int(
        data["embb_qos_violation"].sum()
    )

    metrics["URLLC_qos_violations"] = int(
        data["urllc_qos_violation"].sum()
    )

    metrics["IoT_qos_violations"] = int(
        data["iot_qos_violation"].sum()
    )

    metrics["total_qos_violations"] = (
        metrics["eMBB_qos_violations"]
        + metrics["URLLC_qos_violations"]
        + metrics["IoT_qos_violations"]
    )

    # --------------------------------------------------------
    # Average utilization
    # --------------------------------------------------------

    metrics["eMBB_avg_utilization"] = (
        data["embb_utilization"].mean()
    )

    metrics["URLLC_avg_utilization"] = (
        data["urllc_utilization"].mean()
    )

    metrics["IoT_avg_utilization"] = (
        data["iot_utilization"].mean()
    )

    # --------------------------------------------------------
    # Fairness
    # --------------------------------------------------------

    metrics["average_fairness"] = (
        data["fairness"].mean()
    )

    return metrics


# ============================================================
# Plot helper
# ============================================================

def save_bar_plot(
    categories,
    static_values,
    dynamic_values,
    title,
    ylabel,
    filename,
):
    """
    Create and save a static-vs-dynamic bar chart.
    """

    x = range(len(categories))
    width = 0.35

    plt.figure(figsize=(9, 5))

    plt.bar(
        [i - width / 2 for i in x],
        static_values,
        width=width,
        label="Static",
    )

    plt.bar(
        [i + width / 2 for i in x],
        dynamic_values,
        width=width,
        label="Dynamic",
    )

    plt.xticks(
        list(x),
        categories,
    )

    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(
        PLOTS_DIR,
        filename
    )

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()

    print(f"Saved plot: {path}")


# ============================================================
# Main comparison
# ============================================================

def main():

    print("\nStatic vs Dynamic comparison")
    print("=" * 60)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(STATIC_FILE):
        raise FileNotFoundError(
            f"Static results not found: {STATIC_FILE}"
        )

    if not os.path.exists(DYNAMIC_FILE):
        raise FileNotFoundError(
            f"Dynamic results not found: {DYNAMIC_FILE}"
        )

    # --------------------------------------------------------
    # Load results
    # --------------------------------------------------------

    static = pd.read_csv(
        STATIC_FILE
    )

    dynamic = pd.read_csv(
        DYNAMIC_FILE
    )

    print(
        f"Static rows: {len(static)}"
    )

    print(
        f"Dynamic rows: {len(dynamic)}"
    )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    static_metrics = calculate_metrics(
        static
    )

    dynamic_metrics = calculate_metrics(
        dynamic
    )

    # --------------------------------------------------------
    # Create comparison table
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        {
            "Metric": static_metrics.keys(),
            "Static": static_metrics.values(),
            "Dynamic": dynamic_metrics.values(),
        }
    )

    # --------------------------------------------------------
    # Save comparison
    # --------------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True
    )

    comparison.to_csv(
        SUMMARY_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Print comparison
    # --------------------------------------------------------

    print("\nPerformance comparison")
    print("-" * 60)

    print(
        comparison.to_string(
            index=False
        )
    )

    # ========================================================
    # Dynamic-controller-specific metrics
    # ========================================================

    if "reconfigured" in dynamic.columns:

        reconfigurations = int(
            dynamic["reconfigured"].sum()
        )

    else:

        reconfigurations = 0

    if "reconfiguration_cost" in dynamic.columns:

        reconfiguration_cost = (
            dynamic["reconfiguration_cost"].sum()
        )

    else:

        reconfiguration_cost = 0.0

    # Recalculate oscillations using URLLC allocation direction.
    allocation = dynamic[
        "urllc_allocation_pct"
    ].tolist()

    directions = []

    for previous, current in zip(
        allocation[:-1],
        allocation[1:]
    ):

        if current > previous:
            directions.append("up")

        elif current < previous:
            directions.append("down")

    oscillations = 0

    for previous, current in zip(
        directions[:-1],
        directions[1:]
    ):

        if previous != current:
            oscillations += 1

    print("\nController stability metrics")
    print("-" * 60)

    print(
        f"Reconfigurations: "
        f"{reconfigurations}"
    )

    print(
        f"Total reconfiguration cost: "
        f"{reconfiguration_cost:.2f}%"
    )

    print(
        f"Oscillations: "
        f"{oscillations}"
    )

    # ========================================================
    # Generate plots
    # ========================================================

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Latency plot
    # --------------------------------------------------------

    save_bar_plot(
        categories=[
            "eMBB",
            "URLLC",
            "IoT",
        ],
        static_values=[
            static_metrics[
                "eMBB_avg_latency_ms"
            ],
            static_metrics[
                "URLLC_avg_latency_ms"
            ],
            static_metrics[
                "IoT_avg_latency_ms"
            ],
        ],
        dynamic_values=[
            dynamic_metrics[
                "eMBB_avg_latency_ms"
            ],
            dynamic_metrics[
                "URLLC_avg_latency_ms"
            ],
            dynamic_metrics[
                "IoT_avg_latency_ms"
            ],
        ],
        title="Average latency: static vs dynamic",
        ylabel="Latency (ms)",
        filename="comparison_latency.png",
    )

    # --------------------------------------------------------
    # Throughput plot
    # --------------------------------------------------------

    save_bar_plot(
        categories=[
            "eMBB",
            "URLLC",
            "IoT",
        ],
        static_values=[
            static_metrics[
                "eMBB_avg_throughput_mbps"
            ],
            static_metrics[
                "URLLC_avg_throughput_mbps"
            ],
            static_metrics[
                "IoT_avg_throughput_mbps"
            ],
        ],
        dynamic_values=[
            dynamic_metrics[
                "eMBB_avg_throughput_mbps"
            ],
            dynamic_metrics[
                "URLLC_avg_throughput_mbps"
            ],
            dynamic_metrics[
                "IoT_avg_throughput_mbps"
            ],
        ],
        title="Average throughput: static vs dynamic",
        ylabel="Throughput (Mbps)",
        filename="comparison_throughput.png",
    )

    # --------------------------------------------------------
    # QoS violations plot
    # --------------------------------------------------------

    save_bar_plot(
        categories=[
            "eMBB",
            "URLLC",
            "IoT",
        ],
        static_values=[
            static_metrics[
                "eMBB_qos_violations"
            ],
            static_metrics[
                "URLLC_qos_violations"
            ],
            static_metrics[
                "IoT_qos_violations"
            ],
        ],
        dynamic_values=[
            dynamic_metrics[
                "eMBB_qos_violations"
            ],
            dynamic_metrics[
                "URLLC_qos_violations"
            ],
            dynamic_metrics[
                "IoT_qos_violations"
            ],
        ],
        title="QoS violations: static vs dynamic",
        ylabel="Number of violations",
        filename="comparison_qos_violations.png",
    )

    # --------------------------------------------------------
    # Packet loss plot
    # --------------------------------------------------------

    save_bar_plot(
        categories=[
            "eMBB",
            "URLLC",
            "IoT",
        ],
        static_values=[
            static_metrics[
                "eMBB_avg_packet_loss"
            ] * 100,
            static_metrics[
                "URLLC_avg_packet_loss"
            ] * 100,
            static_metrics[
                "IoT_avg_packet_loss"
            ] * 100,
        ],
        dynamic_values=[
            dynamic_metrics[
                "eMBB_avg_packet_loss"
            ] * 100,
            dynamic_metrics[
                "URLLC_avg_packet_loss"
            ] * 100,
            dynamic_metrics[
                "IoT_avg_packet_loss"
            ] * 100,
        ],
        title="Average packet loss: static vs dynamic",
        ylabel="Packet loss (%)",
        filename="comparison_packet_loss.png",
    )

    # --------------------------------------------------------
    # Fairness plot
    # --------------------------------------------------------

    save_bar_plot(
        categories=[
            "Fairness",
        ],
        static_values=[
            static_metrics[
                "average_fairness"
            ],
        ],
        dynamic_values=[
            dynamic_metrics[
                "average_fairness"
            ],
        ],
        title="Resource allocation fairness",
        ylabel="Jain fairness index",
        filename="comparison_fairness.png",
    )

    print("\nComparison completed.")

    print("\nSummary saved to:")
    print(SUMMARY_FILE)

    print("\nPlots saved to:")
    print(PLOTS_DIR)


if __name__ == "__main__":
    main()