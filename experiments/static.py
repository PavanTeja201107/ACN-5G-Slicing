import os
import pandas as pd

from controller.qos_monitor import calculate_slice_qos
from controller.fairness import allocation_fairness


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "data/raw/traffic_dataset.csv"
OUTPUT_FILE = "data/processed/static_results.csv"

TOTAL_CAPACITY = 100.0

# Fixed resource allocation
STATIC_ALLOCATION = {
    "embb": 50.0,
    "urllc": 30.0,
    "iot": 20.0,
}


# ============================================================
# Main experiment
# ============================================================

def main():

    print("\nStatic slicing experiment")
    print("=" * 60)

    # --------------------------------------------------------
    # Load traffic dataset
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Traffic dataset not found: {INPUT_FILE}"
        )

    data = pd.read_csv(INPUT_FILE)

    print(f"Traffic rows loaded: {len(data)}")

    # --------------------------------------------------------
    # Calculate fixed allocation fairness
    # --------------------------------------------------------

    fairness = allocation_fairness(
        STATIC_ALLOCATION
    )

    results = []

    # --------------------------------------------------------
    # Process every traffic measurement
    # --------------------------------------------------------

    for _, row in data.iterrows():

        demand = {
            "embb": float(row["embb_mbps"]),
            "urllc": float(row["urllc_mbps"]),
            "iot": float(row["iot_mbps"]),
        }

        qos_results = {}

        for slice_name in demand:

            qos_results[slice_name] = calculate_slice_qos(
                demand[slice_name],
                STATIC_ALLOCATION[slice_name],
                slice_name,
            )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        result = {
            "time": row["time"],
            "scenario": row["scenario"],

            # Demand
            "embb_demand_mbps": demand["embb"],
            "urllc_demand_mbps": demand["urllc"],
            "iot_demand_mbps": demand["iot"],

            # Fixed allocation
            "embb_allocation_pct": STATIC_ALLOCATION["embb"],
            "urllc_allocation_pct": STATIC_ALLOCATION["urllc"],
            "iot_allocation_pct": STATIC_ALLOCATION["iot"],

            # Utilization
            "embb_utilization": qos_results["embb"]["utilization"],
            "urllc_utilization": qos_results["urllc"]["utilization"],
            "iot_utilization": qos_results["iot"]["utilization"],

            # Latency
            "embb_latency_ms": qos_results["embb"]["latency_ms"],
            "urllc_latency_ms": qos_results["urllc"]["latency_ms"],
            "iot_latency_ms": qos_results["iot"]["latency_ms"],

            # Throughput
            "embb_throughput_mbps": qos_results["embb"][
                "throughput_mbps"
            ],
            "urllc_throughput_mbps": qos_results["urllc"][
                "throughput_mbps"
            ],
            "iot_throughput_mbps": qos_results["iot"][
                "throughput_mbps"
            ],

            # Packet loss
            "embb_packet_loss": qos_results["embb"][
                "packet_loss"
            ],
            "urllc_packet_loss": qos_results["urllc"][
                "packet_loss"
            ],
            "iot_packet_loss": qos_results["iot"][
                "packet_loss"
            ],

            # QoS violations
            "embb_qos_violation": qos_results["embb"][
                "qos_violation"
            ],
            "urllc_qos_violation": qos_results["urllc"][
                "qos_violation"
            ],
            "iot_qos_violation": qos_results["iot"][
                "qos_violation"
            ],

            # Fairness
            "fairness": fairness,
        }

        results.append(result)

    # ========================================================
    # Create dataframe
    # ========================================================

    results_df = pd.DataFrame(results)

    # ========================================================
    # Summary statistics
    # ========================================================

    qos_violations = {
        "eMBB": int(
            results_df["embb_qos_violation"].sum()
        ),
        "URLLC": int(
            results_df["urllc_qos_violation"].sum()
        ),
        "IoT": int(
            results_df["iot_qos_violation"].sum()
        ),
    }

    # ========================================================
    # Save results
    # ========================================================

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # Print summary
    # ========================================================

    print("\nExperiment completed.")
    print("-" * 60)

    print(
        f"Rows processed: {len(results_df)}"
    )

    print(
        f"Static allocation: "
        f"{STATIC_ALLOCATION}"
    )

    print(
        f"Fairness: "
        f"{fairness:.4f}"
    )

    print("\nQoS violations:")

    for slice_name, count in qos_violations.items():

        print(
            f"{slice_name}: {count}"
        )

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    print("\nAverage latency:")

    print(
        results_df[
            [
                "embb_latency_ms",
                "urllc_latency_ms",
                "iot_latency_ms",
            ]
        ].mean()
    )

    # --------------------------------------------------------
    # Throughput
    # --------------------------------------------------------

    print("\nAverage throughput:")

    print(
        results_df[
            [
                "embb_throughput_mbps",
                "urllc_throughput_mbps",
                "iot_throughput_mbps",
            ]
        ].mean()
    )

    # --------------------------------------------------------
    # Packet loss
    # --------------------------------------------------------

    print("\nAverage packet loss:")

    print(
        results_df[
            [
                "embb_packet_loss",
                "urllc_packet_loss",
                "iot_packet_loss",
            ]
        ].mean()
    )

    # --------------------------------------------------------
    # Utilization
    # --------------------------------------------------------

    print("\nAverage utilization:")

    print(
        results_df[
            [
                "embb_utilization",
                "urllc_utilization",
                "iot_utilization",
            ]
        ].mean()
    )

    print("\nResults saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()