import pandas as pd

from controller.qos_monitor import evaluate_dataset


DATA_FILE = "data/raw/traffic_dataset.csv"


def main():

    # Load traffic workload
    data = pd.read_csv(DATA_FILE)

    # Static resource allocation
    allocation = {
        "embb": 50,
        "urllc": 30,
        "iot": 20,
    }

    print("Testing QoS monitoring...")
    print()
    print("Static allocation:")
    print(allocation)
    print()

    results = evaluate_dataset(
        data,
        allocation,
        total_capacity=100,
    )

    print("QoS results generated.")
    print()
    print("Result shape:")
    print(results.shape)

    print()
    print("First five results:")
    print(results.head())

    print()
    print("Average latency:")
    print(
        results[
            [
                "embb_latency_ms",
                "urllc_latency_ms",
                "iot_latency_ms",
            ]
        ].mean()
    )

    print()
    print("Average throughput:")
    print(
        results[
            [
                "embb_throughput_mbps",
                "urllc_throughput_mbps",
                "iot_throughput_mbps",
            ]
        ].mean()
    )

    print()
    print("QoS violations:")

    print(
        "eMBB:",
        results["embb_qos_violation"].sum()
    )

    print(
        "URLLC:",
        results["urllc_qos_violation"].sum()
    )

    print(
        "IoT:",
        results["iot_qos_violation"].sum()
    )


if __name__ == "__main__":
    main()