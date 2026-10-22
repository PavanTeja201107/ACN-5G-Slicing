import os

import numpy as np
import pandas as pd

from traffic_profiles import (
    normal_traffic,
    embb_burst,
    urllc_spike,
    iot_traffic,
    competition_traffic,
    fluctuation_traffic,
)


DURATION = 180
SEED = 42


def create_dataset():
    """Generate the complete traffic workload dataset."""

    np.random.seed(SEED)

    time = np.arange(DURATION)

    datasets = []

    # ---------------------------------------------------------
    # Scenario 1: Normal mixed traffic
    # ---------------------------------------------------------
    embb = normal_traffic(time)
    urllc = 10 + normal_traffic(time) * 0.15
    iot = iot_traffic(time)

    datasets.append(
        pd.DataFrame({
            "time": time,
            "embb_mbps": embb,
            "urllc_mbps": urllc,
            "iot_mbps": iot,
            "scenario": "normal",
        })
    )

    # ---------------------------------------------------------
    # Scenario 2: eMBB burst
    # ---------------------------------------------------------
    embb = embb_burst(time)
    urllc = 10 + normal_traffic(time) * 0.10
    iot = iot_traffic(time)

    datasets.append(
        pd.DataFrame({
            "time": time,
            "embb_mbps": embb,
            "urllc_mbps": urllc,
            "iot_mbps": iot,
            "scenario": "embb_burst",
        })
    )

    # ---------------------------------------------------------
    # Scenario 3: URLLC spike
    # ---------------------------------------------------------
    embb = normal_traffic(time)
    urllc = urllc_spike(time)
    iot = iot_traffic(time)

    datasets.append(
        pd.DataFrame({
            "time": time,
            "embb_mbps": embb,
            "urllc_mbps": urllc,
            "iot_mbps": iot,
            "scenario": "urllc_spike",
        })
    )

    # ---------------------------------------------------------
    # Scenario 4: IoT burst
    # ---------------------------------------------------------
    embb = normal_traffic(time)
    urllc = 10 + normal_traffic(time) * 0.10
    iot = iot_traffic(time) * 2

    datasets.append(
        pd.DataFrame({
            "time": time,
            "embb_mbps": embb,
            "urllc_mbps": urllc,
            "iot_mbps": iot,
            "scenario": "iot_burst",
        })
    )

    # ---------------------------------------------------------
    # Scenario 5: Competition between slices
    # ---------------------------------------------------------
    embb, urllc, iot = competition_traffic(time)

    datasets.append(
        pd.DataFrame({
            "time": time,
            "embb_mbps": embb,
            "urllc_mbps": urllc,
            "iot_mbps": iot,
            "scenario": "competition",
        })
    )

    # ---------------------------------------------------------
    # Scenario 6: Fluctuating traffic
    # ---------------------------------------------------------
    embb = fluctuation_traffic(time)
    urllc = 15 + fluctuation_traffic(time) * 0.5
    iot = iot_traffic(time)

    datasets.append(
        pd.DataFrame({
            "time": time,
            "embb_mbps": embb,
            "urllc_mbps": urllc,
            "iot_mbps": iot,
            "scenario": "fluctuation",
        })
    )

    return pd.concat(datasets, ignore_index=True)


def main():
    dataset = create_dataset()

    output_dir = "../data/raw"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        "traffic_dataset.csv"
    )

    dataset.to_csv(output_file, index=False)

    print("Traffic dataset generated successfully.")
    print(f"Rows: {len(dataset)}")
    print(f"Scenarios: {dataset['scenario'].nunique()}")
    print(f"Saved to: {output_file}")
    print()
    print("Rows per scenario:")
    print(dataset["scenario"].value_counts())
    print()
    print(dataset.head())


if __name__ == "__main__":
    main()