import pandas as pd
import matplotlib.pyplot as plt


DATA_FILE = "../data/raw/traffic_dataset.csv"


def plot_scenario(data, scenario):
    """Plot traffic for one scenario."""

    scenario_data = data[data["scenario"] == scenario]

    plt.figure(figsize=(12, 5))

    plt.plot(
        scenario_data["time"],
        scenario_data["embb_mbps"],
        label="eMBB"
    )

    plt.plot(
        scenario_data["time"],
        scenario_data["urllc_mbps"],
        label="URLLC"
    )

    plt.plot(
        scenario_data["time"],
        scenario_data["iot_mbps"],
        label="IoT"
    )

    plt.title(f"Traffic Pattern - {scenario}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Traffic (Mbps)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    filename = f"../plots/traffic_{scenario}.png"
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"Saved: {filename}")


def main():

    data = pd.read_csv(DATA_FILE)

    print("Traffic dataset loaded successfully.")
    print()
    print("Dataset shape:")
    print(data.shape)

    print()
    print("Scenarios:")
    print(data["scenario"].unique())

    print()
    print("Traffic statistics:")
    print(
        data[
            ["embb_mbps", "urllc_mbps", "iot_mbps"]
        ].describe()
    )

    scenarios = data["scenario"].unique()

    for scenario in scenarios:
        plot_scenario(data, scenario)


if __name__ == "__main__":
    main()