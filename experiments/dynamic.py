import os
import pandas as pd

from controller.qos_monitor import calculate_slice_qos
from controller.controller import (
    dynamic_reallocation,
    calculate_utilization,
    fairness_index,
)
from controller.stability import hysteresis_decision
from controller.reconfiguration import calculate_reconfiguration_cost
from controller.thresholds import demand_pressure


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "data/raw/traffic_dataset.csv"
OUTPUT_FILE = "data/processed/dynamic_results.csv"

TOTAL_CAPACITY = 100.0

INITIAL_ALLOCATION = {
    "embb": 50.0,
    "urllc": 30.0,
    "iot": 20.0,
}

# Prediction
PREDICTION_WINDOW = 5

# Stability
COOLDOWN = 10

# Hysteresis thresholds
INCREASE_THRESHOLD = 0.80
DECREASE_THRESHOLD = 0.50

# Prediction method (Random Forest extension).
#   "ma" = original moving-average predictor (default, unchanged behaviour)
#   "rf" = Random Forest predictor (controller/ml_predictor.py)
# Select from the command line: python -m experiments.dynamic --predictor rf
PREDICTOR = "ma"
MODEL_PATH = None  # None -> controller.ml_predictor.DEFAULT_MODEL_PATH


# ============================================================
# Helper functions
# ============================================================

def predict_demand(history, window=PREDICTION_WINDOW):
    """
    Predict near-future demand using a simple moving average.

    Example:
        recent demand = [25, 28, 30]
        prediction = 27.67
    """

    if not history:
        return 0.0

    recent = history[-window:]

    return sum(recent) / len(recent)


def allocation_changed(old, new):
    """Check whether resource allocation changed."""

    for name in old:
        if abs(old[name] - new[name]) > 1e-9:
            return True

    return False


def detect_oscillations(allocation_history):
    """
    Count allocation direction changes.

    Example:
        30 -> 35 -> 40 -> 35 -> 30

    Direction:
        up -> up -> down -> down

    One direction change = one oscillation.
    """

    if len(allocation_history) < 3:
        return 0

    directions = []

    for previous, current in zip(
        allocation_history[:-1],
        allocation_history[1:]
    ):

        previous_urlcc = previous["urllc"]
        current_urlcc = current["urllc"]

        if current_urlcc > previous_urlcc:
            directions.append("up")

        elif current_urlcc < previous_urlcc:
            directions.append("down")

    oscillations = 0

    for previous, current in zip(
        directions[:-1],
        directions[1:]
    ):

        if previous != current:
            oscillations += 1

    return oscillations


# ============================================================
# Main experiment
# ============================================================

def main():

    print("\nIntegrated dynamic slicing experiment")
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
    # Initial controller state
    # --------------------------------------------------------

    allocation = INITIAL_ALLOCATION.copy()

    demand_history = {
        "embb": [],
        "urllc": [],
        "iot": [],
    }

    allocation_history = [
        allocation.copy()
    ]

    controller_state = {
        "embb": "hold",
        "urllc": "hold",
        "iot": "hold",
    }

    cooldown_remaining = 0

    total_reconfiguration_cost = 0.0

    # --------------------------------------------------------
    # Optional Random Forest predictor (fails loudly if model missing)
    # --------------------------------------------------------

    rf_predictor = None

    if PREDICTOR == "rf":
        from controller.ml_predictor import RandomForestPredictor

        rf_predictor = RandomForestPredictor.load(MODEL_PATH)

        print(
            f"Prediction method: RANDOM FOREST "
            f"({rf_predictor.model_path})"
        )

    elif PREDICTOR != "ma":
        raise ValueError(
            f"Unknown PREDICTOR: {PREDICTOR!r} (use 'ma' or 'rf')"
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    # --------------------------------------------------------
    # Process each traffic measurement
    # --------------------------------------------------------

    for index, row in data.iterrows():

        time = row["time"]
        scenario = row["scenario"]

        demand = {
            "embb": float(row["embb_mbps"]),
            "urllc": float(row["urllc_mbps"]),
            "iot": float(row["iot_mbps"]),
        }

        # ----------------------------------------------------
        # Store traffic history
        # ----------------------------------------------------

        for slice_name in demand:

            demand_history[slice_name].append(
                demand[slice_name]
            )

        # ----------------------------------------------------
        # Short-term traffic prediction
        # ----------------------------------------------------

        prediction = {
            slice_name: predict_demand(
                demand_history[slice_name]
            )
            for slice_name in demand
        }

        # ----------------------------------------------------
        # Random Forest extension: replace the moving-average
        # prediction. The moving-average value is kept for logging.
        # ----------------------------------------------------

        ma_prediction = prediction

        if rf_predictor is not None:

            rf_result = rf_predictor.predict(demand_history)

            prediction = rf_result.values

        # ----------------------------------------------------
        # Controller demand
        #
        # Use the larger of:
        #   current demand
        #   predicted demand
        #
        # This makes the controller slightly proactive.
        # ----------------------------------------------------

        controller_demand = {
            slice_name: max(
                demand[slice_name],
                prediction[slice_name]
            )
            for slice_name in demand
        }
        demand_pressures = {
            slice_name: demand_pressure(
                demand[slice_name],
                slice_name,
            )
            for slice_name in demand
        }

        # ----------------------------------------------------
        # Calculate current utilization
        # ----------------------------------------------------

        utilization = calculate_utilization(
            controller_demand,
            allocation
        )

        # ----------------------------------------------------
        # Determine controller state using hysteresis
        # ----------------------------------------------------

        decisions = {}

        for slice_name in utilization:

            decisions[slice_name] = hysteresis_decision(
                utilization=utilization[slice_name],
                current_state=controller_state[slice_name],
                increase_threshold=INCREASE_THRESHOLD,
                decrease_threshold=DECREASE_THRESHOLD,
            )
            if demand_pressures[slice_name] > 1.0:
                decisions[slice_name] = "increase"

            controller_state[slice_name] = decisions[
                slice_name
            ]

        # ----------------------------------------------------
        # Save old allocation
        # ----------------------------------------------------

        old_allocation = allocation.copy()

        overloaded_slice = None
        donor_slice = None
        reconfigured = False

        # ----------------------------------------------------
        # Cooldown logic
        #
        # If cooldown is active, do not reconfigure.
        # ----------------------------------------------------

        if cooldown_remaining > 0:

            cooldown_active = True

            cooldown_remaining -= 1

        else:

            cooldown_active = False

            # ------------------------------------------------
            # Find slices that want more resources
            # ------------------------------------------------

            increase_candidates = [
                name
                for name in decisions
                if decisions[name] == "increase"
            ]

            if increase_candidates:

                # Most overloaded slice gets priority.
                overloaded_slice = max(
                    increase_candidates,
                    key=lambda name: utilization[name]
                )

                # ------------------------------------------------
                # Perform one resource reallocation
                # ------------------------------------------------

                new_allocation, changed, overloaded, donor = (
                    dynamic_reallocation(
                        controller_demand,
                        allocation,
                        demand_pressures={
                            slice_name: demand_pressure(
                                controller_demand[slice_name],
                                slice_name,
                            )
                            for slice_name in controller_demand
                        },
                    )
                )

                if changed:

                    allocation = new_allocation

                    overloaded_slice = overloaded
                    donor_slice = donor

                    reconfigured = True

                    # Start cooldown.
                    cooldown_remaining = COOLDOWN

                    # Calculate resource movement.
                    reconfiguration_cost = (
                        calculate_reconfiguration_cost(
                            old_allocation,
                            allocation
                        )
                    )

                    total_reconfiguration_cost += (
                        reconfiguration_cost
                    )

                else:

                    reconfiguration_cost = 0.0

            else:

                reconfiguration_cost = 0.0

        # ----------------------------------------------------
        # If cooldown was active, no reconfiguration happened.
        # ----------------------------------------------------

        if not reconfigured:

            reconfiguration_cost = 0.0

        # ----------------------------------------------------
        # Calculate actual QoS after allocation
        # ----------------------------------------------------

        qos_results = {}

        for slice_name in demand:

            qos_results[slice_name] = calculate_slice_qos(
                demand[slice_name],
                allocation[slice_name],
                slice_name,
            )

        # ----------------------------------------------------
        # Fairness
        # ----------------------------------------------------

        fairness = fairness_index(
            allocation
        )

        # ----------------------------------------------------
        # Save allocation history
        # ----------------------------------------------------

        allocation_history.append(
            allocation.copy()
        )

        # ----------------------------------------------------
        # Store experiment result
        # ----------------------------------------------------

        result = {
            "time": time,
            "scenario": scenario,

            # Demand
            "embb_demand_mbps": demand["embb"],
            "urllc_demand_mbps": demand["urllc"],
            "iot_demand_mbps": demand["iot"],

            # Prediction
            "embb_prediction_mbps": prediction["embb"],
            "urllc_prediction_mbps": prediction["urllc"],
            "iot_prediction_mbps": prediction["iot"],

            # Allocation
            "embb_allocation_pct": allocation["embb"],
            "urllc_allocation_pct": allocation["urllc"],
            "iot_allocation_pct": allocation["iot"],

            # Utilization
            "embb_utilization": qos_results["embb"]["utilization"],
            "urllc_utilization": qos_results["urllc"]["utilization"],
            "iot_utilization": qos_results["iot"]["utilization"],

            # Calibrated demand pressure (separate from QoS violation)
            "embb_demand_pressure": demand_pressures["embb"],
            "urllc_demand_pressure": demand_pressures["urllc"],
            "iot_demand_pressure": demand_pressures["iot"],
            "embb_demand_threshold_exceeded": qos_results["embb"][
                "demand_threshold_exceeded"
            ],
            "urllc_demand_threshold_exceeded": qos_results["urllc"][
                "demand_threshold_exceeded"
            ],
            "iot_demand_threshold_exceeded": qos_results["iot"][
                "demand_threshold_exceeded"
            ],

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

            # Controller state
            "embb_decision": decisions["embb"],
            "urllc_decision": decisions["urllc"],
            "iot_decision": decisions["iot"],

            # Stability
            "cooldown_active": cooldown_active,

            # Reconfiguration
            "reconfigured": reconfigured,
            "overloaded_slice": overloaded_slice,
            "donor_slice": donor_slice,
            "reconfiguration_cost": reconfiguration_cost,

            # Fairness
            "fairness": fairness,
        }

        if rf_predictor is not None:

            result["prediction_source"] = "rf"

            for slice_name in demand:

                result[f"{slice_name}_ma_prediction_mbps"] = (
                    ma_prediction[slice_name]
                )
                result[f"{slice_name}_rf_raw_mbps"] = (
                    rf_result.raw[slice_name]
                )
                result[f"{slice_name}_rf_flag"] = (
                    rf_result.flags[slice_name]
                )

        results.append(result)

    # ========================================================
    # Create results dataframe
    # ========================================================

    results_df = pd.DataFrame(results)

    # ========================================================
    # Calculate total statistics
    # ========================================================

    reconfiguration_count = int(
        results_df["reconfigured"].sum()
    )

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
    # Oscillation calculation
    # ========================================================

    oscillation_count = detect_oscillations(
        allocation_history
    )

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

    if rf_predictor is not None:

        rf_predictor.write_run_info(
            OUTPUT_FILE + ".run_info.json"
        )

    # ========================================================
    # Print summary
    # ========================================================

    print("\nExperiment completed.")
    print("-" * 60)

    print(f"Rows processed: {len(results_df)}")

    print(
        f"Reconfigurations: "
        f"{reconfiguration_count}"
    )

    print(
        f"Total reconfiguration cost: "
        f"{total_reconfiguration_cost:.2f}%"
    )

    print(
        f"Oscillation count: "
        f"{oscillation_count}"
    )

    print(
        f"Average fairness: "
        f"{results_df['fairness'].mean():.4f}"
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


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Dynamic slicing experiment."
    )
    parser.add_argument(
        "--predictor",
        choices=["ma", "rf"],
        default="ma",
        help="ma = original moving average (default); "
             "rf = Random Forest",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    cli = parser.parse_args()

    PREDICTOR = cli.predictor
    MODEL_PATH = cli.model

    if cli.input:
        INPUT_FILE = cli.input

    if cli.output:
        OUTPUT_FILE = cli.output
    elif PREDICTOR == "rf":
        # Never overwrite the original moving-average results.
        OUTPUT_FILE = "data/processed/dynamic_rf_results.csv"

    main()