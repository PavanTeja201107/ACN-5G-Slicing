import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Slice configuration
# ---------------------------------------------------------

SLICE_CONFIG = {
    "embb": {
        "base_latency_ms": 20,
        "latency_target_ms": 50,
    },
    "urllc": {
        "base_latency_ms": 5,
        "latency_target_ms": 20,
    },
    "iot": {
        "base_latency_ms": 30,
        "latency_target_ms": 100,
    },
}


def calculate_slice_qos(
    demand_mbps,
    allocated_mbps,
    slice_name,
):
    """
    Calculate QoS metrics for one slice.

    Parameters
    ----------
    demand_mbps : float
        Traffic requested by the slice.

    allocated_mbps : float
        Resource capacity currently allocated to the slice.

    slice_name : str
        Slice name: embb, urllc, or iot.

    Returns
    -------
    dict
        Calculated QoS metrics.
    """

    config = SLICE_CONFIG[slice_name]

    # Avoid division by zero
    if allocated_mbps <= 0:
        utilization = 0.0
        throughput = 0.0
        packet_loss = 1.0
        latency = config["base_latency_ms"] + 200
    else:
        utilization = demand_mbps / allocated_mbps

        # Traffic that can actually be served
        throughput = min(
            demand_mbps,
            allocated_mbps
        )

        # Traffic that cannot be served
        if demand_mbps > 0:
            packet_loss = max(
                demand_mbps - allocated_mbps,
                0
            ) / demand_mbps
        else:
            packet_loss = 0.0

        # Simple simulated latency model
        if utilization <= 1:
            latency = config["base_latency_ms"]
        else:
            overload = utilization - 1

            latency = (
                config["base_latency_ms"]
                + 100 * overload
            )

    # QoS violation
    qos_violation = (
        latency > config["latency_target_ms"]
        or packet_loss > 0.01
    )

    return {
        "demand_mbps": demand_mbps,
        "allocated_mbps": allocated_mbps,
        "throughput_mbps": throughput,
        "packet_loss": packet_loss,
        "latency_ms": latency,
        "utilization": utilization,
        "qos_violation": qos_violation,
    }


def evaluate_dataset(
    data,
    allocation,
    total_capacity=100,
):
    """
    Evaluate QoS for the complete traffic dataset.

    allocation contains percentages for each slice.

    Example:
        {
            "embb": 50,
            "urllc": 30,
            "iot": 20
        }
    """

    results = []

    # Convert percentages to Mbps
    allocated_capacity = {
        slice_name: total_capacity * percentage / 100
        for slice_name, percentage in allocation.items()
    }

    for _, row in data.iterrows():

        embb_result = calculate_slice_qos(
            row["embb_mbps"],
            allocated_capacity["embb"],
            "embb",
        )

        urllc_result = calculate_slice_qos(
            row["urllc_mbps"],
            allocated_capacity["urllc"],
            "urllc",
        )

        iot_result = calculate_slice_qos(
            row["iot_mbps"],
            allocated_capacity["iot"],
            "iot",
        )

        result = {
            "time": row["time"],
            "scenario": row["scenario"],

            "embb_demand_mbps": embb_result["demand_mbps"],
            "embb_throughput_mbps": embb_result["throughput_mbps"],
            "embb_latency_ms": embb_result["latency_ms"],
            "embb_packet_loss": embb_result["packet_loss"],
            "embb_utilization": embb_result["utilization"],
            "embb_qos_violation": embb_result["qos_violation"],

            "urllc_demand_mbps": urllc_result["demand_mbps"],
            "urllc_throughput_mbps": urllc_result["throughput_mbps"],
            "urllc_latency_ms": urllc_result["latency_ms"],
            "urllc_packet_loss": urllc_result["packet_loss"],
            "urllc_utilization": urllc_result["utilization"],
            "urllc_qos_violation": urllc_result["qos_violation"],

            "iot_demand_mbps": iot_result["demand_mbps"],
            "iot_throughput_mbps": iot_result["throughput_mbps"],
            "iot_latency_ms": iot_result["latency_ms"],
            "iot_packet_loss": iot_result["packet_loss"],
            "iot_utilization": iot_result["utilization"],
            "iot_qos_violation": iot_result["qos_violation"],
        }

        results.append(result)

    return pd.DataFrame(results)