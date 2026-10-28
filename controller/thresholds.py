"""Central loading and evaluation of calibrated QoS thresholds."""

import json
import math
from numbers import Real
from pathlib import Path


CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "qos_thresholds.json"
)
SLICE_NAMES = ("embb", "urllc", "iot")
ALLOCATION_TOLERANCE = 1e-8


def _validate_numeric_metadata(config):
    """Validate optional numeric calibration metadata when present."""

    if "quantile" in config:
        value = config["quantile"]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("Threshold metadata quantile must be numeric")
        if not math.isfinite(value):
            raise ValueError("Threshold metadata quantile must be finite")
        if not 0 < value <= 1:
            raise ValueError(
                "Threshold metadata quantile must be in the interval (0, 1]"
            )

    if "total_capacity_mbps" in config:
        value = config["total_capacity_mbps"]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(
                "Threshold metadata total_capacity_mbps must be numeric"
            )
        if not math.isfinite(value):
            raise ValueError(
                "Threshold metadata total_capacity_mbps must be finite"
            )
        if value <= 0:
            raise ValueError(
                "Threshold metadata total_capacity_mbps must be positive"
            )

    for allocation_key in (
        "reference_allocation_pct",
        "reference_allocation_percent",
    ):
        if allocation_key not in config:
            continue
        allocation = config[allocation_key]
        if not isinstance(allocation, dict) or set(allocation) != set(SLICE_NAMES):
            raise ValueError(
                f"Threshold metadata {allocation_key} must define "
                "embb, urllc, and iot"
            )

        for name, value in allocation.items():
            field = f"{allocation_key}.{name}"
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"Threshold metadata {field} must be numeric")
            if not math.isfinite(value):
                raise ValueError(f"Threshold metadata {field} must be finite")
            if value < 0:
                raise ValueError(f"Threshold metadata {field} must be non-negative")

        total = sum(allocation.values())
        if not math.isclose(
            total,
            100.0,
            rel_tol=0.0,
            abs_tol=ALLOCATION_TOLERANCE,
        ):
            raise ValueError(
                f"Threshold metadata {allocation_key} must sum to 100%, got {total}"
            )


def load_threshold_config(path=CONFIG_PATH):
    """Load and validate the central threshold configuration."""

    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Unable to load QoS threshold configuration: {config_path}"
        ) from exc

    if not isinstance(config, dict):
        raise ValueError("Threshold configuration root must be an object")

    _validate_numeric_metadata(config)

    slices = config.get("slices")
    if not isinstance(slices, dict) or set(slices) != set(SLICE_NAMES):
        raise ValueError("Threshold configuration must define embb, urllc, and iot")

    for name in SLICE_NAMES:
        values = slices[name]
        if not isinstance(values, dict):
            raise ValueError(f"Threshold configuration for {name} must be an object")
        required = (
            "demand_threshold_mbps",
            "latency_threshold_ms",
            "packet_loss_threshold",
        )
        if any(key not in values for key in required):
            raise ValueError(f"Threshold configuration for {name} is incomplete")
        for field in required:
            value = values[field]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(
                    f"Threshold field slices.{name}.{field} must be numeric"
                )
            if not math.isfinite(value):
                raise ValueError(
                    f"Threshold field slices.{name}.{field} must be finite"
                )
        if values["demand_threshold_mbps"] <= 0:
            raise ValueError(
                f"Threshold field slices.{name}.demand_threshold_mbps "
                "must be positive"
            )
        if values["latency_threshold_ms"] <= 0:
            raise ValueError(
                f"Threshold field slices.{name}.latency_threshold_ms "
                "must be positive"
            )
        if not 0 <= values["packet_loss_threshold"] <= 1:
            raise ValueError(
                f"Threshold field slices.{name}.packet_loss_threshold "
                "must be a ratio between 0 and 1"
            )

    return config


THRESHOLD_CONFIG = load_threshold_config()


def slice_thresholds(slice_name):
    """Return calibrated thresholds for one known slice."""

    try:
        return THRESHOLD_CONFIG["slices"][slice_name]
    except KeyError as exc:
        raise ValueError(f"Unknown slice: {slice_name!r}") from exc


def demand_pressure(demand_mbps, slice_name):
    """Return demand divided by the calibrated p95 demand threshold."""

    threshold = float(slice_thresholds(slice_name)["demand_threshold_mbps"])
    return float(demand_mbps) / threshold


def demand_threshold_exceeded(demand_mbps, slice_name):
    """Return whether demand is above its calibrated p95 threshold."""

    return demand_pressure(demand_mbps, slice_name) > 1.0
