import json

import pandas as pd
import pytest

from controller.controller import (
    MAX_ALLOCATION,
    MIN_ALLOCATION,
    dynamic_reallocation,
    validate_allocation,
)
from controller.qos_monitor import calculate_slice_qos
from controller.thresholds import (
    CONFIG_PATH,
    demand_pressure,
    load_threshold_config,
)
from experiments import dynamic


def test_central_configuration_contains_distinct_slice_thresholds():
    config = load_threshold_config()
    assert config["slices"]["embb"]["demand_threshold_mbps"] == pytest.approx(
        70.7370937069
    )
    assert config["slices"]["urllc"]["latency_threshold_ms"] == pytest.approx(
        50.3883216374
    )
    assert config["slices"]["iot"]["packet_loss_threshold"] == 0.0
    assert CONFIG_PATH.exists()


@pytest.mark.parametrize(
    ("slice_name", "demand", "expected"),
    [
        ("embb", 70.7370937069, 1.0),
        ("urllc", 2 * 43.6164964912, 2.0),
        ("iot", 0.0, 0.0),
    ],
)
def test_demand_pressure_uses_slice_specific_thresholds(
    slice_name, demand, expected
):
    assert demand_pressure(demand, slice_name) == pytest.approx(expected)


def test_demand_threshold_exceedance_is_separate_from_qos_violation():
    result = calculate_slice_qos(80.0, 100.0, "embb")
    assert result["demand_threshold_exceeded"] is True
    assert result["qos_violation"] is False


def test_zero_demand_and_zero_allocation_are_safe():
    result = calculate_slice_qos(0.0, 0.0, "iot")
    assert result["throughput_mbps"] == 0.0
    assert result["packet_loss"] == 1.0
    assert result["demand_pressure"] == 0.0
    assert result["qos_violation"] is True


def test_demand_pressure_can_trigger_reallocation():
    allocation = {"embb": 50.0, "urllc": 30.0, "iot": 20.0}
    demand = {"embb": 75.0, "urllc": 10.0, "iot": 5.0}
    pressures = {"embb": 75.0 / 70.7370937069, "urllc": 10.0 / 43.6164964912, "iot": 5.0 / 14.2901527569}
    updated, changed, overloaded, donor = dynamic_reallocation(
        demand,
        allocation,
        demand_pressures=pressures,
    )
    assert changed is True
    assert overloaded == "embb"
    assert donor == "iot"
    assert sum(updated.values()) == pytest.approx(100.0)
    assert updated["embb"] <= 70.0
    assert updated["urllc"] >= 20.0
    assert updated["iot"] >= 10.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_threshold_configuration_is_rejected(tmp_path, value):
    path = tmp_path / "thresholds.json"
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["slices"]["embb"]["latency_threshold_ms"] = value
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="slices\\.embb\\.latency_threshold_ms.*finite",
    ):
        load_threshold_config(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("demand_threshold_mbps", 0),
        ("latency_threshold_ms", -1),
        ("packet_loss_threshold", 1.1),
        ("packet_loss_threshold", "not-a-number"),
    ],
)
def test_invalid_threshold_values_identify_field(tmp_path, field, value):
    path = tmp_path / "thresholds.json"
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["slices"]["urllc"][field] = value
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match=f"slices\\.urllc\\.{field}"):
        load_threshold_config(path)


def test_invalid_threshold_configuration_is_rejected(tmp_path):
    path = tmp_path / "thresholds.json"
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    del config["slices"]["iot"]
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="embb, urllc, and iot"):
        load_threshold_config(path)


def write_config(tmp_path, **updates):
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config.update(updates)
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_valid_metadata_is_accepted(tmp_path):
    path = write_config(
        tmp_path,
        quantile=1.0,
        total_capacity_mbps=100.0,
        reference_allocation_percent={
            "embb": 50.0,
            "urllc": 30.0,
            "iot": 20.0,
        },
    )
    config = load_threshold_config(path)
    assert config["quantile"] == 1.0


def test_optional_metadata_may_be_absent(tmp_path):
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for key in (
        "quantile",
        "total_capacity_mbps",
        "reference_allocation_pct",
    ):
        del config[key]
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert "slices" in load_threshold_config(path)


@pytest.mark.parametrize("value", [0, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_quantile_is_rejected(tmp_path, value):
    path = write_config(tmp_path, quantile=value)
    with pytest.raises(ValueError, match="quantile"):
        load_threshold_config(path)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("-inf")])
def test_invalid_capacity_is_rejected(tmp_path, value):
    path = write_config(tmp_path, total_capacity_mbps=value)
    with pytest.raises(ValueError, match="total_capacity_mbps"):
        load_threshold_config(path)


@pytest.mark.parametrize(
    "allocation",
    [
        {"embb": 50.0, "urllc": 30.0},
        {"embb": 50.0, "urllc": 30.0, "iot": -1.0},
        {"embb": float("nan"), "urllc": 30.0, "iot": 20.0},
        {"embb": 50.0, "urllc": 30.0, "iot": 30.0},
    ],
)
def test_invalid_reference_allocation_is_rejected(tmp_path, allocation):
    path = write_config(tmp_path, reference_allocation_pct=allocation)
    with pytest.raises(ValueError, match="reference_allocation_pct"):
        load_threshold_config(path)


def test_invalid_reference_allocation_percent_alias_is_rejected(tmp_path):
    path = write_config(
        tmp_path,
        reference_allocation_percent={
            "embb": 50.0,
            "urllc": 30.0,
            "iot": 30.0,
        },
    )
    with pytest.raises(ValueError, match="reference_allocation_percent"):
        load_threshold_config(path)


@pytest.mark.parametrize(
    "allocation",
    [
        {"embb": 50.0, "urllc": 30.0},
        {"embb": float("nan"), "urllc": 30.0, "iot": 20.0},
        {"embb": 10.0, "urllc": 50.0, "iot": 40.0},
        {"embb": 70.0, "urllc": 20.0, "iot": 20.0},
        {"embb": 50.0, "urllc": 30.0, "iot": 30.0},
    ],
)
def test_malformed_allocations_are_rejected(allocation):
    with pytest.raises(ValueError, match="Allocation"):
        validate_allocation(allocation)


def test_valid_reallocation_preserves_allocation_invariants():
    allocation = {"embb": 50.0, "urllc": 30.0, "iot": 20.0}
    demand = {"embb": 75.0, "urllc": 10.0, "iot": 5.0}
    updated, changed, _, _ = dynamic_reallocation(demand, allocation)
    assert changed is True
    validate_allocation(updated)
    assert sum(updated.values()) == pytest.approx(100.0)
    assert all(
        MIN_ALLOCATION[name] <= updated[name] <= MAX_ALLOCATION[name]
        for name in updated
    )


def test_no_valid_donor_does_not_reallocate():
    allocation = {"embb": 70.0, "urllc": 20.0, "iot": 10.0}
    demand = {"embb": 100.0, "urllc": 100.0, "iot": 100.0}
    updated, changed, overloaded, donor = dynamic_reallocation(demand, allocation)
    assert updated == allocation
    assert changed is False
    assert overloaded in allocation
    assert donor is None


def test_cooldown_blocks_then_allows_reallocation(tmp_path, monkeypatch):
    rows = pd.DataFrame(
        [
            {
                "time": index,
                "scenario": "cooldown",
                "embb_mbps": 80.0,
                "urllc_mbps": 1.0,
                "iot_mbps": 1.0,
            }
            for index in range(12)
        ]
    )
    input_path = tmp_path / "traffic.csv"
    output_path = tmp_path / "dynamic.csv"
    rows.to_csv(input_path, index=False)
    monkeypatch.setattr(dynamic, "INPUT_FILE", str(input_path))
    monkeypatch.setattr(dynamic, "OUTPUT_FILE", str(output_path))
    monkeypatch.setattr(dynamic, "PREDICTOR", "ma")

    dynamic.main()
    result = pd.read_csv(output_path)

    assert result.index[result["reconfigured"]].tolist() == [0, 11]
    assert result.loc[0, "cooldown_active"] == False
    assert result.loc[1:10, "cooldown_active"].all()
    assert result.loc[11, "cooldown_active"] == False
