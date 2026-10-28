#!/usr/bin/env python3
"""Calibrate empirical traffic/QoS thresholds from a CSV.

Required columns: time, embb_mbps, urllc_mbps, iot_mbps, scenario

IMPORTANT: This input has demand only. Latency and packet loss are calculated
using the simplified model from controller/qos_monitor.py. The resulting
QoS thresholds are model-calibrated descriptive candidates, NOT real 5G SLAs.
Run from repository root:
  python analysis/qos_threshold_calibration.py path/to/traffic.csv
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

SLICES = {
    "embb": {"column": "embb_mbps", "base_latency_ms": 20.0, "old_target_ms": 50.0},
    "urllc": {"column": "urllc_mbps", "base_latency_ms": 5.0, "old_target_ms": 20.0},
    "iot": {"column": "iot_mbps", "base_latency_ms": 30.0, "old_target_ms": 100.0},
}
REQUIRED = {"time", "embb_mbps", "urllc_mbps", "iot_mbps", "scenario"}


def summary(series, quantiles):
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return {"count": 0}
    result = {
        "count": int(x.size), "mean": float(x.mean()),
        "median": float(x.median()),
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "min": float(x.min()), "max": float(x.max()),
    }
    for q in quantiles:
        result[f"p{int(round(q * 100)):02d}"] = float(x.quantile(q))
    return result


def model_qos(demand, allocated, base_latency):
    # Mirrors controller/qos_monitor.py
    if allocated <= 0:
        return {"throughput_mbps": 0.0, "packet_loss": 1.0,
                "latency_ms": base_latency + 200.0, "utilization": 0.0}
    utilization = demand / allocated
    throughput = min(demand, allocated)
    loss = max(demand - allocated, 0.0) / demand if demand > 0 else 0.0
    latency = base_latency if utilization <= 1 else base_latency + 100 * (utilization - 1)
    return {"throughput_mbps": float(throughput), "packet_loss": float(loss),
            "latency_ms": float(latency), "utilization": float(utilization)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--quantile", type=float, default=0.95,
                        help="Candidate threshold quantile: e.g. 0.90, 0.95, 0.99")
    parser.add_argument("--capacity-mbps", type=float, default=100.0)
    parser.add_argument("--allocation", nargs=3, default=["embb=50", "urllc=30", "iot=20"],
                        metavar=("EMBB=PCT", "URLLC=PCT", "IOT=PCT"))
    parser.add_argument("--output-dir", default="results/threshold_calibration")
    args = parser.parse_args()

    if not 0.5 < args.quantile < 1:
        parser.error("--quantile must be > 0.5 and < 1")
    if args.capacity_mbps <= 0:
        parser.error("--capacity-mbps must be positive")

    allocation_pct = {}
    try:
        for item in args.allocation:
            name, val = item.split("=", 1)
            allocation_pct[name.lower()] = float(val)
    except ValueError:
        parser.error("Allocation format must be slice=percentage")
    if set(allocation_pct) != set(SLICES) or any(v <= 0 for v in allocation_pct.values()):
        parser.error("Provide positive allocations for embb, urllc, and iot")
    if not np.isclose(sum(allocation_pct.values()), 100):
        parser.error("Allocation percentages must sum to 100")

    path = Path(args.csv_path)
    if not path.exists():
        parser.error(f"CSV not found: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        parser.error(f"Missing required columns: {sorted(missing)}")
    for spec in SLICES.values():
        df[spec["column"]] = pd.to_numeric(df[spec["column"]], errors="coerce")
    df = df.dropna(subset=[s["column"] for s in SLICES.values()] + ["scenario"]).copy()
    if df.empty:
        parser.error("No valid data rows")
    if (df[[s["column"] for s in SLICES.values()]] < 0).any().any():
        parser.error("Demand values must be non-negative")

    q = args.quantile
    quantiles = sorted(set([0.50, 0.90, 0.95, 0.99, q]))
    demand_rows = []
    groups = [("ALL_SCENARIOS", df)] + list(df.groupby("scenario", sort=True))
    for scenario, group in groups:
        for name, spec in SLICES.items():
            demand_rows.append({"scenario": scenario, "slice": name,
                                **summary(group[spec["column"]], quantiles)})
    demand_stats = pd.DataFrame(demand_rows)

    allocated = {name: args.capacity_mbps * pct / 100
                 for name, pct in allocation_pct.items()}
    records = []
    for _, row in df.iterrows():
        for name, spec in SLICES.items():
            demand = float(row[spec["column"]])
            metrics = model_qos(demand, allocated[name], spec["base_latency_ms"])
            records.append({"time": row["time"], "scenario": row["scenario"],
                            "slice": name, "demand_mbps": demand,
                            "allocated_mbps": allocated[name], **metrics})
    qos = pd.DataFrame(records)

    qos_rows = []
    for scenario, group in [("ALL_SCENARIOS", qos)] + list(qos.groupby("scenario", sort=True)):
        for name in SLICES:
            part = group[group["slice"] == name]
            for metric in ["latency_ms", "packet_loss", "throughput_mbps", "utilization"]:
                qos_rows.append({"scenario": scenario, "slice": name, "metric": metric,
                                 **summary(part[metric], quantiles)})
    qos_stats = pd.DataFrame(qos_rows)

    thresholds = {
        "metadata": {
            "input_csv": str(path), "valid_rows": int(len(df)),
            "quantile": q, "total_capacity_mbps": args.capacity_mbps,
            "reference_allocation_percent": allocation_pct,
            "method": "empirical quantiles; QoS metrics generated by existing simplified model",
            "warning": (
                "Traffic CSV contains demand only. Latency/loss thresholds are descriptive "
                "quantiles of the simplified QoS model under the reference allocation, not "
                "independent application SLAs or measured 5G guarantees. Validate on held-out "
                "seeds/scenarios and later replace model outputs with testbed measurements."
            ),
        },
        "slices": {},
    }
    for name, spec in SLICES.items():
        part = qos[qos["slice"] == name]
        thresholds["slices"][name] = {
            "demand_threshold_mbps": float(df[spec["column"]].quantile(q)),
            "modelled_latency_threshold_ms": float(part["latency_ms"].quantile(q)),
            "modelled_packet_loss_threshold_ratio": float(part["packet_loss"].quantile(q)),
            "existing_hardcoded_latency_target_ms": spec["old_target_ms"],
            "reference_allocated_capacity_mbps": allocated[name],
            "note": f"p{int(round(q*100))} descriptive candidate; not a guaranteed SLA",
        }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "qos_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    demand_stats.to_csv(out / "demand_statistics.csv", index=False)
    qos_stats.to_csv(out / "modelled_qos_statistics.csv", index=False)

    print(f"Analyzed {len(df)} valid rows; selected quantile p{int(round(q*100))}.")
    print(f"Reference allocation: {allocation_pct}; total capacity: {args.capacity_mbps} Mbps")
    for name, vals in thresholds["slices"].items():
        print(f"{name.upper():5s}: demand={vals['demand_threshold_mbps']:.3f} Mbps, "
              f"modelled latency={vals['modelled_latency_threshold_ms']:.3f} ms, "
              f"modelled loss={vals['modelled_packet_loss_threshold_ratio']:.5f}")
    print(f"Outputs saved to: {out.resolve()}")
    print("Reminder: latency/loss thresholds are model-calibrated candidates, not real 5G SLAs.")


if __name__ == "__main__":
    main()
