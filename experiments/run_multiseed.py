"""
Repeat Static / Dynamic-MA / Dynamic-RF on several independent traffic
streams (seeds never used for RF training) and report mean +/- std.

    python -m experiments.run_multiseed --n-seeds 10

Each seed produces one perturbed 6-scenario stream (see
ml/dataset_generator.py; SYNTHETIC). All three approaches get the SAME
stream per seed. Existing runners are used unchanged (their module-level
file paths are redirected; nothing in data/processed is overwritten).

Writes results/multiseed/per_seed_metrics.csv and summary.csv.
"""

import argparse
import contextlib
import io
import os

import numpy as np
import pandas as pd

from controller.ml_predictor import RandomForestPredictor
from experiments import dynamic, static
from experiments.metrics_extra import approach_metrics
from experiments.validate_results import validate
from ml import dataset_generator as dg

KEYS = [
    "total_qos_violations", "eMBB_avg_latency_ms", "URLLC_avg_latency_ms", "IoT_avg_latency_ms",
    "URLLC_p95_latency_ms", "eMBB_avg_throughput_mbps", "URLLC_avg_throughput_mbps",
    "eMBB_avg_packet_loss", "URLLC_avg_packet_loss", "IoT_avg_packet_loss",
    "average_fairness", "reconfigurations", "total_reconfiguration_cost_pct", "oscillations",
]


def run_one(stream_path, workdir, model_path):
    quiet = io.StringIO()
    out = {}
    with contextlib.redirect_stdout(quiet):
        static.INPUT_FILE, static.OUTPUT_FILE = stream_path, os.path.join(workdir, "static.csv")
        static.main()
        dynamic.INPUT_FILE = stream_path
        dynamic.OUTPUT_FILE, dynamic.PREDICTOR, dynamic.MODEL_PATH = os.path.join(workdir, "ma.csv"), "ma", None
        dynamic.main()
        dynamic.OUTPUT_FILE, dynamic.PREDICTOR, dynamic.MODEL_PATH = os.path.join(workdir, "rf.csv"), "rf", model_path
        dynamic.main()
    s, m, r = (pd.read_csv(os.path.join(workdir, f)) for f in ("static.csv", "ma.csv", "rf.csv"))
    problems = validate(s, m, r)
    if problems:
        raise RuntimeError("; ".join(problems))
    return {"Static": approach_metrics(s, False), "Dynamic-MA": approach_metrics(m), "Dynamic-RF": approach_metrics(r)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--model", default=None)
    p.add_argument("--out-dir", default=os.path.join("results", "multiseed"))
    args = p.parse_args()

    RandomForestPredictor.load(args.model)            # fail early with a clear message
    work = os.path.join(args.out_dir, "work")
    os.makedirs(work, exist_ok=True)

    canonical = dg.canonical_stream()
    rows = []
    for seed in dg.seeds_for("multiseed_eval", args.n_seeds):
        stream = os.path.join(work, f"traffic_seed{seed}.csv")
        dg.perturbed_run(seed, canonical).drop(columns="seed").to_csv(stream, index=False)
        res = run_one(stream, work, args.model)
        for approach, m in res.items():
            rows.append({"seed": seed, "approach": approach, **{k: m[k] for k in KEYS}})
        print(f"seed {seed}: violations  static={res['Static']['total_qos_violations']:.0f}  "
              f"MA={res['Dynamic-MA']['total_qos_violations']:.0f}  RF={res['Dynamic-RF']['total_qos_violations']:.0f}")

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(os.path.join(args.out_dir, "per_seed_metrics.csv"), index=False)

    summary = per_seed.groupby("approach")[KEYS].agg(["mean", "std"]).T
    summary.to_csv(os.path.join(args.out_dir, "summary.csv"))
    print("\nMean / std over seeds")
    print(summary.round(3).to_string())

    wide = per_seed.pivot(index="seed", columns="approach", values="total_qos_violations")
    print(f"\nRF fewer total QoS violations than MA in {(wide['Dynamic-RF'] < wide['Dynamic-MA']).sum()} / {len(wide)} seeds "
          f"(equal in {(wide['Dynamic-RF'] == wide['Dynamic-MA']).sum()}, more in {(wide['Dynamic-RF'] > wide['Dynamic-MA']).sum()}).")
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
