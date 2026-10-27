"""
Generate the (synthetic) multi-run traffic dataset for RF training.

    python -m ml.generate_dataset
    python -m ml.generate_dataset --n-train 16 --n-val 4 --n-test 8

Output: data/ml/ml_traffic_runs.csv  (+ dataset_manifest.json)
"""

import argparse
import json
import os

from ml import dataset_generator as dg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=16)
    parser.add_argument("--n-val", type=int, default=4)
    parser.add_argument("--n-test", type=int, default=8)
    parser.add_argument("--output", default=os.path.join("data", "ml", "ml_traffic_runs.csv"))
    args = parser.parse_args()

    runs = dg.generate_runs(args.n_train, args.n_val, args.n_test)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    runs.to_csv(args.output, index=False)

    manifest = {
        "description": (
            "SYNTHETIC traffic runs derived from the project's modelled "
            "scenarios. Not measurements from a real 5G network."
        ),
        "file": args.output,
        "rows": int(len(runs)),
        "runs": {
            split: sorted(runs.loc[runs["split"] == split, "seed"].unique().tolist())
            for split in ("train", "val", "test")
        },
        "scenarios": runs["scenario"].unique().tolist(),
        "perturbation": {
            "time_shift": "circular, per scenario block",
            "scale_range": list(dg.SCALE_RANGE),
            "extra_noise_scale": dg.NOISE_SCALE,
            "floor": dg.FLOOR,
        },
        "columns": list(runs.columns),
        "unit": "Mbps",
    }

    manifest_path = os.path.join(os.path.dirname(args.output), "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("Synthetic traffic runs generated.")
    print(f"Rows: {len(runs)}")
    print(runs.groupby("split")["run_id"].nunique().to_string())
    print(f"Saved to: {args.output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
