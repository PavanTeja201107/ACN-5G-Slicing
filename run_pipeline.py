"""
Run the whole extension end to end, in order, from the project root.

    python run_pipeline.py            # full run
    python run_pipeline.py --quick    # smoke test: tiny dataset/model, 2 seeds (~1 min)

--quick writes the model to models/quick/ and does NOT touch the real
model, but it does regenerate the shared result CSVs/plots. Use the full
run for the numbers you report.
"""

import argparse
import subprocess
import sys


def run(args):
    print("\n$ python " + " ".join(args), flush=True)
    subprocess.run([sys.executable] + args, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seeds", type=int, default=None)
    a = p.parse_args()

    model = ["--model-out", "models/quick/rf.joblib"] if a.quick else []
    model_in = ["--model", "models/quick/rf.joblib"] if a.quick else []
    seeds = a.seeds or (2 if a.quick else 10)

    run(["-m", "ml.generate_dataset"] + (["--n-train", "3", "--n-val", "1", "--n-test", "1"] if a.quick else []))
    run(["-m", "ml.inspect_dataset"])
    run(["-m", "ml.train_rf"] + model + (["--n-estimators", "15"] if a.quick else []))
    run(["-m", "ml.evaluate_predictors"] + model_in)
    run(["-m", "experiments.static"])
    run(["-m", "experiments.dynamic"])
    run(["-m", "experiments.dynamic", "--predictor", "rf"] + model_in)
    run(["-m", "experiments.compare_three"])
    run(["-m", "experiments.run_multiseed", "--n-seeds", str(seeds)] + model_in)
    print("\nPipeline finished. See README.md section 'Output files'.")


if __name__ == "__main__":
    main()
