"""
Training-data generation for the Random Forest predictor.

IMPORTANT - what this data is
-----------------------------
All data here is SYNTHETIC. It is derived from the project's own traffic
scenarios (traffic/traffic_generator.py + traffic/traffic_profiles.py),
which are themselves modelled/simulated traffic. Nothing in this module
is a measurement from a real 5G network.

Why perturbation is needed
--------------------------
The original profiles use hard-coded noise seeds, so the project has
exactly ONE trace per scenario (traffic_dataset.csv, 6 x 180 rows). A
model trained and tested on that single trace would be evaluated on its
own training data. To obtain independent runs WITHOUT touching the
original generator, each run is built from the unmodified original
scenarios (traffic_generator.create_dataset()) and perturbed per seed:

  1. circular time shift of each scenario block (moves where bursts fall)
  2. per-scenario, per-slice amplitude scaling
  3. extra smooth noise from the project's own smooth_random_noise()
  4. floor values identical to the floors used in traffic_profiles.py

Each run keeps the original layout: six scenarios of 180 intervals
(1080 rows), concatenated in the original order, which is exactly how the
controller experiments consume the stream.

Limitation: every run shares the same six underlying scenario templates.
Held-out seeds therefore test robustness to perturbation, not to
completely unseen traffic types. ml/leave_scenario_out.py gives a
stricter check.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAFFIC_DIR = os.path.join(ROOT, "traffic")

if TRAFFIC_DIR not in sys.path:
    sys.path.insert(0, TRAFFIC_DIR)

import traffic_generator  # noqa: E402  (original, unmodified module)
from traffic_profiles import smooth_random_noise  # noqa: E402

SLICE_COLUMNS = ("embb_mbps", "urllc_mbps", "iot_mbps")

# Perturbation parameters (documented in README / dataset manifest).
SCALE_RANGE = (0.75, 1.30)
NOISE_SCALE = {"embb_mbps": 2.0, "urllc_mbps": 1.5, "iot_mbps": 0.6}
FLOOR = {"embb_mbps": 5.0, "urllc_mbps": 2.0, "iot_mbps": 1.0}

# Disjoint seed ranges so splits can never share a seed.
SEED_BASE = {
    "train": 1000,
    "val": 2000,
    "test": 3000,
    "multiseed_eval": 4000,
}


def canonical_stream():
    """The project's original traffic stream (unmodified, 1080 rows)."""
    return traffic_generator.create_dataset()


def perturbed_run(seed, canonical=None):
    """Build one independent, deterministic run for the given seed."""

    if canonical is None:
        canonical = canonical_stream()

    rng = np.random.default_rng(seed)
    blocks = []

    for scenario, block in canonical.groupby("scenario", sort=False):
        block = block.reset_index(drop=True).copy()
        length = len(block)

        shift = int(rng.integers(0, length))

        for column in SLICE_COLUMNS:
            values = np.roll(block[column].to_numpy(dtype=float), shift)
            values = values * rng.uniform(*SCALE_RANGE)

            noise_seed = int(rng.integers(0, 2**31 - 1))
            values = values + smooth_random_noise(
                length,
                scale=NOISE_SCALE[column],
                seed=noise_seed,
            )

            block[column] = np.maximum(values, FLOOR[column])

        blocks.append(block)

    run = pd.concat(blocks, ignore_index=True)
    run.insert(0, "seed", seed)
    return run


def seeds_for(split, count):
    return [SEED_BASE[split] + i + 1 for i in range(count)]


def generate_runs(n_train, n_val, n_test):
    """Generate all runs. Returns a long dataframe with run_id/split."""

    canonical = canonical_stream()
    frames = []

    for split, count in (
        ("train", n_train),
        ("val", n_val),
        ("test", n_test),
    ):
        for seed in seeds_for(split, count):
            run = perturbed_run(seed, canonical)
            run.insert(0, "split", split)
            run.insert(0, "run_id", f"{split}_{seed}")
            frames.append(run)

    return pd.concat(frames, ignore_index=True)
