# Lightweight Predictive and Reconfiguration-Aware QoS Management for 5G Network Slicing

ACN (IT365) course project. This repository contains

1. the **mid-semester implementation** (traffic generation, QoS monitor, static 50/30/20 baseline,
   dynamic controller with a moving-average predictor, fairness, hysteresis, cooldown,
   reconfiguration cost), and
2. the **end-semester extension**: a **Random Forest** short-term traffic-demand predictor that can
   replace the moving-average predictor inside the *same* controller, so the two prediction methods
   can be compared fairly.

> **Scope and honesty note.** Everything here is a **Python simulation with modelled traffic**.
> Slice "allocations" are **modelled percentages of a 100 Mbps capacity**, not real radio resource
> blocks. The Open5GS/UERANSIM testbed described in the proposal is **not** used by, or tested with,
> any code in this repository. Nothing below is a measurement from a real 5G network.

---------------------------------------------------------------------------------------------------

## 1. Contents

| Part | Where |
|---|---|
| Traffic profiles / generator (original) | `traffic/` |
| QoS model, controller, fairness, hysteresis, reconfiguration cost (original) | `controller/controller.py`, `qos_monitor.py`, `fairness.py`, `stability.py`, `reconfiguration.py` |
| Static and dynamic experiments, 2-way comparison (original) | `experiments/static.py`, `dynamic.py`, `compare.py` |
| **RF features + inference (new)** | `controller/ml_features.py`, `controller/ml_predictor.py` |
| **Dataset / training / evaluation (new)** | `ml/` |
| **3-way comparison, multi-seed runner, result validator (new)** | `experiments/compare_three.py`, `run_multiseed.py`, `validate_results.py`, `metrics_extra.py` |
| **Tests (new)** | `tests/` |
| One-command pipeline | `run_pipeline.py` |
| Change summary / test report / presentation guide | `docs/` |

## 2. Naming note: "EWMA" vs what the code really does

The proposal text says "moving average or EWMA". The implemented predictor
(`predict_demand` in `experiments/dynamic.py`) is a **5-sample simple moving average that includes the
current sample**. This repo therefore calls it **MA** ("Dynamic-MA"). It is a smoothing of recent
demand, not a true forecast; it lags behind bursts. The controller then uses
`max(current demand, prediction)` as its input - this rule is **unchanged** for the RF variant.

## 3. The three approaches

| | Approach | Prediction | Allocation |
|---|---|---|---|
| A | **Static** | none | fixed 50/30/20 (`experiments/static.py`, untouched) |
| B | **Dynamic-MA** | existing moving average (window 5) | existing controller (`python -m experiments.dynamic`) |
| C | **Dynamic-RF** | Random Forest, demand at t+1 | **same** controller, same thresholds (0.80/0.50), step 5, cooldown 10, min/max shares |

The RF predicts **traffic demand (Mbps), never allocation**. The controller still decides allocation, so
fairness limits (min 20/20/10 %, max 70/60/40 %), hysteresis, cooldown and the reconfiguration-cost
model all apply exactly as before.

## 4. The ML formulation

* **Target:** demand of the slice at the next observation interval, `demand[t+1]` (Mbps). Horizon is
  configurable (`--horizon`).
* **Model:** one `sklearn.ensemble.RandomForestRegressor` (default 100 trees, `min_samples_leaf=20`,
  `max_features=0.5`, seed 42, CPU, single thread) shared by all three slices; the slice is an input.
* **Internal target trick:** the forest is trained to predict `demand[t+1] - demand[t]` and the current
  demand is added back, so its *output* is still next-interval demand in Mbps (`--target-mode absolute`
  trains on the raw value). `delta` was chosen using the **validation** runs only
  (val MAE 0.848 vs 0.933 Mbps for absolute).
* **Features (23, defined once in `controller/ml_features.py`, used identically in training and inference):**
  slice one-hot (3); own demand lags `lag_0 ... lag_9` (10); `ma5`, `ma10`, `ewma(alpha 0.5)`, `std5`,
  `diff1`, `diff5`, `range5` (7); current demand of all three slices (3).
  Before 10 samples exist the oldest value is repeated (edge padding).
* **Features deliberately NOT used:** current allocation, utilization, latency, packet loss, client
  counts. In this simulator offered demand is **exogenous** (it does not depend on allocation), so these
  carry no causal information about next demand, differ between controllers (train/inference
  distribution shift), and "active clients" does not exist in the data. Only demand history is used.
* **No leakage:** features at time t use `demand[0..t]` only; target is `demand[t+1]`; the last
  `horizon` rows of each trace have no target and are dropped; windows never cross runs; train / val /
  test are different **seeds** (disjoint seed ranges 1001+, 2001+, 3001+; multi-seed experiments use
  4001+). Verified by tests.

### The dataset is synthetic - and why it is built this way
The original generator has hard-coded noise seeds, so it yields exactly **one** trace per scenario
(`data/raw/traffic_dataset.csv`, 6 scenarios x 180 intervals = 1080 rows). Training and testing on that
one trace would test on training data. Without changing the original generator, `ml/dataset_generator.py`
takes the **original six scenarios** (normal, embb_burst, urllc_spike, iot_burst, competition,
fluctuation) and creates independent runs per seed by: circular time-shift per scenario, per-scenario
per-slice amplitude scaling (x0.75-1.30), extra smooth noise (project's own `smooth_random_noise`), and
the original floors. Default: 16 train + 4 val + 8 test runs (30,240 rows).
**Limitation:** all runs share the same six scenario templates.

## 5. Setup (from a fresh machine)

Requirements: Python 3.9+ (tested only on **Python 3.12.3** here). CPU only, no GPU.

**Linux / macOS**
```bash
unzip ACN-5G-Slicing-main-RF.zip && cd ACN-5G-Slicing-main
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-ml.txt
```
**Windows PowerShell** (not tested by the author of this extension)
```powershell
Expand-Archive ACN-5G-Slicing-main-RF.zip .; cd ACN-5G-Slicing-main
py -m venv .venv
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install --upgrade pip
pip install -r requirements-ml.txt
```
**All commands below are run from the project root** (the folder containing `controller/`, `ml/`, ...)
with the virtual environment active. Use `python` (Windows: `python`/`py`).

**Validate the setup**
```bash
python -m pytest tests -q      # expect: all repository tests to pass
```

**Launch the results dashboard**
```bash
streamlit run streamlit_app.py
```
The dashboard visualizes the seeded synthetic traffic data, Random Forest versus
moving-average versus persistence accuracy, the Static/Dynamic-MA/Dynamic-RF
comparison, allocation traces, multi-seed results, and the RF model/run metadata.
Install `requirements-ml.txt` first. It reads the generated CSV and JSON artifacts
from the repository and does not retrain the model or rerun experiments.

The dashboard also defines the comparison direction for each metric: lower is
better for MAE, RMSE, latency, packet loss, QoS violations, reconfigurations,
reconfiguration cost, and oscillations; higher is better for throughput and
fairness. Reconfiguration metrics apply only to Dynamic-MA and Dynamic-RF because
Static never reallocates. The overall dashboard recommendation uses the lowest
mean total QoS violations across the ten independent seeds and displays the
underlying values and exceptions rather than changing experiment results.

## 6. Run everything, in order

**Shortcut:** `python run_pipeline.py` runs steps 1-9 below (about 4 minutes; 214 s measured);
`python run_pipeline.py --quick` is a ~1-minute smoke test using a tiny model (it overwrites the shared
result files - re-run the full pipeline afterwards). The step-by-step version:

| # | Command | What it does / what to check |
|---|---|---|
| 1 | `python -m ml.generate_dataset` | Writes `data/ml/ml_traffic_runs.csv` + `dataset_manifest.json` (synthetic runs). Options: `--n-train --n-val --n-test`. |
| 2 | `python -m ml.inspect_dataset` | Prints splits, statistics, validates (no NaN/negatives), checks end-of-trace handling. Add `--export-features data/ml/rf_features.csv` to look at the exact feature table. |
| 3 | `python -m ml.train_rf` | Trains the RF (about 1 min), prints held-out MAE/RMSE, saves `models/rf_traffic_predictor.joblib` + `_metadata.json`, `results/ml/training_metrics.csv`. Options: `--n-estimators --max-depth --min-samples-leaf --max-features --seed --horizon --target-mode`. |
| 4 | `python -m ml.evaluate_predictors` | MAE/RMSE of RF vs MA vs persistence on held-out test runs and the original stream; writes `results/ml/prediction_accuracy.csv` and plots in `plots/ml/`. |
| 5 | `python -m experiments.static` | **Approach A** -> `data/processed/static_results.csv`. |
| 6 | `python -m experiments.dynamic` | **Approach B** (moving average) -> `data/processed/dynamic_results.csv` (identical to the mid-semester output). |
| 7 | `python -m experiments.dynamic --predictor rf` | **Approach C** -> `data/processed/dynamic_rf_results.csv` + `.run_info.json`. Never overwrites B. Fails with a clear message if no model exists. |
| 8 | `python -m experiments.compare_three` | Validates results (allocations sum to 100, within min/max, same workload), writes `results/comparison_three_way.csv`, `results/prediction_accuracy_in_loop.csv`, `plots/three_way_*.png`. |
| 9 | `python -m experiments.run_multiseed --n-seeds 10` | Repeats A/B/C on 10 independent streams (seeds 4001-4010) -> `results/multiseed/`. Change `--n-seeds` for more. |
| 10 | `python -m ml.leave_scenario_out` | Optional stricter test: train without scenario X, test on X -> `results/ml/leave_scenario_out.csv`. |

Original two-way comparison still works: `python -m experiments.compare` (A vs B; note it re-renders
`plots/comparison_*.png`).

**Different seeds / retraining:** `python -m ml.train_rf --seed 7`; different data:
`python -m ml.generate_dataset --n-train 24`; more evaluation streams: `run_multiseed --n-seeds 30`.
**Reproduce from scratch:** delete the generated files (section 9), then run step 1-9. Training is
deterministic: retraining produced a byte-identical model in this environment.

Command-line selection of the predictor (no source edits): `--predictor ma|rf` and `--model PATH`,
`--input`, `--output` on `experiments/dynamic.py`.

## 7. Verify that the Random Forest is really being used

1. Console prints `Prediction method: RANDOM FOREST (models/rf_traffic_predictor.joblib)`.
2. `data/processed/dynamic_rf_results.csv` has `prediction_source = rf` in every row and extra columns
   `*_ma_prediction_mbps` (what MA would have predicted), `*_rf_raw_mbps` (raw model output),
   `*_rf_flag` (`ok` / `clipped_negative` / `fallback_last_value`).
3. `data/processed/dynamic_rf_results.csv.run_info.json` stores the **SHA-256 of the model file** and
   counts of valid/invalid predictions. Compare with `sha256sum models/rf_traffic_predictor.joblib`.
4. `*_prediction_mbps` differs from `*_ma_prediction_mbps`, and `results/prediction_accuracy_in_loop.csv`
   shows the predictions the controller actually used.
5. `tests/` asserts in-loop predictions equal offline predictions.

Invalid predictions: negative -> clipped to 0 (flag), NaN/inf -> replaced by the last observed demand
(flag). They are counted, never hidden, and never relabelled as MA.

## 8. Results actually obtained (this environment: CPU, Python 3.12.2, scikit-learn 1.9.1)

These numbers were produced by the commands above; your run should match them (deterministic) if the
library versions match; other versions may differ slightly.

**Prediction accuracy** (Mbps, target = demand at t+1; lower is better)

| Dataset | Predictor | MAE | RMSE |
|---|---|---|---|
| Held-out test runs (8 unseen seeds) | **RF** | **0.878** | **2.940** |
| | MA (existing) | 2.442 | 5.230 |
| | persistence (reference) | 1.055 | 3.353 |
| Original stream (traffic_dataset.csv) | **RF** | **0.618** | **2.406** |
| | MA (existing) | 2.107 | 4.705 |
| | persistence (reference) | 0.839 | 2.945 |

Stricter leave-one-scenario-out (RF trained without the tested scenario): RF beats MA in all 6
scenarios, but is **not consistently better than persistence** (persistence is better for
`competition` and `fluctuation`, about equal elsewhere). So the accuracy gain is partly tied to
the scenario templates seen in training.

**Network-level results on the original stream** (single run, deterministic)

| Metric | Static | Dynamic-MA | Dynamic-RF |
|---|---|---|---|
| Total QoS violations | 138 | 132 | 127 |
| eMBB / URLLC / IoT violations | 54 / 54 / 30 | 0 / 76 / 56 | 0 / 76 / 51 |
| URLLC p95 latency (ms) | 50.4 | 80.8 | 81.8 |
| Avg fairness (Jain) | 0.877 | 0.791 | 0.797 |
| Reconfigurations / cost (%) / oscillations | N/A | 29 / 135 / 9 | 30 / 140 / 9 |

**10 independent held-out streams** (mean over seeds 4001-4010; std in `results/multiseed/summary.csv`)

| Metric | Static | Dynamic-MA | Dynamic-RF |
|---|---|---|---|
| Total QoS violations (mean) | 138.6 | 141.3 | 131.2 |
| Reconfigurations (mean) | N/A | 27.6 | 28.1 |
| URLLC p95 latency (ms) | 58.2 | 62.1 | 65.2 |

Paired per-seed results: RF had fewer total QoS violations than MA in 8/10
seeds and more in 2/10. Static has no dynamic reconfiguration metrics because
its allocation is fixed.

**Interpretation (what the data supports):**
* RF predicts next-interval demand about 3x more accurately than the moving average.
* That accuracy translated into a small single-run QoS gain and a lower mean
  violation count over the ten streams. No significance test was done; seeds are
  perturbations of the same templates.
* Why so small (measured): the controller input `max(current, prediction)` differed between RF and MA
  by only about 1 Mbps on average, while the controller moves 5-point steps with a 10-interval cooldown.
  Reconfiguration timing is almost identical in both runs (same first three actions). Allocation-step
  size and cooldown, not prediction error, limit the QoS outcome.
* Dynamic allocation is not better in everything: URLLC p95 latency and IoT violations are *worse* than
  static (and RF's p95 is slightly worse than MA's), and fairness is lower, in exchange for fewer total
  violations and lower eMBB latency/loss.

## 9. Output files

| File | Meaning |
|---|---|
| `data/ml/ml_traffic_runs.csv`, `dataset_manifest.json` | synthetic training/val/test runs; seeds & parameters |
| `models/rf_traffic_predictor.joblib`, `..._metadata.json` | trained model; features, hyperparameters, seeds, dataset hash, library versions, feature importances |
| `results/ml/training_metrics.csv` | val/test MAE, RMSE (RF, MA, persistence) per slice |
| `results/ml/prediction_accuracy.csv` | full accuracy table incl. per scenario |
| `results/ml/leave_scenario_out.csv` | generalisation check |
| `plots/ml/*.png` | actual vs predicted, predicted-vs-actual scatter |
| `data/processed/static_results.csv`, `dynamic_results.csv` | original A and B outputs |
| `data/processed/dynamic_rf_results.csv` (+`.run_info.json`) | approach C output |
| `results/comparison_summary.csv`, `plots/comparison_*.png` | original 2-way comparison |
| `results/comparison_three_way.csv`, `plots/three_way_*.png` | 3-way comparison |
| `results/prediction_accuracy_in_loop.csv` | accuracy of the predictions each controller really used |
| `results/multiseed/per_seed_metrics.csv`, `summary.csv` | multi-seed results (`work/` holds temp streams) |
| `streamlit_app.py` | Streamlit dashboard for seeded traffic, prediction accuracy, three-way results, multi-seed results, and model metadata |

**Clean generated files safely** (original inputs/outputs are not touched):
```bash
rm -rf data/ml models results/ml results/multiseed results/comparison_three_way.csv \
       results/prediction_accuracy_in_loop.csv plots/ml plots/three_way_*.png \
       data/processed/dynamic_rf_results.csv*
```
(Windows: `Remove-Item -Recurse -Force data\ml, models, results\ml, results\multiseed, plots\ml` etc.)
Do **not** delete `data/raw/traffic_dataset.csv`; regenerate it with `cd traffic && python traffic_generator.py`.

## 10. How to read the metrics

* **MAE** - average absolute error in Mbps; "on average the forecast is off by X Mbps".
* **RMSE** - like MAE but penalises big misses (e.g. missing a burst) more. RMSE >> MAE means occasional
  large errors; that is where burst onsets matter.
* **Latency / p95** - simulated latency model (`qos_monitor.py`): base latency while demand <= allocation,
  growing linearly with overload. p95 shows worst-case behaviour.
* **Throughput** - min(demand, allocation). **Packet loss** - unserved fraction. **Utilization** -
  demand / allocation (>1 = overload; very low = wasted capacity).
* **QoS violation** - an interval where latency exceeds the slice target or loss > 1 %; counted per slice.
* **Fairness** - Jain index of the *allocation shares* (1 = equal). A dynamic controller that gives more
  to a loaded slice lowers this by design.
* **Reconfigurations / cost / oscillations** - how often and how much allocation moves (cost = % of
  capacity moved), and direction reversals of the URLLC share.

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: controller` / `ml` | run from the project root, using `python -m package.module` (not `python experiments/x.py`) |
| `ModuleNotFoundError: sklearn` | `pip install -r requirements-ml.txt` in the active venv |
| `ModelNotFoundError: Random Forest model not found` | `python -m ml.generate_dataset && python -m ml.train_rf` |
| `ModelMetadataError` | features changed or model is stale: retrain |
| Unpickle / version warnings or errors on load | the model was saved with another scikit-learn; retrain (1 min) |
| `FileNotFoundError: Traffic dataset not found` | `cd traffic && python traffic_generator.py` |
| `compare_three` says results not found | run steps 5-7 first |
| `Result validation: FAILED` | the files come from different workloads or runs; rerun steps 5-7 together |
| Numbers differ from section 8 | different library versions/platform; compare the **direction** and see `results/` of your run |

## 12. Limitations

* Simulation with modelled traffic and a simple latency model; **no real 5G testbed, no Open5GS/UERANSIM**
  testing; allocations are modelled percentages, not enforced radio resources.
* Training data is synthetic and derived from six scenario templates; held-out seeds test robustness to
  perturbation, not unseen traffic types (see leave-one-scenario-out: no consistent advantage over
  persistence on unseen scenarios).
* Offered demand does not react to allocation, so only demand-history features are used.
* The controller (step 5, cooldown 10, thresholds) was not retuned for RF; a better predictor may need a
  different controller to show larger QoS benefits. No claim is made that RF is better in general.
* RF inference costs more per step than MA (about 9 ms per step measured vs microseconds); irrelevant here, but relevant for a
  real-time deployment. Unit tests exist for the pipeline; only Linux/Python 3.12 was exercised.
* The mid-semester `experiments/test_dynamic.py` is an old print-script that unpacks 2 values from a function
  returning 4 and fails; it was left as is (not part of `tests/`).
