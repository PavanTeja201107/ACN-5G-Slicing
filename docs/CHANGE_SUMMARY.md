# Change summary

Verification method: every original file was compared **byte by byte** (`cmp`) with the uploaded ZIP
after all experiments and tests had been run.

## Modified (1 file)
* `experiments/dynamic.py` - **additive only: +93 lines, 0 lines removed or changed.**
  Why unavoidable: the prediction call is inline inside `main()`, there is no other injection point.
  Added: (1) `PREDICTOR`/`MODEL_PATH` settings (default `"ma"`); (2) lazy model loading before the loop
  when `PREDICTOR == "rf"`; (3) after the original MA prediction, `if rf_predictor is not None:` the RF
  values replace `prediction` (MA kept for logging); (4) extra log columns and `.run_info.json` **only for
  RF runs**; (5) `argparse` CLI in `__main__` (`--predictor --model --input --output`).
  With default settings the output CSV is **byte-identical** to the original (SHA-256 checked, also by a test).

## Added
* `controller/ml_features.py`, `controller/ml_predictor.py`
* `ml/` : `dataset_generator.py`, `generate_dataset.py`, `inspect_dataset.py`, `feature_table.py`,
  `train_rf.py`, `evaluation.py`, `evaluate_predictors.py`, `leave_scenario_out.py`
* `experiments/` : `compare_three.py`, `run_multiseed.py`, `validate_results.py`, `metrics_extra.py`
* `tests/conftest.py`, `tests/test_ml_extension.py`
* `run_pipeline.py`, `requirements-ml.txt`, `README.md` (was empty), `docs/`
* Generated artifacts: `models/*`, `data/ml/*`, `data/processed/dynamic_rf_results.csv*`,
  `results/ml/*`, `results/multiseed/*`, `results/comparison_three_way.csv`,
  `results/prediction_accuracy_in_loop.csv`, `plots/ml/*`, `plots/three_way_*.png`

## Preserved byte-identical (32 of the 33 original files)
All of `traffic/`, `controller/{controller,qos_monitor,fairness,stability,reconfiguration}.py`,
`experiments/{static,compare,scenarios,test_*}.py`, `requirements.txt`, `.gitignore`,
`data/raw/traffic_dataset.csv`, `data/processed/static_results.csv`, `data/processed/dynamic_results.csv`,
`results/comparison_summary.csv`, and the 11 original PNG files in `plots/`.
(Re-running the original `compare.py` re-renders its PNGs with a newer matplotlib; the originals were
restored so they remain exactly as submitted. Its CSV output was identical.)
