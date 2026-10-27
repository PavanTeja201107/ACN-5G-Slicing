# Test report (executed in the author's sandbox: Linux, Python 3.12.3, scikit-learn 1.8.0)

| Check | How | Result |
|---|---|---|
| Original static & dynamic reproduce before any change | re-ran, SHA-256 vs uploaded CSVs | identical |
| Automated suite | `python -m pytest tests -q` | **26 passed** (about 25 s), run before and after final file restore |
| Features: order/length, no future leakage, padding, invalid history rejected, ma5 == existing predictor | tests | passed |
| Dataset: canonical stream == original CSV, deterministic seeds, disjoint splits, target alignment (horizon 1 and 3), trace-end rows dropped, NaN rejected | tests | passed |
| Missing model -> clear error, no output written; metadata mismatch detected; unknown predictor rejected | tests | passed |
| Invalid predictions (NaN, huge negative) through the full controller: flagged, finite, allocations valid, counts exact | tests (stub model) | passed |
| Allocation constraints (sum 100, min/max) on static/MA/RF results; same workload across approaches | `validate_results.py`, tests, `compare_three`, multi-seed | passed |
| Default MA run byte-identical to original; static run identical | test | passed |
| RF run reproducible (two runs identical); in-loop predictions == offline predictions | test | passed |
| Training reproducible with fixed seed; model save/load; metadata; MAE <= RMSE | test; full retrain gave identical model hash | passed |
| End-to-end pipeline | `run_pipeline.py --quick` and full `run_pipeline.py` | both completed, exit 0 |

**Not tested:** Windows and macOS (commands provided but unexecuted); other Python/library versions;
any Open5GS/UERANSIM deployment; statistical significance of QoS differences.
Run everything locally with `python -m pytest tests -q` and `python run_pipeline.py`.
