"""
Automated tests for the Random Forest extension.

    python -m pytest tests -q

Tests that need the trained model are skipped (with a reason) if
models/rf_traffic_predictor.joblib does not exist yet.
"""

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from controller import ml_predictor as mp
from controller.ml_features import FEATURE_NAMES, NUM_LAGS, SLICES, build_features
from experiments import dynamic, static
from experiments.validate_results import validate, check_allocations
from ml import dataset_generator as dg
from ml.feature_table import build_run_table, build_table, validate_table

MODEL = mp.DEFAULT_MODEL_PATH
needs_model = pytest.mark.skipif(not os.path.exists(MODEL), reason="train the model first: python -m ml.train_rf")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ---------------------------------------------------------------- features
def hist(n=30, seed=0):
    r = np.random.default_rng(seed)
    return {s: list(r.uniform(1, 50, n)) for s in SLICES}


def test_feature_vector_shape_and_order():
    f = build_features("urllc", hist())
    assert len(f) == len(FEATURE_NAMES) == 23
    assert f[:3] == [0.0, 1.0, 0.0]                      # slice one-hot


def test_features_use_no_future_information():
    h = hist(30)
    base = build_features("embb", {k: v[:20] for k, v in h.items()})
    h2 = {k: v[:20] + [999.0] * 10 for k, v in h.items()}     # corrupt "future"
    assert build_features("embb", {k: v[:20] for k, v in h2.items()}) == base


def test_short_history_is_padded_not_error():
    f = build_features("iot", {s: [5.0] for s in SLICES})
    assert all(np.isfinite(f))


@pytest.mark.parametrize("bad", [[], [float("nan")], [float("inf")]])
def test_invalid_history_rejected(bad):
    h = hist(5)
    h["embb"] = bad
    with pytest.raises(ValueError):
        build_features("embb", h)


def test_ma5_matches_existing_predictor():
    h = hist(40)
    f = dict(zip(FEATURE_NAMES, build_features("embb", h)))
    assert f["ma5"] == pytest.approx(dynamic.predict_demand(h["embb"]))


# ---------------------------------------------------------------- dataset
def test_canonical_stream_equals_original_csv():
    ref = pd.read_csv("data/raw/traffic_dataset.csv")
    got = dg.canonical_stream()
    assert np.allclose(ref[["embb_mbps", "urllc_mbps", "iot_mbps"]], got[["embb_mbps", "urllc_mbps", "iot_mbps"]])


def test_runs_deterministic_and_seeds_differ():
    a, b, c = dg.perturbed_run(1), dg.perturbed_run(1), dg.perturbed_run(2)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["embb_mbps"], c["embb_mbps"])
    assert len(a) == 1080 and (a[["embb_mbps", "urllc_mbps", "iot_mbps"]] > 0).all().all()


def test_split_seed_ranges_disjoint():
    s = [set(dg.seeds_for(k, 50)) for k in ("train", "val", "test", "multiseed_eval")]
    assert all(s[i].isdisjoint(s[j]) for i in range(4) for j in range(i + 1, 4))


@pytest.mark.parametrize("horizon", [1, 3])
def test_target_alignment_and_trace_end(horizon):
    run = dg.perturbed_run(5)
    t = build_run_table(run, horizon)
    n = len(run)
    assert len(t) == (n - horizon) * 3                     # last rows dropped
    emb = t[t["slice"] == "embb"].reset_index(drop=True)
    series = run["embb_mbps"].to_numpy()
    assert np.allclose(emb["target_mbps"], series[horizon:])
    assert np.allclose(emb["current_mbps"], series[: n - horizon])
    validate_table(t)


def test_validate_table_rejects_nan():
    t = build_run_table(dg.perturbed_run(5))
    t.loc[0, "lag_0"] = np.nan
    with pytest.raises(ValueError):
        validate_table(t)


# ---------------------------------------------------------------- predictor
def test_missing_model_gives_clear_error(tmp_path):
    with pytest.raises(mp.ModelNotFoundError, match="python -m ml.train_rf"):
        mp.RandomForestPredictor.load(str(tmp_path / "nope.joblib"))


def test_validate_prediction_handles_bad_values():
    assert mp.validate_prediction(-3.0, 7.0) == (0.0, mp.FLAG_CLIPPED)
    assert mp.validate_prediction(float("nan"), 7.0) == (7.0, mp.FLAG_FALLBACK)
    assert mp.validate_prediction(float("inf"), 7.0) == (7.0, mp.FLAG_FALLBACK)
    assert mp.validate_prediction("x", 7.0) == (7.0, mp.FLAG_FALLBACK)
    assert mp.validate_prediction(4.2, 7.0) == (4.2, mp.FLAG_OK)


@needs_model
def test_model_loads_and_predicts_valid_values():
    p = mp.RandomForestPredictor.load()
    r = p.predict(hist(25))
    assert set(r.values) == set(SLICES)
    assert all(np.isfinite(v) and v >= 0 for v in r.values.values())
    assert p.run_info()["model_sha256"] == sha(MODEL)


@needs_model
def test_metadata_mismatch_detected(tmp_path):
    import shutil
    m = tmp_path / "m.joblib"
    shutil.copy(MODEL, m)
    meta = json.load(open(mp.metadata_path_for(MODEL)))
    meta["feature_names"] = meta["feature_names"][:-1]
    json.dump(meta, open(mp.metadata_path_for(str(m)), "w"))
    with pytest.raises(mp.ModelMetadataError):
        mp.RandomForestPredictor.load(str(m))


# ---------------------------------------------------------------- integration
@pytest.fixture
def reset_dynamic():
    saved = (dynamic.INPUT_FILE, dynamic.OUTPUT_FILE, dynamic.PREDICTOR, dynamic.MODEL_PATH)
    yield
    dynamic.INPUT_FILE, dynamic.OUTPUT_FILE, dynamic.PREDICTOR, dynamic.MODEL_PATH = saved


def run_dynamic(out, predictor):
    dynamic.OUTPUT_FILE, dynamic.PREDICTOR, dynamic.MODEL_PATH = str(out), predictor, None
    dynamic.main()
    return pd.read_csv(out)


def test_default_moving_average_run_is_unchanged(tmp_path, reset_dynamic):
    """Output of the modified runner (default predictor) == original saved output."""
    out = tmp_path / "ma.csv"
    run_dynamic(out, "ma")
    assert sha(out) == sha("data/processed/dynamic_results.csv")
    assert "prediction_source" not in pd.read_csv(out).columns


def test_static_run_is_unchanged(tmp_path):
    saved = (static.OUTPUT_FILE,)
    static.OUTPUT_FILE = str(tmp_path / "s.csv")
    try:
        static.main()
    finally:
        static.OUTPUT_FILE = saved[0]
    assert sha(tmp_path / "s.csv") == sha("data/processed/static_results.csv")


def test_unknown_predictor_rejected(tmp_path, reset_dynamic):
    with pytest.raises(ValueError):
        run_dynamic(tmp_path / "x.csv", "lstm")


def test_rf_missing_model_stops_run(tmp_path, reset_dynamic):
    dynamic.OUTPUT_FILE, dynamic.PREDICTOR = str(tmp_path / "x.csv"), "rf"
    dynamic.MODEL_PATH = str(tmp_path / "missing.joblib")
    with pytest.raises(mp.ModelNotFoundError):
        dynamic.main()
    assert not (tmp_path / "x.csv").exists()               # nothing silently written


@needs_model
def test_rf_run_valid_labelled_reproducible_and_consistent(tmp_path, reset_dynamic):
    a = run_dynamic(tmp_path / "rf1.csv", "rf")
    b = run_dynamic(tmp_path / "rf2.csv", "rf")
    pd.testing.assert_frame_equal(a, b)                     # reproducible

    assert (a["prediction_source"] == "rf").all()
    ma = pd.read_csv("data/processed/dynamic_results.csv")
    static_df = pd.read_csv("data/processed/static_results.csv")
    assert validate(static_df, ma, a) == []                 # allocations valid, same workload
    assert not np.allclose(a["embb_prediction_mbps"], a["embb_ma_prediction_mbps"])   # RF != MA

    # in-loop predictions == offline batch predictions on the same stream
    p = mp.RandomForestPredictor.load()
    stream = pd.read_csv("data/raw/traffic_dataset.csv")
    h = {s: stream[f"{s}_mbps"].tolist() for s in SLICES}
    for t in (0, 10, 500, 1079):
        r = p.predict({s: h[s][: t + 1] for s in SLICES})
        for s in SLICES:
            assert a[f"{s}_prediction_mbps"].iloc[t] == pytest.approx(r.values[s])


class _Stub:
    """Fake model returning invalid outputs, to test controller safety."""
    def __init__(self, value): self.value = value
    def predict(self, X): return np.full(len(X), self.value)


@pytest.mark.parametrize("value, flag", [(float("nan"), mp.FLAG_FALLBACK), (-500.0, mp.FLAG_CLIPPED)])
def test_controller_survives_invalid_predictions(tmp_path, reset_dynamic, monkeypatch, value, flag):
    stub = mp.RandomForestPredictor(model=_Stub(value), metadata={"target_mode": "absolute", "horizon": 1},
                                    model_path="stub", model_sha256="stub")
    monkeypatch.setattr(mp.RandomForestPredictor, "load", classmethod(lambda cls, path=None: stub))
    df = run_dynamic(tmp_path / "bad.csv", "rf")
    assert (df["embb_rf_flag"] == flag).all()
    assert np.isfinite(df["embb_prediction_mbps"]).all() and (df["embb_prediction_mbps"] >= 0).all()
    assert check_allocations(df, "bad") == []
    assert stub.counts[flag] == 3240


def test_allocation_checker_detects_violations():
    bad = pd.DataFrame({"embb_allocation_pct": [50.0], "urllc_allocation_pct": [30.0], "iot_allocation_pct": [25.0]})
    assert check_allocations(bad, "x")
    low = pd.DataFrame({"embb_allocation_pct": [75.0], "urllc_allocation_pct": [15.0], "iot_allocation_pct": [10.0]})
    assert check_allocations(low, "x")


# ---------------------------------------------------------------- training
def test_training_pipeline_small_and_reproducible(tmp_path, monkeypatch):
    from ml import generate_dataset, train_rf
    runs = tmp_path / "runs.csv"
    monkeypatch.setattr(sys, "argv", ["x", "--n-train", "2", "--n-val", "1", "--n-test", "1", "--output", str(runs)])
    generate_dataset.main()

    preds = []
    for i in (1, 2):
        model = tmp_path / f"m{i}.joblib"
        monkeypatch.setattr(sys, "argv", ["x", "--runs", str(runs), "--model-out", str(model),
                                          "--metrics-out", str(tmp_path / f"met{i}.csv"), "--n-estimators", "8"])
        train_rf.main()
        p = mp.RandomForestPredictor.load(str(model))
        preds.append(p.predict(hist(25)).values)
        meta = json.load(open(mp.metadata_path_for(str(model))))
        assert meta["feature_names"] == list(FEATURE_NAMES)
        assert set(meta["training_runs"]).isdisjoint(meta["test_runs"])
    assert preds[0] == preds[1]                             # fixed seed -> identical model
    met = pd.read_csv(tmp_path / "met1.csv")
    assert (met["mae_mbps"] >= 0).all() and (met["rmse_mbps"] >= met["mae_mbps"] - 1e-9).all()
