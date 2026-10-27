"""
Random Forest traffic-demand predictor (inference side).

Loads a model trained by ml/train_rf.py and predicts the demand of every
slice at the next observation interval (t + horizon). It returns
predictions in the same format as the existing moving-average predictor
output inside experiments/dynamic.py: {"embb": float, "urllc": float,
"iot": float} in Mbps. The existing controller then decides the resource
allocation; this module never allocates resources itself.

Failure policy
--------------
* Model or metadata file missing  -> ModelNotFoundError (clear message).
  There is NO silent fallback to the moving average.
* Metadata/feature mismatch       -> ModelMetadataError.
* Invalid prediction values       -> handled per value and REPORTED:
    - negative    -> clipped to 0.0       flag "clipped_negative"
    - non-finite  -> last observed demand flag "fallback_last_value"
  Counts are kept and written to the run-info file.
"""

import hashlib
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from controller.ml_features import FEATURE_NAMES, NUM_LAGS, SLICES, build_features

DEFAULT_MODEL_PATH = os.path.join("models", "rf_traffic_predictor.joblib")

FLAG_OK = "ok"
FLAG_CLIPPED = "clipped_negative"
FLAG_FALLBACK = "fallback_last_value"


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model or its metadata is missing."""


class ModelMetadataError(ValueError):
    """Raised when model metadata does not match the feature code."""


def metadata_path_for(model_path):
    base, _ = os.path.splitext(model_path)
    return base + "_metadata.json"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_prediction(raw_value, last_observed):
    """
    Make one raw model output safe for the controller.

    Returns (value, flag). Never raises for bad numeric values.
    """

    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return float(last_observed), FLAG_FALLBACK

    if not math.isfinite(value):
        return float(last_observed), FLAG_FALLBACK

    if value < 0.0:
        return 0.0, FLAG_CLIPPED

    return value, FLAG_OK


@dataclass
class PredictionResult:
    values: dict                # validated predictions (Mbps)
    raw: dict                   # raw model outputs before validation
    flags: dict                 # per-slice validation flag


@dataclass
class RandomForestPredictor:
    model: object
    metadata: dict
    model_path: str
    model_sha256: str
    counts: dict = field(
        default_factory=lambda: {
            "calls": 0,
            FLAG_OK: 0,
            FLAG_CLIPPED: 0,
            FLAG_FALLBACK: 0,
        }
    )

    # ------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------

    @classmethod
    def load(cls, model_path=None):
        model_path = model_path or DEFAULT_MODEL_PATH
        meta_path = metadata_path_for(model_path)

        if not os.path.exists(model_path):
            raise ModelNotFoundError(
                f"Random Forest model not found: {model_path}\n"
                f"Train it first (from the project root):\n"
                f"  python -m ml.generate_dataset\n"
                f"  python -m ml.train_rf\n"
                f"No fallback to the moving-average predictor is "
                f"performed automatically."
            )

        if not os.path.exists(meta_path):
            raise ModelNotFoundError(
                f"Model metadata not found: {meta_path}\n"
                f"The metadata file is written next to the model by "
                f"ml/train_rf.py. Retrain the model."
            )

        try:
            import joblib
        except ImportError as error:
            raise ImportError(
                "joblib/scikit-learn are required for the Random "
                "Forest predictor: pip install -r requirements-ml.txt"
            ) from error

        with open(meta_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        if list(metadata.get("feature_names", [])) != list(FEATURE_NAMES):
            raise ModelMetadataError(
                "The model was trained with a different feature list "
                "than controller/ml_features.py. Retrain with "
                "'python -m ml.train_rf'."
            )

        if int(metadata.get("num_lags", -1)) != NUM_LAGS:
            raise ModelMetadataError(
                "NUM_LAGS in metadata differs from ml_features.py. "
                "Retrain the model."
            )

        model = joblib.load(model_path)

        n_in = getattr(model, "n_features_in_", None)
        if n_in is not None and int(n_in) != len(FEATURE_NAMES):
            raise ModelMetadataError(
                f"Model expects {n_in} features, code provides "
                f"{len(FEATURE_NAMES)}. Retrain the model."
            )

        # Tiny batches (3 rows) are faster single-threaded.
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1

        return cls(
            model=model,
            metadata=metadata,
            model_path=model_path,
            model_sha256=file_sha256(model_path),
        )

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------

    def predict_raw_batch(self, feature_matrix, last_observed):
        """
        Raw predicted demand (Mbps) for a matrix of feature rows.

        last_observed : lag_0 of each row; only used when the model was
        trained on the "delta" target (output = lag_0 + predicted change).
        """

        outputs = np.asarray(
            self.model.predict(np.asarray(feature_matrix, dtype=float)),
            dtype=float,
        )

        if self.metadata.get("target_mode") == "delta":
            outputs = outputs + np.asarray(last_observed, dtype=float)

        return outputs

    def predict(self, histories):
        """
        Predict next-interval demand for every slice.

        histories : {"embb": [...], "urllc": [...], "iot": [...]}
                    demand history (Mbps) up to and including time t.
        """

        rows = [build_features(name, histories) for name in SLICES]
        last = [float(histories[name][-1]) for name in SLICES]

        raw = self.predict_raw_batch(rows, last)

        values, raws, flags = {}, {}, {}

        for index, name in enumerate(SLICES):
            value, flag = validate_prediction(raw[index], last[index])
            values[name] = value
            raws[name] = float(raw[index])
            flags[name] = flag
            self.counts[flag] += 1

        self.counts["calls"] += 1

        return PredictionResult(values=values, raw=raws, flags=flags)

    # ------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------

    def run_info(self):
        return {
            "predictor": "random_forest",
            "model_path": self.model_path,
            "model_sha256": self.model_sha256,
            "n_features": len(FEATURE_NAMES),
            "horizon": self.metadata.get("horizon"),
            "target_mode": self.metadata.get("target_mode"),
            "prediction_counts": dict(self.counts),
        }

    def write_run_info(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.run_info(), handle, indent=2)
