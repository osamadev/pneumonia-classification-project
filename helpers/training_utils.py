"""Training/evaluation helper functions for pneumonia notebooks."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import keras
from sklearn.metrics import classification_report, roc_auc_score


def to_serializable(value):
    """Convert numpy values to JSON-serializable types."""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_model_with_meta(
    model: keras.Model,
    name: str,
    save_dir: str | Path,
    metrics_dict: dict,
    history,
    hyperparams: dict,
    threshold: float,
):
    """Save Keras model and sidecar metadata JSON."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / f"{name}.keras"
    meta_path = save_dir / f"{name}_meta.json"

    model.save(model_path)
    history_dict = history.history if hasattr(history, "history") else {}

    metadata = {
        "model_name": name,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "model_path": str(model_path),
        "threshold": float(threshold),
        "metrics": {k: to_serializable(v) for k, v in metrics_dict.items()},
        "hyperparameters": {k: to_serializable(v) for k, v in hyperparams.items()},
        "history": {k: [to_serializable(x) for x in vals] for k, vals in history_dict.items()},
        "model_params": {
            "total": int(model.count_params()),
            "trainable": int(np.sum([np.prod(v.shape) for v in model.trainable_weights])),
            "non_trainable": int(np.sum([np.prod(v.shape) for v in model.non_trainable_weights])),
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model to: {model_path}")
    print(f"Saved metadata to: {meta_path}")


def make_model_saver(save_dir: str | Path):
    """Create a save_model_with_meta callable bound to one save directory."""
    def _save(model, name, metrics_dict, history, hyperparams, threshold):
        return save_model_with_meta(
            model=model,
            name=name,
            save_dir=save_dir,
            metrics_dict=metrics_dict,
            history=history,
            hyperparams=hyperparams,
            threshold=threshold,
        )

    return _save


def load_model_compat(model_path: str | Path, compile: bool = True):
    """Load a Keras model with fallback fixes for augmentation layer configs.

    This handles checkpoints saved with RandomShear configs that serialize
    factors as symmetric ranges (e.g. [-0.2, 0.2]) that some newer Keras
    versions reject during deserialization.
    """
    model_path = str(model_path)
    try:
        return keras.models.load_model(model_path, compile=compile)
    except Exception as first_error:
        random_shear_cls = keras.layers.RandomShear
        original_from_config_attr = random_shear_cls.__dict__["from_config"]
        original_from_config = random_shear_cls.from_config

        def _patched_from_config(cls, config):
            config = dict(config)
            for key in ("x_factor", "y_factor"):
                value = config.get(key)
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    low = float(value[0])
                    high = float(value[1])
                    # Legacy configs may store symmetric ranges like [-0.2, 0.2].
                    config[key] = max(abs(low), abs(high))
            return original_from_config(config)

        random_shear_cls.from_config = classmethod(_patched_from_config)
        try:
            return keras.models.load_model(model_path, compile=compile)
        except Exception:
            # Preserve the original error context for easier debugging.
            raise first_error
        finally:
            random_shear_cls.from_config = original_from_config_attr


def load_model_meta(save_dir: str | Path, name: str):
    """Load a model metadata JSON by name; return None if missing."""
    meta_path = Path(save_dir) / f"{name}_meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_training_callbacks(
    checkpoint_path: str | Path,
    patience: int = 4,
    monitor: str = "val_loss",
):
    """Shared callback bundle used for both notebooks."""
    return [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.3,
            patience=max(1, patience // 2),
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor,
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.TerminateOnNaN(),
    ]


def tune_threshold(model: keras.Model, val_ds, num_points: int = 91):
    """Tune decision threshold on validation data using F1 on PNEUMONIA class."""
    y_true_batches, y_prob_batches = [], []
    for images, labels in val_ds:
        probs = model.predict(images, verbose=0).ravel()
        y_prob_batches.append(probs)
        y_true_batches.append(labels.numpy().ravel().astype(int))

    y_val_true = np.concatenate(y_true_batches)
    y_val_prob = np.concatenate(y_prob_batches)

    thresholds = np.linspace(0.05, 0.95, num_points)
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in thresholds:
        y_pred = (y_val_prob >= threshold).astype(int)
        report = classification_report(
            y_val_true,
            y_pred,
            target_names=["NORMAL", "PNEUMONIA"],
            output_dict=True,
            zero_division=0,
        )
        f1 = report["PNEUMONIA"]["f1-score"]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold, float(best_f1), y_val_true, y_val_prob


def evaluate_model(model: keras.Model, test_ds, threshold: float):
    """Evaluate model on test dataset and return metrics + raw vectors."""
    y_true_batches, y_prob_batches = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0).ravel()
        y_prob_batches.append(probs)
        y_true_batches.append(labels.numpy().ravel().astype(int))

    y_true = np.concatenate(y_true_batches)
    y_prob = np.concatenate(y_prob_batches)
    y_pred = (y_prob >= threshold).astype(int)

    report = classification_report(
        y_true,
        y_pred,
        target_names=["NORMAL", "PNEUMONIA"],
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(report["accuracy"]),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(report["PNEUMONIA"]["precision"]),
        "recall": float(report["PNEUMONIA"]["recall"]),
        "f1": float(report["PNEUMONIA"]["f1-score"]),
    }
    return metrics, report, y_true, y_prob, y_pred
