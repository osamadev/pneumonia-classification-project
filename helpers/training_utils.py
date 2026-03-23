"""Training/evaluation helper functions for pneumonia notebooks."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import keras
from sklearn.metrics import classification_report, roc_auc_score


def to_serializable(value):
    """
    Convert NumPy objects to JSON-serializable Python types.

    This is useful when saving metrics, hyperparameters, or other
    results containing NumPy types to JSON, which does not support
    np.ndarray, np.int, or np.float directly.

    Args:
        value: Any Python object, potentially a NumPy scalar or array.

    Returns:
        A JSON-serializable equivalent:
            - np.floating or np.integer → Python float or int
            - np.ndarray → list
            - Other types returned unchanged
    """
    if isinstance(value, (np.floating, np.integer)):
        return value.item()  # Convert NumPy scalar to native Python float/int
    if isinstance(value, np.ndarray):
        return value.tolist()  # Convert NumPy array to Python list
    return value  # Return other types unchanged


def save_model_with_meta(
    model: keras.Model,
    name: str,
    save_dir: str | Path,
    metrics_dict: dict,
    history,
    hyperparams: dict,
    threshold: float,
):
    """
    Save a Keras model along with accompanying metadata as a JSON sidecar file.

    The metadata includes:
        - Model name and save timestamp
        - File path of the saved model
        - Classification threshold used
        - Evaluation metrics
        - Training history
        - Hyperparameters
        - Model parameter counts (total, trainable, non-trainable)

    Args:
        model (keras.Model): The trained Keras model to save.
        name (str): Base name for model and metadata files.
        save_dir (str | Path): Directory to save the model and metadata.
        metrics_dict (dict): Dictionary of evaluation metrics to save.
        history: Keras History object from training.
        hyperparams (dict): Dictionary of model hyperparameters.
        threshold (float): Threshold value used for binary classification decisions.

    Behavior:
        - Creates `save_dir` if it does not exist.
        - Saves the model in the Keras `.keras` format.
        - Serializes metadata to JSON, converting NumPy types to native Python types.
    """

    # Ensure save directory exists
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Paths for model and metadata files
    model_path = save_dir / f"{name}.keras"
    meta_path = save_dir / f"{name}_meta.json"

    # Save the Keras model
    model.save(model_path)

    # Extract training history if available
    history_dict = history.history if hasattr(history, "history") else {}

    # Construct metadata dictionary
    metadata = {
        "model_name": name,                                   # Model identifier
        "saved_at": datetime.utcnow().isoformat() + "Z",      # UTC timestamp
        "model_path": str(model_path),                        # Path to saved model
        "threshold": float(threshold),                        # Classification threshold
        "metrics": {k: to_serializable(v) for k, v in metrics_dict.items()},  # Metrics
        "hyperparameters": {k: to_serializable(v) for k, v in hyperparams.items()},  # Hyperparameters
        "history": {k: [to_serializable(x) for x in vals] for k, vals in history_dict.items()},  # Training history
        "model_params": {                                     # Model parameter counts
            "total": int(model.count_params()),
            "trainable": int(np.sum([np.prod(v.shape) for v in model.trainable_weights])),
            "non_trainable": int(np.sum([np.prod(v.shape) for v in model.non_trainable_weights])),
        },
    }

    # Save metadata to JSON
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Print confirmation messages
    print(f"Saved model to: {model_path}")
    print(f"Saved metadata to: {meta_path}")


def make_model_saver(save_dir: str | Path):
    """
    Create a closure around `save_model_with_meta` that binds a fixed save directory.

    This allows repeatedly saving different models and metadata to the same
    directory without specifying `save_dir` each time.

    Args:
        save_dir (str | Path): Directory where models and metadata will be saved.

    Returns:
        Callable: A function with signature
                  `(model, name, metrics_dict, history, hyperparams, threshold) -> None`
                  that saves the model and its metadata to `save_dir`.
    """

    # Inner function that calls save_model_with_meta with bound save_dir
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
    """
    Load a Keras model with backward-compatible fixes for augmentation layers.

    Specifically addresses cases where `RandomShear` layers saved with older
    Keras versions serialize `x_factor` and `y_factor` as symmetric ranges
    (e.g., [-0.2, 0.2]). Newer Keras versions may reject these during deserialization.

    Args:
        model_path (str | Path): Path to the saved Keras model (.keras or folder).
        compile (bool): Whether to compile the model after loading.

    Returns:
        keras.Model: Loaded Keras model with compatible augmentation layer configs.

    Behavior:
        - First tries to load the model normally.
        - If deserialization fails due to RandomShear config, patches the `from_config`
          method to convert symmetric ranges to single float factors.
        - Restores the original `from_config` method after attempting the load.
    """

    try:
        # Attempt normal model loading
        return keras.models.load_model(model_path, compile=compile)

    except Exception as first_error:
        # Patch RandomShear layer to handle old serialized factor ranges
        random_shear_cls = keras.layers.RandomShear

        # Save original from_config method
        original_from_config = getattr(random_shear_cls, "from_config")

        def _patched_from_config(cls, config):
            config = dict(config)

            # Convert x_factor / y_factor from symmetric range [-a, a] → a
            for key in ("x_factor", "y_factor"):
                value = config.get(key)
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    low, high = float(value[0]), float(value[1])
                    config[key] = max(abs(low), abs(high))

            return original_from_config(config)

        # Apply patched method
        random_shear_cls.from_config = classmethod(_patched_from_config)

        try:
            # Retry model loading with patched RandomShear
            return keras.models.load_model(model_path, compile=compile)

        except Exception:
            # If still fails, raise the original error
            raise first_error

        finally:
            # Restore the original from_config method to avoid side effects
            random_shear_cls.from_config = original_from_config


def load_model_meta(save_dir: str | Path, name: str):
    """
    Load metadata JSON associated with a saved model.

    This reads the `<name>_meta.json` file from `save_dir` and returns
    the deserialized dictionary. Returns `None` if the file does not exist.

    Args:
        save_dir (str | Path): Directory containing the metadata file.
        name (str): Base name of the model whose metadata to load.

    Returns:
        dict | None: Metadata dictionary if file exists, else None.
    """

    # Construct path to metadata JSON
    meta_path = Path(save_dir) / f"{name}_meta.json"

    # Return None if metadata file does not exist
    if not meta_path.exists():
        return None

    # Load and return JSON metadata
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_training_callbacks(
    checkpoint_path: str | Path,
    patience: int = 4,
    monitor: str = "val_loss",
    verbose: int = 0,
):
    """
    Create a standard set of Keras callbacks for training models.

    Includes:
        - EarlyStopping: Stops training when monitored metric stops improving.
        - ReduceLROnPlateau: Reduces learning rate if metric plateaus.
        - ModelCheckpoint: Saves best model to disk based on monitored metric.
        - TerminateOnNaN: Stops training if NaN loss occurs.

    Args:
        checkpoint_path (str | Path): Path to save the best model checkpoint.
        patience (int): Number of epochs to wait before stopping or reducing LR.
        monitor (str): Metric to monitor (e.g., "val_loss", "val_auc").
        verbose (int): Verbosity level for callbacks.

    Returns:
        list: List of Keras callback instances.
    """

    # Determine whether metric should be minimized or maximized
    mode = "min" if "loss" in monitor else "max"

    return [
        # Stop training early if metric does not improve
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=verbose,
            mode=mode,
        ),

        # Reduce learning rate when metric plateaus
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.3,
            patience=max(1, patience // 2),
            min_lr=1e-7,
            verbose=verbose,
            mode=mode,
        ),

        # Save the best model to checkpoint_path
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor,
            save_best_only=True,
            verbose=verbose,
            mode=mode,
        ),

        # Terminate training if loss becomes NaN
        keras.callbacks.TerminateOnNaN(),
    ]


def tune_threshold(model: keras.Model, val_ds, num_points: int = 91):
    """
    Tune the optimal classification threshold on validation data to maximize F1 score
    for the PNEUMONIA class.

    Args:
        model (keras.Model): Trained Keras model for binary classification.
        val_ds: Validation dataset as a tf.data.Dataset yielding (images, labels).
        num_points (int): Number of thresholds to evaluate between 0.05 and 0.95.

    Returns:
        tuple:
            best_threshold (float): Threshold achieving highest F1 score on PNEUMONIA.
            best_f1 (float): Best F1 score corresponding to best_threshold.
            y_val_true (np.ndarray): Ground truth labels from validation set.
            y_val_prob (np.ndarray): Model-predicted probabilities for PNEUMONIA class.
    """

    # Collect true labels and predicted probabilities across all batches
    y_true_batches, y_prob_batches = [], []
    for images, labels in val_ds:
        probs = model(images, training=False).numpy().ravel()   # Predicted probabilities
        y_prob_batches.append(probs)
        y_true_batches.append(labels.numpy().ravel().astype(int))  # Convert labels to int

    # Flatten batch lists into full arrays
    y_val_true = np.concatenate(y_true_batches)
    y_val_prob = np.concatenate(y_prob_batches)

    # Search thresholds in the range [0.05, 0.95]
    thresholds = np.linspace(0.05, 0.95, num_points)
    best_threshold, best_f1 = 0.5, -1.0

    for threshold in thresholds:
        # Convert probabilities to binary predictions using current threshold
        y_pred = (y_val_prob >= threshold).astype(int)

        # Compute classification metrics
        report = classification_report(
            y_val_true,
            y_pred,
            target_names=["NORMAL", "PNEUMONIA"],
            output_dict=True,
            zero_division=0,
        )

        # Extract F1 score for PNEUMONIA class
        f1 = report["PNEUMONIA"]["f1-score"]

        # Update best threshold if current F1 is higher
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold, float(best_f1), y_val_true, y_val_prob


def evaluate_model(model: keras.Model, test_ds, threshold: float):
    """
    Evaluate a binary classification model on a test dataset at a given threshold.

    Collects predicted probabilities and ground truth labels for the entire test set,
    applies the threshold to produce binary predictions, and computes key metrics.

    Args:
        model (keras.Model): Trained Keras model for binary classification.
        test_ds: Test dataset as a tf.data.Dataset yielding (images, labels).
        threshold (float): Decision threshold to convert probabilities into class labels.

    Returns:
        tuple:
            metrics (dict): Summary metrics including accuracy, AUC, precision, recall, F1 for PNEUMONIA.
            report (dict): Full classification report as returned by sklearn's classification_report.
            y_true (np.ndarray): Ground truth labels.
            y_prob (np.ndarray): Model-predicted probabilities for PNEUMONIA class.
            y_pred (np.ndarray): Binary predictions obtained using the threshold.
    """

    # Collect predictions and true labels batch-wise
    y_true_batches, y_prob_batches = [], []
    for images, labels in test_ds:
        probs = model(images, training=False).numpy().ravel()  # Predicted probabilities
        y_prob_batches.append(probs)
        y_true_batches.append(labels.numpy().ravel().astype(int))  # Convert labels to int

    # Concatenate batches to form full arrays
    y_true = np.concatenate(y_true_batches)
    y_prob = np.concatenate(y_prob_batches)

    # Apply threshold to get binary predictions
    y_pred = (y_prob >= threshold).astype(int)

    # Compute classification report
    report = classification_report(
        y_true,
        y_pred,
        target_names=["NORMAL", "PNEUMONIA"],
        output_dict=True,
        zero_division=0,
    )

    # Aggregate key metrics
    metrics = {
        "accuracy": float(report["accuracy"]),
        "auc": float(roc_auc_score(y_true, y_prob)),                    # ROC-AUC on probabilities
        "precision": float(report["PNEUMONIA"]["precision"]),           # Precision for PNEUMONIA
        "recall": float(report["PNEUMONIA"]["recall"]),                 # Recall for PNEUMONIA
        "f1": float(report["PNEUMONIA"]["f1-score"]),                   # F1 score for PNEUMONIA
    }

    return metrics, report, y_true, y_prob, y_pred
