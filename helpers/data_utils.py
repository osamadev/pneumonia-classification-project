"""Dataset and preprocessing helpers for pneumonia notebooks."""

from __future__ import annotations

import os
from typing import Callable

import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight as sk_compute_class_weight

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")


def _is_supported_image_file(path: str) -> bool:
    """Return True if path is a regular file with supported image extension."""
    return os.path.isfile(path) and path.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)


def count_images(split_dir: str) -> dict[str, int]:
    """Count images by class name for a dataset split path."""
    counts: dict[str, int] = {}
    for class_name in ["NORMAL", "PNEUMONIA"]:
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            counts[class_name] = 0
            continue
        counts[class_name] = len(
            [
                f
                for f in os.listdir(class_dir)
                if _is_supported_image_file(os.path.join(class_dir, f))
            ]
        )
    return counts


def gather_split_paths_labels(
    train_dir: str,
    test_dir: str,
    val_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect train(+optional val) and test file paths with binary labels."""
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []

    for class_name, class_idx in [("NORMAL", 0), ("PNEUMONIA", 1)]:
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)

        if os.path.isdir(train_class_dir):
            for fname in os.listdir(train_class_dir):
                fpath = os.path.join(train_class_dir, fname)
                if _is_supported_image_file(fpath):
                    train_paths.append(fpath)
                    train_labels.append(class_idx)

        if os.path.isdir(test_class_dir):
            for fname in os.listdir(test_class_dir):
                fpath = os.path.join(test_class_dir, fname)
                if _is_supported_image_file(fpath):
                    test_paths.append(fpath)
                    test_labels.append(class_idx)

    if val_dir and os.path.isdir(val_dir):
        for class_name, class_idx in [("NORMAL", 0), ("PNEUMONIA", 1)]:
            val_class_dir = os.path.join(val_dir, class_name)
            if not os.path.isdir(val_class_dir):
                continue
            for fname in os.listdir(val_class_dir):
                fpath = os.path.join(val_class_dir, fname)
                if _is_supported_image_file(fpath):
                    train_paths.append(fpath)
                    train_labels.append(class_idx)

    return (
        np.array(train_paths),
        np.array(train_labels, dtype=np.int32),
        np.array(test_paths),
        np.array(test_labels, dtype=np.int32),
    )


def load_image(path: tf.Tensor, label: tf.Tensor, img_size: tuple[int, int]):
    """Decode image (jpeg/png/gif/bmp/webp), resize, and return tensor + label."""
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, img_size)
    return img, tf.expand_dims(tf.cast(label, tf.float32), axis=0)


def load_image_with_preprocessing(
    path: tf.Tensor,
    label: tf.Tensor,
    img_size: tuple[int, int],
    preprocess_fn: Callable[[tf.Tensor], tf.Tensor],
):
    """Decode/resize image then apply model-specific preprocessing function."""
    img, label = load_image(path, label, img_size)
    img = preprocess_fn(img)
    return img, label


def build_data_augmentation(
    rotation: float = 30.0,
    width_shift: float = 0.1,
    height_shift: float = 0.1,
    shear: float = 0.2,
    zoom: float = 0.2,
    brightness_delta: float = 0.05,
    name: str = "data_augmentation",
) -> tf.keras.Sequential:
    """Augmentation stack for image data."""
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            # Keras uses fraction of 2*pi for RandomRotation factor.
            layers.RandomRotation(rotation / 360.0),
            layers.RandomTranslation(height_factor=height_shift, width_factor=width_shift),
            layers.RandomZoom(height_factor=zoom, width_factor=zoom),
            layers.RandomShear(x_factor=shear, y_factor=shear),
            layers.RandomBrightness(factor=brightness_delta),
        ],
        name=name,
    )


def compute_class_weights(
    labels: np.ndarray,
    strategy: str = "balanced",
    minority_boost_ratio: float | None = None,
) -> dict[int, float]:
    """Compute class weights from labels.

    strategy:
      - 'balanced': sklearn balanced weighting.
      - 'predefined_ratio': use majority/minority ratio as the minority class weight.
    """
    labels_arr = np.asarray(labels).astype(int)
    classes = np.unique(labels_arr)
    if classes.shape[0] < 2:
        return {int(classes[0]): 1.0}

    if strategy == "predefined_ratio":
        class_counts = {int(c): int(np.sum(labels_arr == c)) for c in classes}
        minority = min(class_counts, key=class_counts.get)
        majority = max(class_counts, key=class_counts.get)
        ratio = class_counts[majority] / max(class_counts[minority], 1)
        if minority_boost_ratio is not None:
            ratio = minority_boost_ratio
        weights = {int(c): 1.0 for c in classes}
        weights[minority] = float(ratio)
        return weights

    # Default sklearn-balanced behavior.
    computed = sk_compute_class_weight(class_weight="balanced", classes=classes, y=labels_arr)
    return {int(c): float(w) for c, w in zip(classes, computed)}


def build_train_val_test_datasets(
    train_paths: np.ndarray,
    train_labels: np.ndarray,
    test_paths: np.ndarray,
    test_labels: np.ndarray,
    img_size: tuple[int, int],
    batch_size: int,
    val_split: float = 0.1,
    seed: int = 42,
    preprocess_fn: Callable[[tf.Tensor], tf.Tensor] | None = None,
    autotune: int | tf.data.AUTOTUNE = tf.data.AUTOTUNE,
):
    """Build train/val/test tf.data datasets with optional balanced sampling."""
    train_paths, val_paths, train_labels_raw, val_labels = train_test_split(
        train_paths,
        train_labels,
        test_size=val_split,
        random_state=seed,
        stratify=train_labels,
    )

    if preprocess_fn is None:
        map_fn = lambda p, y: load_image(p, y, img_size)
    else:
        map_fn = lambda p, y: load_image_with_preprocessing(p, y, img_size, preprocess_fn)

    normal_mask = train_labels_raw == 0
    pneumonia_mask = train_labels_raw == 1

    normal_count = int(normal_mask.sum())
    pneumonia_count = int(pneumonia_mask.sum())
    steps_per_epoch = int(np.ceil((normal_count + pneumonia_count) / batch_size))

    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_paths, train_labels_raw.astype("float32")))
        .shuffle(len(train_paths), seed=seed, reshuffle_each_iteration=True)
        .map(map_fn, num_parallel_calls=autotune)
        .batch(batch_size)
        .cache()
        .prefetch(autotune)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((val_paths, val_labels.astype("float32")))
        .map(map_fn, num_parallel_calls=autotune)
        .batch(batch_size)
        .cache()
        .prefetch(autotune)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((test_paths, test_labels.astype("float32")))
        .map(map_fn, num_parallel_calls=autotune)
        .batch(batch_size)
        .cache()
        .prefetch(autotune)
    )

    metadata = {
        "train_paths": train_paths,
        "train_labels_raw": train_labels_raw,
        "val_paths": val_paths,
        "val_labels": val_labels,
        "steps_per_epoch": steps_per_epoch,
        "normal_count": normal_count,
        "pneumonia_count": pneumonia_count,
    }
    return train_ds, val_ds, test_ds, metadata
