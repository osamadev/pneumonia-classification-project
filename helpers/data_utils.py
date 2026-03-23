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
    """
    Return True if path is a regular file with supported image extension.

    Args:
        path (str): Full file path to check.

    Returns:
        bool: True if the path points to an existing file and its extension
              matches one of the allowed image formats defined in
              SUPPORTED_IMAGE_EXTENSIONS; otherwise False.
    """
    # Check that the path exists and is a file (not a directory)
    # AND verify that the file name ends with one of the supported
    # image extensions (case-insensitive comparison)
    return os.path.isfile(path) and path.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)


def count_images(split_dir: str) -> dict[str, int]:
    """
    Count images by class name for a dataset split path.

    Args:
        split_dir (str): Path to the dataset split directory (e.g., train, test, val).
                         This directory is expected to contain subdirectories for each class.

    Returns:
        dict[str, int]: A dictionary mapping each class name ("NORMAL", "PNEUMONIA")
                        to the number of valid image files found in its corresponding folder.
    """
    # Initialize an empty dictionary to store image counts per class
    counts: dict[str, int] = {}

    # Iterate over the expected class names
    for class_name in ["NORMAL", "PNEUMONIA"]:
        # Construct the full path to the class-specific directory
        class_dir = os.path.join(split_dir, class_name)

        # Check if the class directory exists
        if not os.path.isdir(class_dir):
            # If the directory does not exist, assign a count of 0 for this class
            counts[class_name] = 0
            continue

        # Count the number of supported image files in the directory
        counts[class_name] = len(
            [
                f  # file name
                for f in os.listdir(class_dir)  # list all files in the class directory
                if _is_supported_image_file(
                    os.path.join(class_dir, f)
                )  # include only valid image files
            ]
        )

    # Return the dictionary containing counts for each class
    return counts


def gather_split_paths_labels(
    train_dir: str,
    test_dir: str,
    val_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect train (+ optional validation) and test file paths along with their binary labels.

    This function scans dataset directories organized by class names ("NORMAL", "PNEUMONIA"),
    gathers valid image file paths, and assigns corresponding numeric labels:
        - "NORMAL"     -> 0
        - "PNEUMONIA"  -> 1

    Optionally, validation data (if provided) is merged into the training set.

    Args:
        train_dir (str): Path to the training dataset directory.
                         Expected structure: train_dir/NORMAL and train_dir/PNEUMONIA
        test_dir (str): Path to the test dataset directory.
                        Expected structure: test_dir/NORMAL and test_dir/PNEUMONIA
        val_dir (str | None): Optional path to a validation dataset directory.
                              If provided, its data will be appended to the training set.

    Returns:
        tuple:
            - np.ndarray: Array of training file paths (including validation if provided)
            - np.ndarray: Array of training labels (int32)
            - np.ndarray: Array of test file paths
            - np.ndarray: Array of test labels (int32)
    """

    # Lists to store file paths and labels for training data
    train_paths, train_labels = [], []

    # Lists to store file paths and labels for test data
    test_paths, test_labels = [], []

    # Loop through each class and assign a numeric label
    for class_name, class_idx in [("NORMAL", 0), ("PNEUMONIA", 1)]:
        # Construct full directory paths for the current class in train and test splits
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)

        # ----------- Process training data -----------
        if os.path.isdir(train_class_dir):
            # Iterate over all files in the class directory
            for fname in os.listdir(train_class_dir):
                # Build full file path
                fpath = os.path.join(train_class_dir, fname)

                # Check if file is a valid supported image
                if _is_supported_image_file(fpath):
                    # Append valid image path and its corresponding label
                    train_paths.append(fpath)
                    train_labels.append(class_idx)

        # ----------- Process test data -----------
        if os.path.isdir(test_class_dir):
            # Iterate over all files in the class directory
            for fname in os.listdir(test_class_dir):
                # Build full file path
                fpath = os.path.join(test_class_dir, fname)

                # Check if file is a valid supported image
                if _is_supported_image_file(fpath):
                    # Append valid image path and its corresponding label
                    test_paths.append(fpath)
                    test_labels.append(class_idx)

    # ----------- Optionally process validation data -----------
    # If a validation directory is provided and exists,
    # its data is merged into the training dataset
    if val_dir and os.path.isdir(val_dir):
        for class_name, class_idx in [("NORMAL", 0), ("PNEUMONIA", 1)]:
            # Construct validation class directory path
            val_class_dir = os.path.join(val_dir, class_name)

            # Skip if the class directory does not exist
            if not os.path.isdir(val_class_dir):
                continue

            # Iterate over validation files
            for fname in os.listdir(val_class_dir):
                # Build full file path
                fpath = os.path.join(val_class_dir, fname)

                # Check if file is a valid supported image
                if _is_supported_image_file(fpath):
                    # Append to training data (validation is merged into train)
                    train_paths.append(fpath)
                    train_labels.append(class_idx)

    # ----------- Convert lists to NumPy arrays -----------
    return (
        np.array(train_paths),  # Array of training file paths
        np.array(train_labels, dtype=np.int32),  # Training labels as int32
        np.array(test_paths),  # Array of test file paths
        np.array(test_labels, dtype=np.int32),  # Test labels as int32
    )


def load_image(path: tf.Tensor, label: tf.Tensor, img_size: tuple[int, int]):
    """
    Decode an image file, resize it to a target size, and return the image tensor with its label.

    This function is typically used in a TensorFlow data pipeline (e.g., tf.data.Dataset)
    where `path` and `label` are tensors rather than standard Python types.

    Args:
        path (tf.Tensor): A scalar string tensor representing the file path to the image.
        label (tf.Tensor): A tensor containing the class label (e.g., 0 or 1).
        img_size (tuple[int, int]): Target image size as (height, width).

    Returns:
        tuple:
            - tf.Tensor: The processed image tensor of shape (height, width, 3)
            - tf.Tensor: The label tensor as a float32 with shape (1,)
    """

    # Read the raw file contents from disk using the provided file path
    raw = tf.io.read_file(path)

    # Decode the image into a tensor
    # Supports multiple formats: JPEG, PNG, GIF, BMP, WEBP
    # `channels=3` ensures output is RGB (3 channels)
    # `expand_animations=False` ensures GIFs are treated as single images (no time dimension)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)

    # Explicitly set the shape to (height, width, 3)
    # Height and width are unknown at this stage (None), but channels are fixed to 3
    # This helps TensorFlow build static shape information for the graph
    img.set_shape([None, None, 3])

    # Resize the image to the target size (img_size)
    # This ensures all images have consistent dimensions for model input
    img = tf.image.resize(img, img_size)

    # Return:
    # - the processed image tensor
    # - the label cast to float32 and expanded to shape (1,)
    #   (useful for models expecting labels with an explicit dimension)
    return img, tf.expand_dims(tf.cast(label, tf.float32), axis=0)


def load_image_with_preprocessing(
    path: tf.Tensor,
    label: tf.Tensor,
    img_size: tuple[int, int],
    preprocess_fn: Callable[[tf.Tensor], tf.Tensor],
):
    """
    Load an image and apply additional model-specific preprocessing.

    This function builds on `load_image` by first decoding and resizing the image,
    then applying a custom preprocessing function (e.g., normalization, scaling,
    or model-specific transformations such as those required by pretrained models).

    Args:
        path (tf.Tensor): A scalar string tensor representing the file path to the image.
        label (tf.Tensor): A tensor containing the class label (e.g., 0 or 1).
        img_size (tuple[int, int]): Target image size as (height, width).
        preprocess_fn (Callable[[tf.Tensor], tf.Tensor]):
            A function that takes an image tensor and returns a processed image tensor.
            This is typically a preprocessing function from a deep learning library
            (e.g., normalization for ResNet, EfficientNet, etc.).

    Returns:
        tuple:
            - tf.Tensor: The preprocessed image tensor
            - tf.Tensor: The label tensor (unchanged from `load_image`)
    """

    # Load and resize the image using the base helper function
    # This step handles file reading, decoding, and resizing
    img, label = load_image(path, label, img_size)

    # Apply the provided preprocessing function to the image
    # This may include normalization, scaling pixel values,
    # or other transformations required by a specific model
    img = preprocess_fn(img)

    # Return the processed image along with its label
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
    """
    Create a data augmentation pipeline using Keras preprocessing layers.

    This function builds a sequential stack of random image transformations
    that are applied on-the-fly during training to improve model generalization.
    Augmentation helps the model become more robust to variations in the input data
    (e.g., orientation, position, lighting).

    Args:
        rotation (float): Maximum rotation angle in degrees.
                          Internally converted to a fraction of 360° for Keras.
        width_shift (float): Fraction of total width for horizontal translations.
        height_shift (float): Fraction of total height for vertical translations.
        shear (float): Shear intensity applied along both x and y axes.
        zoom (float): Zoom range (fractional scaling in/out).
        brightness_delta (float): Range for random brightness adjustment.
        name (str): Name assigned to the Keras Sequential model.

    Returns:
        tf.keras.Sequential: A sequential model containing augmentation layers,
                             intended to be used as part of a training pipeline.
    """

    # Build and return a sequential pipeline of augmentation layers
    return tf.keras.Sequential(
        [
            # Randomly flip images horizontally (left ↔ right)
            layers.RandomFlip("horizontal"),

            # Randomly rotate images
            layers.RandomRotation(rotation / 360.0),

            # Randomly shift images vertically and horizontally
            layers.RandomTranslation(
                height_factor=height_shift,
                width_factor=width_shift
            ),

            # Randomly zoom in/out of the image
            layers.RandomZoom(
                height_factor=zoom,
                width_factor=zoom
            ),

            # Apply random shear transformations (distortion along axes)
            layers.RandomShear(
                x_factor=shear,
                y_factor=shear
            ),

            # Randomly adjust image brightness
            layers.RandomBrightness(
                factor=brightness_delta
            ),
        ],
        name=name,  # Assign a name to the augmentation pipeline
    )


def compute_class_weights(
    labels: np.ndarray,
    strategy: str = "balanced",
    minority_boost_ratio: float | None = None,
) -> dict[int, float]:
    """
    Compute class weights for handling class imbalance in classification tasks.

    This function calculates per-class weights that can be used to balance
    the contribution of each class during model training, useful for datasets
    where one class is underrepresented.

    Args:
        labels (np.ndarray): Array of integer labels corresponding to dataset samples.
        strategy (str): Strategy for computing class weights. Options:
            - 'balanced': Use sklearn's balanced weighting.
            - 'predefined_ratio': Set minority class weight based on majority/minority ratio.
        minority_boost_ratio (float | None): Optional override for minority class weight
            when using 'predefined_ratio' strategy.

    Returns:
        dict[int, float]: Dictionary mapping class index to computed weight.

    Notes:
        - Balanced weighting assigns weight inversely proportional to class frequency.
        - Predefined ratio allows explicit control over minority class importance.
    """

    # Ensure labels are a NumPy integer array
    labels_arr = np.asarray(labels).astype(int)

    # Identify unique classes in the labels
    classes = np.unique(labels_arr)

    # If there's only one class, return weight 1.0 for it (no imbalance)
    if classes.shape[0] < 2:
        return {int(classes[0]): 1.0}

    # ----------- Predefined ratio strategy -----------
    if strategy == "predefined_ratio":
        # Count number of samples per class
        class_counts = {int(c): int(np.sum(labels_arr == c)) for c in classes}

        # Identify minority and majority classes
        minority = min(class_counts, key=class_counts.get)
        majority = max(class_counts, key=class_counts.get)

        # Compute ratio of majority to minority class as the weight
        ratio = class_counts[majority] / max(class_counts[minority], 1)

        # Override ratio if user provided a minority_boost_ratio
        if minority_boost_ratio is not None:
            ratio = minority_boost_ratio

        # Initialize weights for all classes to 1.0
        weights = {int(c): 1.0 for c in classes}

        # Assign computed ratio to the minority class
        weights[minority] = float(ratio)

        # Return the weights dictionary
        return weights

    # ----------- Balanced strategy (default) -----------
    # Use sklearn's compute_class_weight to calculate balanced weights
    computed = sk_compute_class_weight(class_weight="balanced", classes=classes, y=labels_arr)

    # Convert result to dictionary mapping class index -> weight
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
    """
    Build TensorFlow training, validation, and test datasets with optional preprocessing
    and batching. Splits the training data into training and validation sets while
    preserving class balance.

    Args:
        train_paths (np.ndarray): Array of training image file paths.
        train_labels (np.ndarray): Array of corresponding training labels.
        test_paths (np.ndarray): Array of test image file paths.
        test_labels (np.ndarray): Array of corresponding test labels.
        img_size (tuple[int, int]): Target image size (height, width) for resizing.
        batch_size (int): Number of samples per batch.
        val_split (float): Fraction of training data to reserve for validation.
        seed (int): Random seed for reproducible shuffling and splitting.
        preprocess_fn (Callable | None): Optional function to preprocess images
                                         (e.g., normalization or model-specific preprocessing).
        autotune (int | tf.data.AUTOTUNE): Number of parallel calls for dataset mapping.

    Returns:
        tuple: (train_ds, val_ds, test_ds, metadata)
            - train_ds (tf.data.Dataset): Training dataset, shuffled, mapped, batched.
            - val_ds (tf.data.Dataset): Validation dataset, mapped, batched.
            - test_ds (tf.data.Dataset): Test dataset, mapped, batched.
            - metadata (dict): Dictionary with dataset info and counts.
    """

    # Split training data into train/validation sets while preserving class balance
    train_paths, val_paths, train_labels_raw, val_labels = train_test_split(
        train_paths,
        train_labels,
        test_size=val_split,
        random_state=seed,
        stratify=train_labels,
    )

    # Choose the appropriate mapping function for loading images
    # Either simple loading or loading with model-specific preprocessing
    if preprocess_fn is None:
        map_fn = lambda p, y: load_image(p, y, img_size)
    else:
        map_fn = lambda p, y: load_image_with_preprocessing(p, y, img_size, preprocess_fn)

    # Create boolean masks for each class in the training set
    normal_mask = train_labels_raw == 0
    pneumonia_mask = train_labels_raw == 1

    # Count the number of samples per class
    normal_count = int(normal_mask.sum())
    pneumonia_count = int(pneumonia_mask.sum())

    # Calculate steps per epoch based on batch size
    steps_per_epoch = int(np.ceil((normal_count + pneumonia_count) / batch_size))

    # ----------- Build training dataset -----------
    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_paths, train_labels_raw.astype("float32")))  # Create dataset from arrays
        .shuffle(len(train_paths), seed=seed, reshuffle_each_iteration=True)                    # Shuffle dataset each epoch
        .map(map_fn, num_parallel_calls=autotune)                                               # Apply mapping function in parallel
        .batch(batch_size)                                                                       # Batch the dataset
        .cache()                                                                                 # Cache in memory for performance
        .prefetch(autotune)                                                                     # Prefetch to improve throughput
    )

    # ----------- Build validation dataset -----------
    val_ds = (
        tf.data.Dataset.from_tensor_slices((val_paths, val_labels.astype("float32")))           # Validation dataset
        .map(map_fn, num_parallel_calls=autotune)                                               # Apply mapping function
        .batch(batch_size)                                                                       # Batch the dataset
        .cache()                                                                                 # Cache in memory
        .prefetch(autotune)                                                                     # Prefetch for performance
    )

    # ----------- Build test dataset -----------
    test_ds = (
        tf.data.Dataset.from_tensor_slices((test_paths, test_labels.astype("float32")))         # Test dataset
        .map(map_fn, num_parallel_calls=autotune)                                               # Apply mapping function
        .batch(batch_size)                                                                       # Batch the dataset
        .cache()                                                                                 # Cache in memory
        .prefetch(autotune)                                                                     # Prefetch for performance
    )

    # ----------- Metadata for monitoring and training purposes -----------
    metadata = {
        "train_paths": train_paths,           # Paths in training dataset
        "train_labels_raw": train_labels_raw, # Labels in training dataset
        "val_paths": val_paths,               # Paths in validation dataset
        "val_labels": val_labels,             # Labels in validation dataset
        "steps_per_epoch": steps_per_epoch,   # Steps per epoch for training
        "normal_count": normal_count,         # Number of "NORMAL" samples
        "pneumonia_count": pneumonia_count,   # Number of "PNEUMONIA" samples
    }

    # Return the datasets and metadata
    return train_ds, val_ds, test_ds, metadata
