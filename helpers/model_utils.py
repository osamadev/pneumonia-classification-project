"""Model-builder utilities for pneumonia notebooks."""

from __future__ import annotations

import numpy as np
import keras
from keras import layers


def build_baseline_cnn(
    img_size: tuple[int, int],
    augmentation_layer: keras.Model | None = None,
    learning_rate: float = 1e-3,
    l2_reg: float = 1e-4,
    name: str = "pneumonia_cnn",
) -> keras.Model:
    """
    Build a baseline CNN model for binary classification (e.g., pneumonia detection).

    The architecture:
        - Accepts raw images of size `img_size`.
        - Optionally applies a data augmentation layer before scaling.
        - Contains 4 convolutional blocks with Conv2D → BatchNorm → Conv2D → BatchNorm → MaxPool → Dropout.
        - Ends with a GlobalAveragePooling, Dense layer with L2 regularization, and output sigmoid.
    
    Args:
        img_size (tuple[int, int]): Input image size (height, width).
        augmentation_layer (keras.Model | None): Optional data augmentation layer applied to inputs.
        learning_rate (float): Learning rate for Adam optimizer.
        l2_reg (float): L2 regularization factor for the dense layer.
        name (str): Name of the Keras model.

    Returns:
        keras.Model: Compiled baseline CNN model.
    """

    # Input layer for raw images
    inputs = keras.Input(shape=(img_size[0], img_size[1], 3))

    # Apply augmentation if provided; otherwise pass inputs directly
    if augmentation_layer is not None:
        x = augmentation_layer(inputs)
    else:
        x = inputs

    # Scale pixel values to [0, 1]
    x = layers.Rescaling(1.0 / 255)(x)

    # ----------- Convolutional blocks -----------
    for filters in [32, 64, 128, 256]:
        x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)  # Conv layer
        x = layers.BatchNormalization()(x)                                       # Batch normalization
        x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)  # Second Conv layer
        x = layers.BatchNormalization()(x)                                       # Batch normalization
        x = layers.MaxPooling2D((2, 2))(x)                                       # Downsample with MaxPool
        x = layers.Dropout(0.25)(x)                                              # Regularization with dropout

    # Global average pooling to reduce spatial dimensions
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected dense layer with L2 regularization
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2_reg),
    )(x)
    x = layers.Dropout(0.20)(x)  # Dropout for regularization

    # Output layer for binary classification
    outputs = layers.Dense(1, activation="sigmoid")(x)

    # Assemble and compile the model
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.02),  # Slight label smoothing
        metrics=[
            "accuracy",  # Standard classification accuracy
            keras.metrics.AUC(name="auc"),  # ROC-AUC metric
            keras.metrics.AUC(name="pr_auc", curve="PR"),  # Precision-recall AUC
        ],
    )

    return model


def build_baseline_model_from_hp(
    hp,
    img_size: tuple[int, int],
    augmentation_layer: keras.Model,
) -> keras.Model:
    """
    Build a hyperparameter-tunable baseline CNN for use with Keras Tuner.

    This function allows tuning of:
        - Number of convolutional blocks
        - Initial filter count
        - Kernel size
        - Dropout rates per block and dense layer
        - Dense layer size and L2 regularization
        - Learning rate

    Args:
        hp: HyperParameters object from keras-tuner.
        img_size (tuple[int, int]): Input image size (height, width).
        augmentation_layer (keras.Model): Data augmentation layer applied to inputs.

    Returns:
        keras.Model: Compiled Keras model ready for hyperparameter tuning.
    """

    # Input layer for raw images
    inputs = keras.Input(shape=(img_size[0], img_size[1], 3))

    # Apply data augmentation
    x = augmentation_layer(inputs)

    # Rescale pixel values to [0, 1]
    x = layers.Rescaling(1.0 / 255)(x)

    # ----------- Convolutional blocks with hyperparameter tuning -----------
    num_blocks = hp.Choice("num_blocks", [3, 4, 5])               # Number of Conv blocks
    start_filters = hp.Choice("start_filters", [32, 64])          # Initial number of filters
    kernel_size = hp.Choice("kernel_size", [3, 5])                # Kernel size for Conv layers
    block_dropout = hp.Float("block_dropout", min_value=0.15, max_value=0.35, step=0.05)  # Dropout per block

    filters = start_filters
    for _ in range(num_blocks):
        # First convolution in the block
        x = layers.Conv2D(filters, (kernel_size, kernel_size), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)

        # Second convolution in the block
        x = layers.Conv2D(filters, (kernel_size, kernel_size), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)

        # Downsample with max pooling
        x = layers.MaxPooling2D((2, 2))(x)

        # Apply dropout for regularization
        x = layers.Dropout(block_dropout)(x)

        # Double filters for next block, capped at 512
        filters = min(filters * 2, 512)

    # ----------- Dense head with hyperparameter tuning -----------
    l2_value = hp.Choice("l2_reg", [1e-4, 1e-3, 1e-2])            # L2 regularization
    dense_units = hp.Choice("dense_units", [128, 256, 512])       # Units in dense layer
    dense_dropout = hp.Float("dense_dropout", min_value=0.2, max_value=0.5, step=0.05)  # Dropout

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2_value),
    )(x)
    x = layers.Dropout(dense_dropout)(x)

    # Output layer for binary classification
    outputs = layers.Dense(1, activation="sigmoid")(x)

    # Assemble the model
    model = keras.Model(inputs=inputs, outputs=outputs, name="baseline_cnn_tunable")

    # Select learning rate for Adam optimizer from hyperparameters
    learning_rate = hp.Choice("learning_rate", [0.01, 0.005, 0.001, 0.0005, 0.0001])

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.02),
        metrics=[
            "accuracy",                          # Standard accuracy
            keras.metrics.AUC(name="auc"),       # ROC-AUC
            keras.metrics.AUC(name="pr_auc", curve="PR"),  # Precision-Recall AUC
        ],
    )

    return model


def build_vgg16_model(
    img_size: tuple[int, int],
    augmentation_layer: keras.Model,
    dense_units: int = 256,
    dense_units_2: int = 128,
    dropout: float = 0.5,
    dropout_2: float = 0.3,
    learning_rate: float = 1e-4,
    freeze_base: bool = True,
    name: str = "vgg16_frozen",
) -> keras.Model:
    """
    Build a VGG16-based transfer learning model with a batch-normalized dense head.

    This model:
        - Uses VGG16 pretrained on ImageNet as the feature extractor.
        - Applies data augmentation before VGG16 preprocessing.
        - Adds two fully connected dense layers with L2 regularization and dropout.
        - Outputs a single sigmoid unit for binary classification.

    Notes:
        - Expects raw images with pixel values in [0, 255].
        - Do NOT feed pre-processed images; preprocessing is applied internally.
        - The base VGG16 can be frozen or fine-tuned based on `freeze_base`.

    Args:
        img_size (tuple[int, int]): Input image size (height, width).
        augmentation_layer (keras.Model): Keras layer/model for data augmentation.
        dense_units (int): Number of units in the first dense layer.
        dense_units_2 (int): Number of units in the second dense layer.
        dropout (float): Dropout rate after first dense layer.
        dropout_2 (float): Dropout rate after second dense layer.
        learning_rate (float): Adam optimizer learning rate.
        freeze_base (bool): If True, freezes VGG16 base model during training.
        name (str): Name of the Keras model.

    Returns:
        keras.Model: Compiled VGG16-based binary classification model.
    """

    # Load the VGG16 base model with pretrained ImageNet weights, excluding top layers
    base_model = keras.applications.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size[0], img_size[1], 3),
    )
    base_model.trainable = not freeze_base  # Freeze or unfreeze the base

    # Input layer for raw images
    inputs = keras.Input(shape=(img_size[0], img_size[1], 3))

    # Apply data augmentation
    x = augmentation_layer(inputs)

    # Apply VGG16 preprocessing (scaling & mean subtraction)
    x = keras.applications.vgg16.preprocess_input(x)

    # Extract features with the base VGG16 model (training=False ensures BN is not updated)
    x = base_model(x, training=False)

    # Global average pooling to reduce spatial dimensions
    x = layers.GlobalAveragePooling2D()(x)

    # First dense block with L2 regularization, batch normalization, and dropout
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    # Second dense block with L2 regularization, batch normalization, and dropout
    x = layers.Dense(
        dense_units_2,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_2)(x)

    # Output layer: single sigmoid unit for binary classification
    outputs = layers.Dense(1, activation="sigmoid")(x)

    # Assemble the full model
    model = keras.Model(inputs, outputs, name=name)

    # Compile model with Adam optimizer and common classification metrics
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def unfreeze_vgg16_top_block(model: keras.Model, learning_rate: float = 1e-5) -> keras.Model:
    """
    Unfreeze the last convolutional block (block5) of VGG16 for fine-tuning.

    This function locates the VGG16 base inside a larger model, makes the last conv block
    (block5_conv1, block5_conv2, block5_conv3, block5_pool) trainable while keeping earlier
    layers frozen, then recompiles the model with a lower learning rate. The model is modified
    in-place and returned.

    Args:
        model (keras.Model): A Keras model containing a VGG16 base layer named 'vgg16'.
        learning_rate (float): Learning rate for fine-tuning the unfrozen layers.

    Returns:
        keras.Model: The same model object, with VGG16 block5 layers trainable and
                     compiled for fine-tuning.
    """

    # Locate the VGG16 base model inside the full model
    vgg_base = None
    for layer in model.layers:
        if isinstance(layer, keras.Model) and layer.name == "vgg16":
            vgg_base = layer
            break

    # Raise error if the VGG16 base is not found
    if vgg_base is None:
        raise ValueError("Could not find VGG16 base layer named 'vgg16' inside model.")

    # Define the names of layers in the last convolutional block (block5)
    block5_names = {"block5_conv1", "block5_conv2", "block5_conv3", "block5_pool"}

    # Enable training for the VGG16 base
    vgg_base.trainable = True

    # Set only block5 layers as trainable; freeze all other layers
    for layer in vgg_base.layers:
        layer.trainable = layer.name in block5_names

    # Recompile the model with a lower learning rate for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    # Compute total number of trainable parameters for logging
    trainable_count = int(sum(np.prod(w.shape) for w in model.trainable_weights))

    # Print debug info about unfrozen layers
    print(f"Unfrozen: {block5_names}")
    print(f"Trainable params after unfreeze: {trainable_count:,}")

    return model
