"""Model-builder utilities for pneumonia notebooks."""

from __future__ import annotations

import numpy as np
import keras
from keras import layers


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
    """Build VGG16 transfer learning model with BN-regularized dense head.

    Expects raw [0, 255] images — VGG16 preprocessing is applied internally
    after augmentation so that augmentation operates on natural pixel values.
    Do NOT pass pre-processed datasets to this model.
    """
    base_model = keras.applications.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size[0], img_size[1], 3),
    )
    base_model.trainable = not freeze_base

    inputs = keras.Input(shape=(img_size[0], img_size[1], 3))
    x = augmentation_layer(inputs)
    x = keras.applications.vgg16.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        dense_units, activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(
        dense_units_2, activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name=name)
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
    """Unfreeze VGG16 block5 (last conv block) for fine-tuning.

    Locates the VGG16 base inside the model, makes block5_conv1/2/3 and
    block5_pool trainable, then re-compiles with a reduced learning rate.
    Returns the same model object (modified in-place).
    """
    vgg_base = None
    for layer in model.layers:
        if isinstance(layer, keras.Model) and layer.name == "vgg16":
            vgg_base = layer
            break
    if vgg_base is None:
        raise ValueError("Could not find VGG16 base layer named 'vgg16' inside model.")

    # Unfreeze only the last conv block to avoid disrupting lower-level features.
    block5_names = {"block5_conv1", "block5_conv2", "block5_conv3", "block5_pool"}
    vgg_base.trainable = True
    for layer in vgg_base.layers:
        layer.trainable = layer.name in block5_names

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
    trainable_count = int(sum(
        np.prod(w.shape) for w in model.trainable_weights
    ))
    print(f"Unfrozen: {block5_names}")
    print(f"Trainable params after unfreeze: {trainable_count:,}")
    return model


def build_baseline_cnn(
    img_size: tuple[int, int],
    augmentation_layer: keras.Model | None = None,
    learning_rate: float = 1e-3,
    l2_reg: float = 1e-4,
    name: str = "pneumonia_cnn",
) -> keras.Model:
    """Build the baseline CNN architecture extracted from notebook.
    If augmentation_layer is None, the input will be directly scaled.
    """
    inputs = keras.Input(shape=(img_size[0], img_size[1], 3))
    if augmentation_layer is not None:
        x = augmentation_layer(inputs)
    else:
        x = inputs
    x = layers.Rescaling(1.0 / 255)(x)

    for filters in [32, 64, 128, 256]:
        x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2_reg),
    )(x)
    x = layers.Dropout(0.20)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.02),
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def build_baseline_model_from_hp(
    hp,
    img_size: tuple[int, int],
    augmentation_layer: keras.Model,
) -> keras.Model:
    """Build hyperparameter-tunable baseline model for keras-tuner."""
    inputs = keras.Input(shape=(img_size[0], img_size[1], 3))
    x = augmentation_layer(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    num_blocks = hp.Choice("num_blocks", [3, 4, 5])
    start_filters = hp.Choice("start_filters", [32, 64])
    kernel_size = hp.Choice("kernel_size", [3, 5])
    block_dropout = hp.Float("block_dropout", min_value=0.15, max_value=0.35, step=0.05)

    filters = start_filters
    for _ in range(num_blocks):
        x = layers.Conv2D(filters, (kernel_size, kernel_size), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, (kernel_size, kernel_size), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(block_dropout)(x)
        filters = min(filters * 2, 512)

    l2_value = hp.Choice("l2_reg", [1e-4, 1e-3, 1e-2])
    dense_units = hp.Choice("dense_units", [128, 256, 512])
    dense_dropout = hp.Float("dense_dropout", min_value=0.2, max_value=0.5, step=0.05)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2_value),
    )(x)
    x = layers.Dropout(dense_dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="baseline_cnn_tunable")
    learning_rate = hp.Choice("learning_rate", [0.01, 0.005, 0.001, 0.0005, 0.0001])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.02),
        metrics=["accuracy", keras.metrics.AUC(name="auc"), keras.metrics.AUC(name="pr_auc", curve="PR")],
    )
    return model
