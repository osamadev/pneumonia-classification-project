import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pathlib import Path
from PIL import Image
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight

from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops

import tensorflow as tf
import keras
from keras import layers

#from tensorflow.keras import layers

from tensorflow.keras import layers, models, regularizers

keras.utils.set_random_seed(42)
np.random.seed(42)
random.seed(42)

def show_samples(label, dir):

    folder = os.path.join(dir, label)
    files = os.listdir(folder)[:5]
    plt.figure(figsize=(12,3))

    for i, file in enumerate(files):
        img = Image.open(os.path.join(folder, file))
        plt.subplot(1,5,i+1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")

    plt.suptitle(label)
    plt.show()


def compute_brightness(folder, sample_size=1000):

    brightness = []
    files = os.listdir(folder)[:sample_size]

    for file in files:
        img = Image.open(os.path.join(folder, file)).convert("L")
        img_array = np.array(img)
        brightness.append(img_array.mean())

    return brightness

def compute_entropy(folder, sample_size=1000):

    entropy_values = []
    files = os.listdir(folder)[:sample_size]

    for file in files:
        img = Image.open(os.path.join(folder,file)).convert("L")
        img_array = np.array(img)
        entropy_values.append(shannon_entropy(img_array))

    return entropy_values

def compute_texture(folder, sample_size=1000):

    contrast_values = []
    files = os.listdir(folder)[:sample_size]

    for file in files:
        img = Image.open(os.path.join(folder,file)).convert("L")
        img_array = np.array(img)
        glcm = graycomatrix(
            img_array,
            distances=[5],
            angles=[0],
            levels=256,
            symmetric=True,
            normed=True
        )
        contrast = graycoprops(glcm,"contrast")[0,0]
        contrast_values.append(contrast)

    return contrast_values

def compute_sharpness(folder, sample_size=1000):

    sharpness_values = []
    files = os.listdir(folder)[:sample_size]

    for file in files:
        img = cv2.imread(os.path.join(folder,file), cv2.IMREAD_GRAYSCALE)
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        sharpness_values.append(laplacian_var)

    return sharpness_values

def load_dataset(data_dir, name="Dataset"):
    normal = list(Path(data_dir, "NORMAL").glob("*.jpeg"))
    pneumonia = list(Path(data_dir, "PNEUMONIA").glob("*.jpeg"))

    X = np.array(normal + pneumonia)
    y = np.array([0]*len(normal) + [1]*len(pneumonia))

    print(f"{name} images:", len(X))
    print(f"{name} Normal:", len(normal))
    print(f"{name} Pneumonia:", len(pneumonia))

    return X, y

def preprocess_image(path, label, IMG_SIZE=224):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

geom_augment = tf.keras.Sequential([
    #layers.RandomFlip("horizontal"),
    #layers.RandomRotation(0.1),
    #layers.RandomZoom(0.15),
    #layers.RandomTranslation(0.1, 0.1),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(0.05, 0.05),
    layers.RandomContrast(0.10),
])

def augment(image, label):
    # geometric augmentations
    image = geom_augment(image)

    # pixel-level augmentations
    #image = tf.image.random_brightness(image, 0.1)
    #image = tf.image.random_contrast(image, 0.9, 1.1)

    # resize + random crop
    image = tf.image.random_crop(
        tf.image.resize_with_pad(image, 230, 230),
        size=[224, 224, 1]
    )

    return image, label

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32

def build_dataset(paths, labels, training=False):
    paths = [str(p) for p in paths]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)

    if training:
        dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)
        dataset = dataset.shuffle(2000)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset

def get_class_weights(y, method='balanced'):
    """
    Computes class weights for imbalanced datasets.

    Args:
        y (array-like): Array of labels
        method (str, optional): Method for computing weights. Default is 'balanced'

    Returns:
        dict: Mapping from class index to weight
    """
    classes = np.unique(y)
    weights = compute_class_weight(class_weight=method, classes=classes, y=y)
    return dict(enumerate(weights))
    

def se_block(inputs, reduction):
    filters = inputs.shape[-1]
    se = layers.GlobalAveragePooling2D()(inputs)
    se = layers.Dense(filters // reduction, activation="relu")(se)
    se = layers.Dense(filters, activation="sigmoid")(se)
    se = layers.Reshape((1,1,filters))(se)
    return layers.Multiply()([inputs, se])

def residual_block(x, filters, stride=1, reduction=16, l2_reg=1e-4):
    shortcut = x
    x = layers.Conv2D(
        filters,
        3,
        strides=stride,
        padding="same",
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = se_block(x, reduction)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters,1,strides=stride)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x



def build_baseline_model(input_shape=(224,224,1)):

    inputs = layers.Input(shape=input_shape)

    # Initial feature extractor
    x = layers.Conv2D(32, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Residual stages
    x = residual_block(x, 32)
    x = residual_block(x, 32)

    x = residual_block(x, 64, stride=2)
    x = residual_block(x, 64)

    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)

    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)

    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)

    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)

    return model

def get_callbacks(config=None):

    if config is None:
        config = {}

    early_patience = config.get("early_stopping_patience", 5)
    lr_patience = config.get("lr_reduce_patience", 3)
    lr_factor = config.get("lr_reduce_factor", 0.3)
    min_lr = config.get("min_lr", 1e-6)

    checkpoint_path = config.get("checkpoint_path", "best_cxr_model.h5")

    log_dir = config.get(
        "tensorboard_log_dir",
        os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    )

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=early_patience,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            factor=lr_factor,
            patience=lr_patience,
            min_lr=min_lr,
            verbose=1
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1
        ),

        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
    ]

    return callbacks


def get_predictions(model, dataset, threshold=0.5):
    y_true = []
    y_pred_probs = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)

        y_true.extend(labels.numpy())
        y_pred_probs.extend(preds.flatten())

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs > threshold).astype(int)

    return y_true, y_pred_probs, y_pred

def get_predictions(model, dataset, threshold=0.5):
    y_true = []
    y_pred_probs = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)

        y_true.extend(labels.numpy())
        y_pred_probs.extend(preds.flatten())

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs > threshold).astype(int)

    return y_true, y_pred_probs, y_pred

import numpy as np

def get_predictions_with_tf(model, dataset, threshold=0.5):
    """
    Efficiently get predictions from a TensorFlow/Keras model on a tf.data.Dataset.

    Returns:
        y_true: np.array of true labels
        y_pred_probs: np.array of predicted probabilities
        y_pred: np.array of binary predictions
    """
    # Collect all labels
    y_true = np.concatenate([labels.numpy() for _, labels in dataset], axis=0)
    
    # Predict for the entire dataset at once (TensorFlow handles batching)
    y_pred_probs = model.predict(dataset, verbose=0).flatten()
    
    # Convert probabilities to binary predictions
    y_pred = (y_pred_probs > threshold).astype(int)
    
    return y_true, y_pred_probs, y_pred

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, labels=None, figsize=(6,5), cmap="Blues", title="Confusion Matrix"):
    """
    Plots a confusion matrix using Seaborn heatmap.

    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        labels (list of str, optional): Class names in order. Default: numeric labels
        figsize (tuple, optional): Figure size
        cmap (str, optional): Colormap
        title (str, optional): Plot title
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=labels if labels else range(len(cm)),
        yticklabels=labels if labels else range(len(cm))
    )
    
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_pred_probs, figsize=(6,5), title="ROC Curve"):
    """
    Plots ROC curve and computes AUC for binary classification.

    Args:
        y_true (array-like): True labels (0/1)
        y_pred_probs (array-like): Predicted probabilities for the positive class
        figsize (tuple, optional): Figure size
        title (str, optional): Plot title
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], '--', color='gray')
    
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.show()
    
    return fpr, tpr, roc_auc

import tensorflow as tf

def get_last_conv_layer(model):
    """
    Returns the name of the last Conv2D layer in a Keras model.

    Args:
        model (tf.keras.Model): The model to inspect

    Returns:
        str or None: Name of the last Conv2D layer, or None if not found
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

def get_images_and_labels(dataset):
    """
    Extracts all images and labels from a tf.data.Dataset.
    
    Args:
        dataset (tf.data.Dataset)
        
    Returns:
        images (np.array), labels (np.array)
    """
    all_images = []
    all_labels = []

    for images, labels in dataset:
        all_images.extend(images.numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_images), np.array(all_labels)

def visualize_gradcam_samples(
    model,
    dataset,
    last_conv_layer_name,
    class_names=["NORMAL", "PNEUMONIA"],
    num_images=10
):
    """
    Randomly selects images from a dataset and visualizes them with Grad-CAM overlay.
    
    Args:
        model (tf.keras.Model)
        dataset (tf.data.Dataset)
        last_conv_layer_name (str)
        class_names (list of str)
        num_images (int)
    """
    images, labels = get_images_and_labels(dataset)
    indices = random.sample(range(len(images)), num_images)

    plt.figure(figsize=(10, num_images * 3))

    for i, idx in enumerate(indices):
        test_image = images[idx]
        true_label = int(labels[idx])
        img_array = np.expand_dims(test_image, axis=0)

        # Prediction
        prediction = model.predict(img_array, verbose=0)[0][0]
        pred_class = 1 if prediction > 0.5 else 0
        pred_label = class_names[pred_class]
        true_label_name = class_names[true_label]
        confidence = prediction if pred_class == 1 else 1 - prediction
        result = "✓" if pred_class == true_label else "✗"

        # Grad-CAM
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        img = test_image.squeeze()
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # Original image
        plt.subplot(num_images, 2, 2*i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"True: {true_label_name}")
        plt.axis("off")

        # Grad-CAM overlay
        plt.subplot(num_images, 2, 2*i + 2)
        plt.imshow(img, cmap="gray")
        plt.imshow(heatmap, cmap="jet", alpha=0.4)
        plt.title(f"{result} Pred: {pred_label} ({confidence:.2f})")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
