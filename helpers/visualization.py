"""Visualization utilities for training and evaluation."""

from __future__ import annotations

from . import training_utils

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from pathlib import Path


import matplotlib.cm as cm
import cv2
import matplotlib as mpl

from sklearn.metrics import confusion_matrix, roc_curve, auc

from IPython.display import display

from sklearn.metrics import confusion_matrix, classification_report 



def plot_training_curves(history):
    """
    Plot training and validation curves for accuracy, loss, and AUC.

    Args:
        history: Keras History object returned by model.fit().
                 Should contain 'history' attribute with metric lists.

    Behavior:
        - Creates a 1x3 subplot figure:
            1. Accuracy (train vs val)
            2. Loss (train vs val)
            3. AUC (train vs val)
        - Skips any metric if it is missing from the history.
        - Displays the figure using matplotlib.
    """

    if history is None or not getattr(history, "history", None):
        print("No history available to plot.")
        return

    hist = history.history
    plt.figure(figsize=(16, 4))

    # ----------------- Accuracy subplot -----------------
    plt.subplot(1, 3, 1)
    if "accuracy" in hist:
        plt.plot(hist["accuracy"], label="train")          # Training accuracy
    if "val_accuracy" in hist:
        plt.plot(hist["val_accuracy"], label="val")       # Validation accuracy
    plt.title("Accuracy")
    plt.legend()

    # ----------------- Loss subplot -----------------
    plt.subplot(1, 3, 2)
    if "loss" in hist:
        plt.plot(hist["loss"], label="train")             # Training loss
    if "val_loss" in hist:
        plt.plot(hist["val_loss"], label="val")           # Validation loss
    plt.title("Loss")
    plt.legend()

    # ----------------- AUC subplot -----------------
    plt.subplot(1, 3, 3)
    if "auc" in hist:
        plt.plot(hist["auc"], label="train")              # Training AUC
    if "val_auc" in hist:
        plt.plot(hist["val_auc"], label="val")           # Validation AUC
    plt.title("AUC")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels=("NORMAL", "PNEUMONIA")):
    """
    Plot a heatmap of the confusion matrix with labeled axes.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted binary labels.
        labels (tuple): Class names corresponding to 0 and 1.

    Behavior:
        - Computes the confusion matrix using sklearn.metrics.confusion_matrix.
        - Plots a colored heatmap with annotation of counts.
        - X-axis represents predicted labels, Y-axis represents actual labels.
        - Uses a blue color map for visualization.
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create figure
    plt.figure(figsize=(5, 4))

    # Plot heatmap with annotated counts
    sns.heatmap(
        cm,
        annot=True,          # Show counts
        fmt="d",             # Integer format
        cmap="Blues",        # Color map
        xticklabels=labels,  # Label x-axis
        yticklabels=labels,  # Label y-axis
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def plot_sample_predictions(
    model,
    test_ds,
    threshold: float,
    n: int = 9,
):
    """
    Display a grid of random test images with predicted classes and probabilities.

    Args:
        model (keras.Model): Trained binary classification model.
        test_ds: Test dataset as a tf.data.Dataset yielding (images, labels).
        threshold (float): Decision threshold for classifying probabilities.
        n (int): Number of random samples to display (default 9).

    Behavior:
        - Aggregates all images and labels from the test dataset.
        - Computes model predictions as probabilities and binary classes.
        - Randomly selects `n` images for visualization.
        - Displays each image with its true label, predicted label, and predicted probability.
        - Handles images in [0, 255] or [0, 1] ranges, scaling for display.
    """

    # Collect all images and labels from test dataset
    images_all, labels_all = [], []
    for batch_images, batch_labels in test_ds:
        images_all.append(batch_images.numpy())
        labels_all.append(batch_labels.numpy().ravel().astype(int))

    images_all = np.concatenate(images_all)
    labels_all = np.concatenate(labels_all)

    # Predict probabilities and binary labels
    probs = model.predict(images_all, verbose=0).ravel()
    preds = (probs >= threshold).astype(int)

    # Randomly sample `n` images
    sample_idx = random.sample(range(len(images_all)), k=min(n, len(images_all)))
    cols = 3
    rows = int(np.ceil(len(sample_idx) / cols))
    plt.figure(figsize=(5 * cols, 4 * rows))

    for i, idx in enumerate(sample_idx):
        ax = plt.subplot(rows, cols, i + 1)
        img = images_all[idx]

        # Scale image to [0,1] for display if necessary
        img_display = img
        if img_display.max() > 1.5:
            img_display = img_display / 255.0
        ax.imshow(np.clip(img_display, 0.0, 1.0))

        # Determine labels
        true_lbl = "PNEUMONIA" if labels_all[idx] == 1 else "NORMAL"
        pred_lbl = "PNEUMONIA" if preds[idx] == 1 else "NORMAL"

        # Set title with true, predicted labels and probability
        ax.set_title(f"T:{true_lbl} | P:{pred_lbl}\nprob={probs[idx]:.3f}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_combined_training_curves(history1, history2=None,
                                   phase1_label="Phase 1 – Frozen Base",
                                   phase2_label="Phase 2 – Fine-tune"):
    """
    Plot stitched training curves for accuracy, loss, and AUC across one or two training phases.

    This is useful for transfer learning workflows where a model is first trained
    with a frozen base and then fine-tuned. If `history2` is provided, the curves
    from the second phase are appended, and a vertical dashed line marks the phase boundary.

    Args:
        history1: Keras History object for phase 1 training.
        history2: Optional Keras History object for phase 2 training.
        phase1_label (str): Label for first training phase.
        phase2_label (str): Label for second training phase.

    Behavior:
        - Plots a 1x3 subplot for Accuracy, Loss, and AUC.
        - Combines training and validation curves across phases.
        - Marks the transition between phases with a vertical dashed line if history2 exists.
    """

    # Helper to safely extract metric lists
    def _get(h, key):
        return h.history.get(key, []) if hasattr(h, "history") else []

    metrics = [
        ("accuracy", "val_accuracy", "Accuracy"),
        ("loss", "val_loss", "Loss"),
        ("auc", "val_auc", "AUC"),
    ]

    plt.figure(figsize=(16, 4))
    for col, (train_key, val_key, title) in enumerate(metrics, start=1):
        plt.subplot(1, 3, col)

        # Extract phase 1 metrics
        train_vals = list(_get(history1, train_key))
        val_vals   = list(_get(history1, val_key))
        boundary   = len(train_vals)  # Epoch where phase 1 ends

        # Append phase 2 metrics if provided
        if history2 is not None:
            train_vals += list(_get(history2, train_key))
            val_vals   += list(_get(history2, val_key))

        epochs = range(1, len(train_vals) + 1)

        # Plot curves
        if train_vals:
            plt.plot(epochs, train_vals, label="train")
        if val_vals:
            plt.plot(epochs, val_vals, label="val")

        # Mark phase boundary
        if history2 is not None and boundary > 0:
            plt.axvline(
                x=boundary + 0.5,
                color="gray",
                linestyle="--",
                linewidth=1,
                label="phase boundary",
            )

        plt.title(title)
        plt.xlabel("Epoch")
        plt.legend(fontsize=8)

    plt.suptitle(f"{phase1_label}  +  {phase2_label}", fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices_grid(models_data, labels=("NORMAL", "PNEUMONIA")):
    """
    Display side-by-side confusion matrices for multiple models with counts and row percentages.

    Args:
        models_data: List of tuples (model_label, y_true, y_pred) for each model.
        labels: Tuple/list of class names corresponding to label indices.

    Behavior:
        - Plots a 1-row grid of heatmaps, one per model.
        - Each cell shows absolute count and row-normalized percentage.
        - Highlights the false negative cell (missed PNEUMONIA predicted as NORMAL) with a red border.
        - No colorbar to keep figures compact.
    """

    n = len(models_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]  # Ensure axes is iterable

    for ax, (label, y_true, y_pred) in zip(axes, models_data):
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # Row-normalized

        # Prepare annotations with counts and percentages
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"

        # Plot heatmap
        sns.heatmap(
            cm_norm,
            annot=annot,
            fmt="",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            vmin=0,
            vmax=1,
            ax=ax,
            cbar=False,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(label)

        # Highlight false negative (missed PNEUMONIA) with red rectangle
        fn_row, fn_col = 1, 0  # actual=PNEUMONIA (row 1), predicted=NORMAL (col 0)
        rect = plt.Rectangle((fn_col, fn_row), 1, 1, fill=False, edgecolor="red", linewidth=2)
        ax.add_patch(rect)

    plt.suptitle(
        "Confusion Matrices  (counts + row %, red = missed pneumonia)",
        fontsize=11
    )
    plt.tight_layout()
    plt.show()


def plot_roc_curves(models_data):
    """
    Plot overlaid ROC curves for multiple models with AUC annotations and tuned thresholds.

    Args:
        models_data: List of tuples (label, y_true, y_prob, threshold) for each model.

    Behavior:
        - Computes ROC curve (FPR, TPR) for each model using sklearn.metrics.roc_curve.
        - Calculates AUC for each model and annotates it in the legend.
        - Marks the operating point closest to the model's tuned threshold with a scatter point.
        - Adds a diagonal reference line for a random classifier.
        - Uses distinct colors for each model, up to 10 models (tab10 colormap).
        - Configures axis labels, title, legend, and layout for clarity.
    """

    plt.figure(figsize=(7, 6))
    colors = plt.cm.tab10.colors  # Up to 10 distinct colors

    for (label, y_true, y_prob, threshold), color in zip(models_data, colors):
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f"{label}  (AUC = {roc_auc:.3f})")

        # Highlight operating point corresponding to tuned threshold
        idx = np.argmin(np.abs(thresholds - threshold))
        plt.scatter(fpr[idx], tpr[idx], color=color, s=80, zorder=5)

    # Diagonal reference line (random classifier)
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")

    plt.xlabel("False Positive Rate (1 – Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title("ROC Curves – Test Set")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_augmented_samples(
    augmentation_layer,
    sample_image,
    n: int = 8,
):
    """
    Display multiple augmented versions of a single input image using a Keras augmentation layer.

    Args:
        augmentation_layer: A tf.keras.layers.Layer or Sequential that applies data augmentation.
        sample_image: Input image as a numpy array or tf.Tensor, shape (H, W, C) or (1, H, W, C).
        n (int): Number of augmented samples to display.

    Behavior:
        - Converts numpy input to tf.Tensor if necessary.
        - Adds batch dimension if image is single (3D) tensor.
        - Generates `n` augmented images by calling the augmentation layer in training mode.
        - Automatically scales images to [0,1] for display if necessary.
        - Arranges images in a grid (4 columns by computed rows) with titles "Aug 1", "Aug 2", etc.
        - Turns off axes and adjusts layout for clean visualization.
    """

    # Convert numpy image to tensor if needed
    if isinstance(sample_image, np.ndarray):
        sample_tensor = tf.convert_to_tensor(sample_image)
    else:
        sample_tensor = sample_image

    # Ensure batch dimension exists
    if sample_tensor.ndim == 3:
        sample_tensor = tf.expand_dims(sample_tensor, axis=0)

    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for idx in range(rows * cols):
        ax = axes[idx]
        if idx < n:
            # Generate augmented image in training mode
            aug_img = augmentation_layer(sample_tensor, training=True)[0]

            # Scale to [0,1] if necessary
            if tf.reduce_max(aug_img) > 1.5:
                aug_img = aug_img / 255.0

            ax.imshow(tf.clip_by_value(aug_img, 0.0, 1.0))
            ax.set_title(f"Aug {idx + 1}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """
    Plot training and validation curves for accuracy, loss, and AUC from a Keras History object.

    Args:
        history: Keras History object returned by model.fit().

    Behavior:
        - Creates a 1x3 subplot layout:
            1. Accuracy over epochs (train vs validation)
            2. Loss over epochs (train vs validation)
            3. AUC over epochs (train vs validation)
        - Adds labels, titles, and legends for clarity.
        - Adjusts layout to prevent overlap and displays the plots.
    """

    plt.figure(figsize=(15, 4))

    # Accuracy subplot
    plt.subplot(1, 3, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss subplot
    plt.subplot(1, 3, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # AUC subplot
    plt.subplot(1, 3, 3)
    plt.plot(history.history["auc"], label="Train AUC")
    plt.plot(history.history["val_auc"], label="Val AUC")
    plt.title("AUC Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_and_report(y_true, y_pred, best_threshold):
    """
    Plot a confusion matrix heatmap and print a detailed classification report.

    Args:
        y_true: Array-like of true labels (0=NORMAL, 1=PNEUMONIA).
        y_pred: Array-like of predicted labels using the threshold.
        best_threshold: Float, the decision threshold used for predictions.

    Behavior:
        - Computes confusion matrix (counts only) using sklearn.metrics.confusion_matrix.
        - Plots a heatmap of the confusion matrix with class labels "NORMAL" and "PNEUMONIA".
        - Disables colorbar for a cleaner visualization.
        - Titles the plot with the applied threshold for context.
        - Prints the classification report showing precision, recall, F1-score for each class.
        - Handles zero-division safely in metrics.
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["NORMAL", "PNEUMONIA"],
        yticklabels=["NORMAL", "PNEUMONIA"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (Test, threshold={best_threshold:.2f})")
    plt.tight_layout()
    plt.show()

    # Print classification metrics
    print("Classification Report (Test):\n")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["NORMAL", "PNEUMONIA"],
        zero_division=0
    ))

def plot_probability_distribution(y_true, y_prob, best_threshold):
    """
    Plot the predicted probability distributions for NORMAL vs PNEUMONIA samples
    and mark the decision threshold.

    Args:
        y_true: Array-like of true labels (0=NORMAL, 1=PNEUMONIA).
        y_prob: Array-like of predicted probabilities for the PNEUMONIA class.
        best_threshold: Float, the tuned decision threshold for classification.

    Behavior:
        - Converts inputs to flattened NumPy arrays.
        - Separates predicted probabilities by true class (NORMAL vs PNEUMONIA).
        - Prints the number of samples in each class for reference.
        - Plots overlapping histograms for the two classes with semi-transparency.
        - Draws vertical lines for the best threshold and the default 0.5 threshold.
        - Labels axes, adds a legend, and titles the plot.
        - Uses `display(fig)` and closes the figure to integrate with notebook outputs cleanly.
    """

    y_true_arr = np.asarray(y_true).reshape(-1)
    y_prob_arr = np.asarray(y_prob).reshape(-1)

    # Split probabilities by true class
    normal_probs = y_prob_arr[y_true_arr == 0]
    pneumonia_probs = y_prob_arr[y_true_arr == 1]

    # Print sample counts
    print(f"NORMAL samples: {normal_probs.size} | PNEUMONIA samples: {pneumonia_probs.size}")

    # Plot histograms
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(normal_probs, bins=40, alpha=0.6, label="NORMAL", color="steelblue")
    ax.hist(pneumonia_probs, bins=40, alpha=0.6, label="PNEUMONIA", color="salmon")
    ax.axvline(best_threshold, color="black", linestyle="--", label=f"best_threshold={best_threshold:.2f}")
    ax.axvline(0.5, color="gray", linestyle=":", label="0.5 (hardcoded)")
    ax.set_xlabel("Predicted Probability (PNEUMONIA)")
    ax.set_ylabel("Count")
    ax.set_title("Test Set: Predicted Probability Distribution by True Class")
    ax.legend()
    fig.tight_layout()

    display(fig)
    plt.close(fig)


def plot_sample_test_predictions(test_ds, model, best_threshold):
    """
    Display a small set of sample test images with predicted labels and confidence scores.

    Args:
        test_ds: tf.data.Dataset of test images and labels.
        model: Trained Keras model to generate predictions.
        best_threshold: Decision threshold to convert probabilities into binary labels.

    Behavior:
        - Collects up to 5 NORMAL and 4 PNEUMONIA images from the test dataset.
        - Predicts probabilities using the model and converts to binary predictions.
        - Computes a confidence score: for PNEUMONIA predictions, it's the probability;
          for NORMAL predictions, it's 1 minus the probability.
        - Arranges images in a 3x3 grid with titles showing true label, predicted label, and confidence.
        - Ensures images are displayed in uint8 format for proper visualization.
    """

    normal_images, pneumonia_images = [], []
    normal_true, pneumonia_true = [], []

    # Collect a few samples per class
    for images, labels in test_ds:
        lbls = labels.numpy().ravel().astype(int)
        for i, lbl in enumerate(lbls):
            if lbl == 0 and len(normal_images) < 5:
                normal_images.append(images[i])
                normal_true.append(lbl)
            elif lbl == 1 and len(pneumonia_images) < 4:
                pneumonia_images.append(images[i])
                pneumonia_true.append(lbl)
        if len(normal_images) >= 5 and len(pneumonia_images) >= 4:
            break

    sample_images = tf.stack(normal_images + pneumonia_images)
    sample_true = np.array(normal_true + pneumonia_true, dtype=int)

    # Predict probabilities and convert to binary predictions
    sample_probs = model.predict(sample_images, verbose=0).ravel()
    sample_preds = (sample_probs >= best_threshold).astype(int)

    idx_to_class = {0: "NORMAL", 1: "PNEUMONIA"}

    # Plot grid of sample images with annotations
    plt.figure(figsize=(12, 12))
    plt.suptitle(f"Sample Test Predictions (threshold={best_threshold:.2f})", y=1.02)
    for i in range(len(sample_images)):
        img = sample_images[i].numpy().astype("uint8")
        true_label = idx_to_class[int(sample_true[i])]
        pred_label = idx_to_class[int(sample_preds[i])]
        conf = sample_probs[i] if sample_preds[i] == 1 else (1 - sample_probs[i])

        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.title(f"T:{true_label} | P:{pred_label}\nConf:{conf:.2f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def plot_model_comparison(SAVE_DIR, models):
    """
    Summarize and visualize test-set performance for multiple trained models.

    Args:
        SAVE_DIR: directory where model metadata JSON files are stored.
        models: list of (model_name, display_label) tuples.

    Behavior:
        - Loads metrics from each model's metadata JSON (accuracy, AUC, precision, recall, F1, threshold).
        - Prints a table of all models and their metrics.
        - Highlights the best model per metric in the console.
        - Plots a grouped bar chart of all key metrics per model with Accuracy also overlaid as a line.
        - Handles missing metadata gracefully by skipping the model.
    """
    records = []

    # Collect metrics for each model
    for model_name, label in models:
        meta = training_utils.load_model_meta(SAVE_DIR, model_name)
        if meta is None:
            print(f"  {label}: metadata not found, skipping")
            continue
        m = meta["metrics"]
        records.append({
            "Model": label,
            "Accuracy": round(m.get("accuracy", 0), 4),
            "AUC": round(m.get("auc", 0), 4),
            "Precision": round(m.get("precision", 0), 4),
            "Recall": round(m.get("recall", 0), 4),
            "F1": round(m.get("f1", 0), 4),
            "Threshold": round(meta.get("threshold", 0), 2),
        })

    if records:
        # Create DataFrame for display
        df = pd.DataFrame(records).set_index("Model")
        print(df.to_string())
        print()

        # Highlight best model per metric
        for col in ["Accuracy", "AUC", "Precision", "Recall", "F1"]:
            best_model = df[col].idxmax()
            print(f"  Best {col}: {best_model} ({df.loc[best_model, col]:.4f})")

        # Plot grouped bar chart
        ax = df[["Accuracy", "AUC", "Recall", "Precision", "F1"]].plot(
            kind="bar", figsize=(10, 5), colormap="tab10", edgecolor="white",
        )

        # Overlay accuracy as line plot
        x = ax.patches[0:len(df)]
        x_pos = [p.get_x() + p.get_width()/2 for p in x]
        ax.plot(
            x_pos,
            df["Accuracy"],
            color="red",
            marker="o",
            linewidth=2,
            label="Accuracy (line)"
        )

        ax.set_ylim(0.5, 1.02)
        ax.set_ylabel("Score")
        ax.set_title("CNN Model Comparison — Test Set Metrics")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
        ax.legend(loc="lower right", fontsize="x-small")
        plt.tight_layout()
        plt.show()
    else:
        print("No model metadata found. Run training cells first.")


def make_gradcam_heatmap(img_array, model, layer_name, inner_model_name=None, pred_index=None):
    """
    Generate a Grad-CAM heatmap for a single image and target convolutional layer.

    Supports both:
      - Nested models (e.g., VGG16 inside a custom model) via `inner_model_name`.
      - Flat models (baseline CNNs).

    Args:
        img_array: preprocessed image tensor with shape (1, H, W, C).
        model: tf.keras.Model containing the layer of interest.
        layer_name: name of the convolutional layer to visualize.
        inner_model_name: optional, name of nested inner model (e.g., "vgg16") for complex architectures.
        pred_index: optional, index of the class to compute Grad-CAM for; defaults to predicted class.

    Returns:
        heatmap: 2D numpy array (H x W) normalized to [0, 1].
    
    Notes:
        - Uses GradientTape to compute gradients of the class output w.r.t. last conv layer.
        - Handles layers that do not accept `training` keyword (e.g., TFOpLambda preprocessing).
        - Pools gradients spatially and weights the conv output to generate heatmap.
    """
    if inner_model_name:
        with tf.GradientTape() as tape:
            x = img_array
            last_conv_layer_output = None
            
            # Forward pass manually through layers
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                    
                if layer.name == inner_model_name:
                    # Nested model: extract intermediate conv output + final output
                    inner_grad_model = tf.keras.Model(
                        inputs=layer.inputs, 
                        outputs=[layer.get_layer(layer_name).output, layer.output]
                    )
                    last_conv_layer_output, x = inner_grad_model(x, training=False)
                else:
                    # Standard layers (augmentation, dense, etc.)
                    try:
                        x = layer(x, training=False)
                    except TypeError:
                        x = layer(x)
            
            preds = x
            if pred_index is None:
                pred_index = tf.argmax(preds[0]) if preds.shape[-1] > 1 else 0
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        
    else:
        # Flat model
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [model.get_layer(layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array, training=False)
            if pred_index is None:
                pred_index = tf.argmax(preds[0]) if preds.shape[-1] > 1 else 0
            class_channel = preds[:, pred_index]
        grads = tape.gradient(class_channel, last_conv_layer_output)

    # Grad-CAM: weight conv features by pooled gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def superimpose_heatmap_array(img_array, heatmap, alpha=0.4):
    """
    Superimpose a Grad-CAM heatmap onto an image array for visualization.

    Args:
        img_array: HxWxC image tensor or array, either [0,1] float or [0,255] uint8.
        heatmap: 2D numpy array of same aspect ratio as img_array (values [0,1]).
        alpha: blending factor, how strongly the heatmap overlays the image.

    Returns:
        superimposed_img: HxWxC uint8 array with heatmap blended on top.
    
    Notes:
        - Converts image to uint8 [0,255] if needed.
        - Resizes heatmap to match image dimensions.
        - Uses Matplotlib "jet" colormap for coloring heatmap.
        - Clips output to valid uint8 range.
    """
    # Ensure image is in [0, 255] uint8 format for visualization
    if tf.reduce_max(img_array) <= 1.0:
        img_display = np.uint8(255 * img_array)
    else:
        img_display = np.uint8(img_array)

    # Resize heatmap to match image size and scale to [0,255]
    heatmap = cv2.resize(heatmap, (img_display.shape[1], img_display.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    # Map heatmap through jet colormap
    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]  # RGB only, ignore alpha
    jet_heatmap = jet_colors[heatmap] * 255.0

    # Blend heatmap and original image
    superimposed_img = jet_heatmap * alpha + img_display
    return np.clip(superimposed_img, 0, 255).astype(np.uint8)

def compare_models_gradcam_dataset(test_dataset, models_info, samples_per_class=10):
    """
    Collect a fixed number of samples per class from a tf.data.Dataset and visualize Grad-CAM 
    heatmaps side-by-side for multiple models.

    Each row corresponds to a sample, with the first column showing the original image, and 
    subsequent columns showing Grad-CAM overlays for each model.

    Args:
        test_dataset: tf.data.Dataset yielding (image, label) batches.
        models_info: list of dicts for each model with keys:
            - 'model': Keras model object
            - 'layer': name of the last conv layer for Grad-CAM
            - 'name': display label for the model
            - 'inner_model': optional, name of nested base model for Grad-CAM
            - 'preprocess': optional, callable to preprocess images (default: scale to [0,1])
        samples_per_class: number of NORMAL and PNEUMONIA images to collect.

    Notes:
        - Assumes label 0 = NORMAL, 1 = PNEUMONIA.
        - Displays titles in green if prediction matches true label, red otherwise.
    """
    collected_images, collected_labels = [], []
    count_normal = count_pneumonia = 0
    
    # Collect samples
    for img, lbl in test_dataset.unbatch():
        label_val = int(np.asarray(lbl).flatten()[0])
        if label_val == 0 and count_normal < samples_per_class:
            collected_images.append(img)
            collected_labels.append(lbl)
            count_normal += 1
        elif label_val == 1 and count_pneumonia < samples_per_class:
            collected_images.append(img)
            collected_labels.append(lbl)
            count_pneumonia += 1
        if count_normal == samples_per_class and count_pneumonia == samples_per_class:
            break

    num_samples = len(collected_images)
    if num_samples == 0:
        print("No samples found in the dataset.")
        return

    num_models = len(models_info)
    fig, axes = plt.subplots(num_samples, num_models + 1, 
                             figsize=(5 * (num_models + 1), 5 * num_samples))

    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    # Loop over each sample
    for row_idx in range(num_samples):
        img_tensor = collected_images[row_idx]
        true_label = int(np.asarray(collected_labels[row_idx]).flatten()[0])
        true_label_name = "Pneumonia" if true_label == 1 else "Normal"

        # Original image
        display_img = np.uint8(img_tensor.numpy() * 255) if tf.reduce_max(img_tensor) <= 1.0 else np.uint8(img_tensor.numpy())
        axes[row_idx, 0].imshow(display_img)
        axes[row_idx, 0].set_title(f"Original X-Ray\nTrue: {true_label_name}", fontsize=14)
        axes[row_idx, 0].axis('off')

        # Grad-CAM for each model
        for col_idx, info in enumerate(models_info):
            model = info['model']
            layer_name = info['layer']
            name = info['name']
            preprocess_func = info.get('preprocess', lambda x: x / 255.0)
            inner_model_name = info.get('inner_model', None)

            img_array_expanded = tf.expand_dims(img_tensor, axis=0)
            processed_img = preprocess_func(tf.cast(img_array_expanded, tf.float32))

            preds = model.predict(processed_img, verbose=0)
            pred_score = preds[0][0] if preds.shape[-1] == 1 else preds[0][1]
            pred_label = "Pneumonia" if pred_score > 0.5 else "Normal"

            heatmap = make_gradcam_heatmap(processed_img, model, layer_name, inner_model_name=inner_model_name)
            cam_img = superimpose_heatmap_array(img_tensor.numpy(), heatmap)

            ax = axes[row_idx, col_idx + 1]
            ax.imshow(cam_img)
            color = "green" if pred_label == true_label_name else "red"
            ax.set_title(f"{name}\nPred: {pred_label}", fontsize=14, color=color)
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def get_layer_location(model):
    """
    Returns a tuple: (inner_model_name, conv_layer_name).
    - inner_model_name: Name of the nested model if the last Conv2D layer is inside a sub-model; otherwise None.
    - conv_layer_name: Name of the last convolutional layer found.
    
    This is useful for Grad-CAM visualization where you need the last convolutional layer.
    """
    # Iterate over layers in reverse order to find the last Conv2D layer first
    for layer in reversed(model.layers):
        # If this layer is a top-level Conv2D, return it directly
        if isinstance(layer, tf.keras.layers.Conv2D):
            return None, layer.name
        
        # If this layer is a nested Keras Model (e.g., VGG16 inside a custom model)
        if isinstance(layer, tf.keras.Model):
            try:
                # Iterate over inner layers in reverse to find the last Conv2D
                for inner_layer in reversed(layer.layers):
                    if isinstance(inner_layer, tf.keras.layers.Conv2D):
                        # Return the nested model's name and the inner Conv2D layer's name
                        return layer.name, inner_layer.name
            except ValueError:
                # Skip any nested models that may raise errors during access
                continue
                
    # Raise an error if no Conv2D layers were found in the entire model
    raise ValueError(f"Could not find any Conv2D layer in model: {model.name}")


def baseline_preprocess(x):
    """
    Simple preprocessing function for baseline CNN models.
    
    Converts pixel values from [0, 255] range to [0, 1] for model input.
    """
    return x / 255.0

def prepare_gradcam_configs(model_definitions):
    """
    Prepares Grad-CAM configuration dictionaries for multiple models.

    Args:
        model_definitions (dict): Keys are model names, values are paths to saved Keras models.

    Returns:
        List of dicts containing:
            - 'name': model name
            - 'model': loaded Keras model
            - 'inner_model': name of nested model if the last conv layer is inside one; None otherwise
            - 'layer': name of the last Conv2D layer to use for Grad-CAM
            - 'preprocess': preprocessing function to apply before feeding input to model
    """
    configs = []

    # Iterate through each model definition
    for name, (path) in model_definitions.items():
        print(f"Loading {name}...")  # Log which model is being loaded
        try:
            # Load the Keras model with compatibility fixes (e.g., for augmentation layers)
            model = training_utils.load_model_compat(path)

            # Identify the last convolutional layer for Grad-CAM
            inner_name, conv_name = get_layer_location(model)

            # Build a readable string to show location
            location_str = f"{inner_name} -> {conv_name}" if inner_name else conv_name
            print(f"  -> Found last conv layer: {location_str}")

            # Append config dict for this model
            configs.append({
                'name': name,
                'model': model,
                'inner_model': inner_name,
                'layer': conv_name,
                'preprocess': lambda x: x  # Placeholder; adjust if model requires preprocessing
            })

        except Exception as e:
            # Print error but continue with other models
            print(f"  -> Failed: {e}")

    return configs