"""Visualization utilities for training and evaluation."""

from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_training_curves(history):
    """Plot accuracy/loss/AUC train-vs-val curves."""
    if history is None or not getattr(history, "history", None):
        print("No history available to plot.")
        return

    hist = history.history
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 3, 1)
    if "accuracy" in hist:
        plt.plot(hist["accuracy"], label="train")
    if "val_accuracy" in hist:
        plt.plot(hist["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 3, 2)
    if "loss" in hist:
        plt.plot(hist["loss"], label="train")
    if "val_loss" in hist:
        plt.plot(hist["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 3, 3)
    if "auc" in hist:
        plt.plot(hist["auc"], label="train")
    if "val_auc" in hist:
        plt.plot(hist["val_auc"], label="val")
    plt.title("AUC")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels=("NORMAL", "PNEUMONIA")):
    """Plot labeled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
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
    """Show random test samples with predicted class + probability."""
    images_all, labels_all = [], []
    for batch_images, batch_labels in test_ds:
        images_all.append(batch_images.numpy())
        labels_all.append(batch_labels.numpy().ravel().astype(int))

    images_all = np.concatenate(images_all)
    labels_all = np.concatenate(labels_all)
    probs = model.predict(images_all, verbose=0).ravel()
    preds = (probs >= threshold).astype(int)

    sample_idx = random.sample(range(len(images_all)), k=min(n, len(images_all)))
    cols = 3
    rows = int(np.ceil(len(sample_idx) / cols))
    plt.figure(figsize=(5 * cols, 4 * rows))

    for i, idx in enumerate(sample_idx):
        ax = plt.subplot(rows, cols, i + 1)
        img = images_all[idx]
        # Assume data may be either in [0,255] or preprocessed; clip for display.
        img_display = img
        if img_display.max() > 1.5:
            img_display = img_display / 255.0
        ax.imshow(np.clip(img_display, 0.0, 1.0))
        true_lbl = "PNEUMONIA" if labels_all[idx] == 1 else "NORMAL"
        pred_lbl = "PNEUMONIA" if preds[idx] == 1 else "NORMAL"
        ax.set_title(f"T:{true_lbl} | P:{pred_lbl}\nprob={probs[idx]:.3f}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_combined_training_curves(history1, history2=None,
                                   phase1_label="Phase 1 – Frozen Base",
                                   phase2_label="Phase 2 – Fine-tune"):
    """Plot stitched accuracy/loss/AUC curves across one or two training phases.

    If history2 is provided, its curves are appended after history1 and a
    vertical dashed line marks the phase boundary.
    """
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

        train_vals = list(_get(history1, train_key))
        val_vals   = list(_get(history1, val_key))
        boundary   = len(train_vals)

        if history2 is not None:
            train_vals += list(_get(history2, train_key))
            val_vals   += list(_get(history2, val_key))

        epochs = range(1, len(train_vals) + 1)
        if train_vals:
            plt.plot(epochs, train_vals, label="train")
        if val_vals:
            plt.plot(epochs, val_vals,   label="val")
        if history2 is not None and boundary > 0:
            plt.axvline(x=boundary + 0.5, color="gray", linestyle="--",
                        linewidth=1, label="phase boundary")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.legend(fontsize=8)

    plt.suptitle(f"{phase1_label}  +  {phase2_label}", fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices_grid(models_data, labels=("NORMAL", "PNEUMONIA")):
    """Render side-by-side confusion matrices for multiple models.

    Args:
        models_data: list of (model_label, y_true, y_pred) tuples.
        labels: class name sequence matching label indices.
    """
    n = len(models_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (label, y_true, y_pred) in zip(axes, models_data):
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        # Build annotation: counts + row-normalised %
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"

        sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues",
                    xticklabels=labels, yticklabels=labels,
                    vmin=0, vmax=1, ax=ax, cbar=False)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(label)

        # Highlight FN cell (missed pneumonia) with a red border
        fn_row, fn_col = 1, 0          # actual=PNEUMONIA, predicted=NORMAL
        for spine in ax.patches:
            pass
        rect = plt.Rectangle((fn_col, fn_row), 1, 1,
                              fill=False, edgecolor="red", linewidth=2)
        ax.add_patch(rect)

    plt.suptitle("Confusion Matrices  (counts + row %,  red = missed pneumonia)",
                 fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_roc_curves(models_data):
    """Overlay ROC curves with AUC annotations and tuned operating points.

    Args:
        models_data: list of (label, y_true, y_prob, threshold) tuples.
    """
    plt.figure(figsize=(7, 6))
    colors = plt.cm.tab10.colors

    for (label, y_true, y_prob, threshold), color in zip(models_data, colors):
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f"{label}  (AUC = {roc_auc:.3f})")

        # Mark operating point closest to the tuned threshold
        idx = np.argmin(np.abs(thresholds - threshold))
        plt.scatter(fpr[idx], tpr[idx], color=color, s=80, zorder=5)

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
    """Visualize n augmented versions of one sample image."""
    if isinstance(sample_image, np.ndarray):
        sample_tensor = tf.convert_to_tensor(sample_image)
    else:
        sample_tensor = sample_image

    if sample_tensor.ndim == 3:
        sample_tensor = tf.expand_dims(sample_tensor, axis=0)

    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for idx in range(rows * cols):
        ax = axes[idx]
        if idx < n:
            aug_img = augmentation_layer(sample_tensor, training=True)[0]
            if tf.reduce_max(aug_img) > 1.5:
                aug_img = aug_img / 255.0
            ax.imshow(tf.clip_by_value(aug_img, 0.0, 1.0))
            ax.set_title(f"Aug {idx + 1}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
