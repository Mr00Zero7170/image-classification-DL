"""
utils.py
--------
Shared helper functions used across train.py, predict.py, and the notebook.
Covers:
  • CIFAR-10 loading & normalisation
  • Data augmentation pipeline
  • Plotting training curves
  • Confusion-matrix visualisation
  • Classification report pretty-print
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend – safe for servers & CI
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ── CIFAR-10 class labels ─────────────────────────────────────────────────────
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ─────────────────────────────────────────────────────────────
# 1.  Data loading
# ─────────────────────────────────────────────────────────────

def load_cifar10(val_split: float = 0.1, target_size: tuple = None):
    """
    Download (or load cached) CIFAR-10 and return train / val / test splits.

    Parameters
    ----------
    val_split   : fraction of the training set held out for validation
    target_size : if given as (H, W), images are resized to that shape.
                  Use (96, 96) for MobileNetV2.

    Returns
    -------
    (x_train, y_train), (x_val, y_val), (x_test, y_test)
    All pixel values are float32 in [0, 1].
    """
    (x_train_full, y_train_full), (x_test, y_test) = (
        tf.keras.datasets.cifar10.load_data()
    )

    # Flatten label arrays: (N, 1) → (N,)
    y_train_full = y_train_full.flatten()
    y_test       = y_test.flatten()

    # Resize if a target size was requested (needed for MobileNetV2 ≥ 96×96)
    if target_size is not None:
        x_train_full = _batch_resize(x_train_full, target_size)
        x_test       = _batch_resize(x_test,       target_size)

    # Normalise to [0, 1]
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test       = x_test.astype("float32") / 255.0

    # Train / validation split
    n_val = int(len(x_train_full) * val_split)
    x_val,   y_val   = x_train_full[:n_val],  y_train_full[:n_val]
    x_train, y_train = x_train_full[n_val:],  y_train_full[n_val:]

    print(
        f"[data] train={len(x_train):,}  val={len(x_val):,}  test={len(x_test):,}  "
        f"shape={x_train.shape[1:]}"
    )
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def _batch_resize(images: np.ndarray, target_size: tuple) -> np.ndarray:
    """Resize a batch of images using tf.image (GPU-accelerated if available)."""
    h, w = target_size
    resized = tf.image.resize(images, [h, w])
    return resized.numpy().astype("uint8")


# ─────────────────────────────────────────────────────────────
# 2.  Data augmentation
# ─────────────────────────────────────────────────────────────

def get_augmented_generator(
    x_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int = 64,
) -> tf.keras.preprocessing.image.NumpyArrayIterator:
    """
    Build a Keras ImageDataGenerator with standard augmentations and
    return a generator that yields (batch_images, batch_labels).

    Augmentations applied
    ─────────────────────
    • Horizontal flip        – accounts for left/right symmetry
    • Width / height shift   – simulates camera position changes
    • Rotation (±15°)        – slight orientation invariance
    • Zoom (±10%)            – scale invariance
    • Channel shift          – mild colour jitter
    """
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=15,
        zoom_range=0.1,
        channel_shift_range=0.1,
        fill_mode="nearest",
    )
    datagen.fit(x_train)
    return datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True)


# ─────────────────────────────────────────────────────────────
# 3.  Plotting helpers
# ─────────────────────────────────────────────────────────────

def plot_training_history(history, save_path: str = "results/training_curves.png"):
    """
    Save a 2-panel figure showing loss and accuracy over epochs.

    Parameters
    ----------
    history   : tf.keras.callbacks.History object returned by model.fit()
    save_path : file path for the saved figure
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = range(1, len(history.history["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Loss ──────────────────────────────────────────────────
    ax1.plot(epochs, history.history["loss"],     label="Train loss",      linewidth=2)
    ax1.plot(epochs, history.history["val_loss"], label="Val loss",        linewidth=2, linestyle="--")
    ax1.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Accuracy ──────────────────────────────────────────────
    ax2.plot(epochs, history.history["accuracy"],     label="Train accuracy", linewidth=2)
    ax2.plot(epochs, history.history["val_accuracy"], label="Val accuracy",   linewidth=2, linestyle="--")
    ax2.set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Training curves saved → {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    save_path: str = "results/confusion_matrix.png",
    normalise: bool = True,
):
    """
    Compute and save a heatmap of the confusion matrix.

    Parameters
    ----------
    y_true      : ground-truth integer labels
    y_pred      : predicted integer labels
    class_names : list of human-readable class names
    save_path   : where to write the figure
    normalise   : if True, show row percentages instead of raw counts
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    if normalise:
        cm_display = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt, vmax, title_suffix = ".2f", 1.0, " (normalised)"
    else:
        cm_display = cm
        fmt, vmax, title_suffix = "d", cm.max(), ""

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=vmax,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix{title_suffix}", fontsize=14, fontweight="bold", pad=14)
    ax.set_ylabel("True label",      fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Confusion matrix saved → {save_path}")


def plot_sample_predictions(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    n: int = 16,
    save_path: str = "results/sample_predictions.png",
):
    """
    Show a grid of n random test images with their true and predicted labels.
    Green title = correct, red title = wrong.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    idx    = np.random.choice(len(images), n, replace=False)
    cols   = 4
    rows   = n // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    for ax, i in zip(axes.flat, idx):
        img = np.clip(images[i], 0, 1)   # guard against float drift
        ax.imshow(img)
        ax.axis("off")
        color = "green" if y_pred[i] == y_true[i] else "red"
        ax.set_title(
            f"T: {class_names[y_true[i]]}\nP: {class_names[y_pred[i]]}",
            fontsize=8,
            color=color,
        )

    plt.suptitle("Sample Predictions  (green=correct, red=wrong)", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Sample predictions saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# 4.  Metrics report
# ─────────────────────────────────────────────────────────────

def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    save_path: str = "results/classification_report.txt",
):
    """
    Print a full per-class precision / recall / F1 report and save it
    to a text file.
    """
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(report)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write("Classification Report\n")
        f.write("=" * 60 + "\n")
        f.write(report)
    print(f"[report] Saved → {save_path}")
