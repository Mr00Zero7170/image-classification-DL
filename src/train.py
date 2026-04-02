"""
train.py
--------
Entry-point for training an image-classification model on CIFAR-10.

Usage
─────
# Train the custom CNN (fast, good for learning):
  python src/train.py --model cnn --epochs 50

# Train with MobileNetV2 transfer learning (better accuracy):
  python src/train.py --model mobilenet --epochs 30 --finetune

# Full example with custom hyper-parameters:
  python src/train.py --model cnn --epochs 60 --batch_size 128 --lr 0.001
"""

import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf

# ── make sure 'src/' is on the path when running from project root ────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import CustomCNN, TransferModel
from src.utils import (
    CIFAR10_CLASSES,
    load_cifar10,
    get_augmented_generator,
    plot_training_history,
    plot_confusion_matrix,
    plot_sample_predictions,
    print_classification_report,
)


# ─────────────────────────────────────────────────────────────
# CLI arguments
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an image classifier on CIFAR-10")

    p.add_argument(
        "--model",
        choices=["cnn", "mobilenet"],
        default="cnn",
        help="Architecture to use: 'cnn' (from scratch) or 'mobilenet' (transfer learning)",
    )
    p.add_argument("--epochs",     type=int,   default=50,    help="Training epochs")
    p.add_argument("--batch_size", type=int,   default=64,    help="Mini-batch size")
    p.add_argument("--lr",         type=float, default=1e-3,  help="Initial learning rate")
    p.add_argument("--val_split",  type=float, default=0.1,   help="Validation fraction")
    p.add_argument(
        "--finetune",
        action="store_true",
        help="(MobileNet only) unfreeze top 30 layers for fine-tuning after warm-up",
    )
    p.add_argument(
        "--finetune_epochs",
        type=int,
        default=20,
        help="Extra epochs for fine-tuning phase (only used with --finetune)",
    )
    p.add_argument(
        "--save_dir",
        default="models",
        help="Directory where the trained model is saved",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────

def build_callbacks(save_dir: str, model_name: str) -> list:
    """
    Returns a list of Keras callbacks:
      • ModelCheckpoint  – saves the best model (by val_accuracy)
      • EarlyStopping    – stops when val_loss stops improving
      • ReduceLROnPlateau– halves LR when val_loss plateaus
      • TensorBoard      – for live loss/acc monitoring
    """
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(save_dir, f"{model_name}_best.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1,
    )

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join("results", "tensorboard_logs", model_name),
        histogram_freq=1,
    )

    return [checkpoint, early_stop, reduce_lr, tensorboard]


# ─────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Reproducibility ──────────────────────────────────────
    tf.random.set_seed(42)
    np.random.seed(42)

    # ── GPU memory growth (prevents OOM on shared GPUs) ──────
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    print("\n" + "=" * 60)
    print(f" Model     : {args.model.upper()}")
    print(f" Epochs    : {args.epochs}")
    print(f" Batch size: {args.batch_size}")
    print(f" Fine-tune : {args.finetune}")
    print("=" * 60 + "\n")

    # ── 1. Load data ─────────────────────────────────────────
    # MobileNetV2 requires at least 96×96 inputs.
    target_size = (96, 96) if args.model == "mobilenet" else None

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10(
        val_split=args.val_split,
        target_size=target_size,
    )

    input_shape = x_train.shape[1:]   # (H, W, C)
    num_classes = 10

    # ── 2. Build model ───────────────────────────────────────
    if args.model == "cnn":
        model = CustomCNN(input_shape=input_shape, num_classes=num_classes).build()
        transfer_obj = None
    else:
        transfer_obj = TransferModel(input_shape=input_shape, num_classes=num_classes)
        model = transfer_obj.build()

    model.summary()

    # ── 3. Augmented data generator (training set only) ──────
    train_gen = get_augmented_generator(x_train, y_train, batch_size=args.batch_size)
    steps_per_epoch = len(x_train) // args.batch_size

    callbacks = build_callbacks(args.save_dir, args.model)

    # ── 4. Phase 1 training ──────────────────────────────────
    print("\n[Phase 1] Training with backbone frozen (or full CNN)…\n")
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    # ── 5. Fine-tuning (MobileNet only) ──────────────────────
    if args.finetune and transfer_obj is not None:
        print("\n[Phase 2] Fine-tuning top 30 backbone layers…\n")
        transfer_obj.unfreeze(n_layers=30, learning_rate=1e-5)

        fine_tune_history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=args.finetune_epochs,
            validation_data=(x_val, y_val),
            callbacks=build_callbacks(args.save_dir, f"{args.model}_finetune"),
            verbose=1,
        )
        # Merge histories for unified plots
        for key in history.history:
            history.history[key].extend(fine_tune_history.history[key])

    # ── 6. Evaluate on the held-out test set ─────────────────
    print("\n[Evaluation] Test set…")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"  Test loss    : {test_loss:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}  ({test_acc*100:.2f}%)")

    # ── 7. Predictions & metrics ─────────────────────────────
    y_prob = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    os.makedirs("results", exist_ok=True)
    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred, CIFAR10_CLASSES)
    plot_sample_predictions(x_test, y_test, y_pred, CIFAR10_CLASSES)
    print_classification_report(y_test, y_pred, CIFAR10_CLASSES)

    # ── 8. Save final model ──────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    final_path = os.path.join(args.save_dir, f"{args.model}_final.keras")
    model.save(final_path)
    print(f"\n[save] Final model saved → {final_path}")

    # ── 9. Persist training metadata ─────────────────────────
    meta = {
        "model"      : args.model,
        "epochs_run" : len(history.history["loss"]),
        "test_loss"  : float(test_loss),
        "test_accuracy": float(test_acc),
        "input_shape": list(input_shape),
        "num_classes": num_classes,
        "classes"    : CIFAR10_CLASSES,
    }
    meta_path = os.path.join(args.save_dir, f"{args.model}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[save] Metadata saved    → {meta_path}")
    print("\nTraining complete!\n")


if __name__ == "__main__":
    main()
