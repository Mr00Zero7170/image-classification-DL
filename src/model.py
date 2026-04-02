"""
model.py
--------
Defines two model architectures:
  1. CustomCNN    – a lightweight CNN built from scratch (great for learning).
  2. TransferModel – MobileNetV2 backbone with a custom head (production-ready).

Both expose the same .build() interface so train.py can swap them with one flag.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications, regularizers
from tensorflow.keras.optimizers import Adam


# ─────────────────────────────────────────────────────────────
# 1.  Custom CNN  (built from scratch)
# ─────────────────────────────────────────────────────────────

class CustomCNN:
    """
    A moderately deep CNN with three convolutional blocks followed by
    two fully-connected layers.

    Architecture overview
    ─────────────────────
    Input (32×32×3)
      └─ Conv Block 1: Conv2D(32) → BN → ReLU → MaxPool → Dropout
      └─ Conv Block 2: Conv2D(64) → BN → ReLU → MaxPool → Dropout
      └─ Conv Block 3: Conv2D(128) → BN → ReLU → MaxPool → Dropout
      └─ Flatten
      └─ Dense(256) → BN → ReLU → Dropout(0.5)
      └─ Dense(num_classes) → Softmax
    """

    def __init__(self, input_shape: tuple, num_classes: int, dropout_rate: float = 0.3):
        """
        Parameters
        ----------
        input_shape  : (H, W, C)  e.g. (32, 32, 3)
        num_classes  : number of output categories
        dropout_rate : spatial dropout applied after each pool layer
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

    def _conv_block(self, x, filters: int, block_id: int):
        """Single (Conv → BN → ReLU → Pool → Dropout) block."""
        x = layers.Conv2D(
            filters,
            kernel_size=(3, 3),
            padding="same",
            kernel_regularizer=regularizers.l2(1e-4),
            name=f"conv{block_id}_conv",
        )(x)
        x = layers.BatchNormalization(name=f"conv{block_id}_bn")(x)
        x = layers.Activation("relu", name=f"conv{block_id}_relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name=f"conv{block_id}_pool")(x)
        x = layers.Dropout(self.dropout_rate, name=f"conv{block_id}_drop")(x)
        return x

    def build(self) -> tf.keras.Model:
        """Assemble and return the compiled Keras model."""
        inputs = layers.Input(shape=self.input_shape, name="input_image")

        x = self._conv_block(inputs, filters=32,  block_id=1)
        x = self._conv_block(x,      filters=64,  block_id=2)
        x = self._conv_block(x,      filters=128, block_id=3)

        x = layers.Flatten(name="flatten")(x)
        x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4), name="fc1")(x)
        x = layers.BatchNormalization(name="fc1_bn")(x)
        x = layers.Activation("relu", name="fc1_relu")(x)
        x = layers.Dropout(0.5, name="fc1_drop")(x)

        outputs = layers.Dense(self.num_classes, activation="softmax", name="predictions")(x)

        model = models.Model(inputs, outputs, name="CustomCNN")
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


# ─────────────────────────────────────────────────────────────
# 2.  Transfer Learning Model  (MobileNetV2 backbone)
# ─────────────────────────────────────────────────────────────

class TransferModel:
    """
    Uses a pre-trained MobileNetV2 (ImageNet weights) as a frozen feature
    extractor, topped with a custom classifier head.

    Two-phase training strategy
    ───────────────────────────
    Phase 1 – Warm-up:   backbone frozen, only the head is trained.
    Phase 2 – Fine-tune: top N layers of backbone unfrozen, trained at
                         a lower learning rate.

    Call build() then unfreeze(n_layers) when ready for phase 2.
    """

    def __init__(
        self,
        input_shape: tuple,
        num_classes: int,
        dropout_rate: float = 0.4,
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

    def build(self) -> tf.keras.Model:
        """
        Build the model with the MobileNetV2 backbone fully frozen.
        Returns the compiled Keras model (ready for phase-1 training).
        """
        # MobileNetV2 requires minimum 96×96 input; we'll upsample smaller images.
        backbone = applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,       # remove ImageNet classifier head
            weights="imagenet",      # start from pre-trained weights
        )
        backbone.trainable = False   # freeze all backbone weights initially

        inputs  = layers.Input(shape=self.input_shape, name="input_image")

        # Preprocess pixels to the range MobileNetV2 expects ([-1, 1])
        x = applications.mobilenet_v2.preprocess_input(inputs)

        # Feature extraction (no training yet)
        x = backbone(x, training=False)

        # Classifier head
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = layers.Dense(256, name="fc1")(x)
        x = layers.BatchNormalization(name="fc1_bn")(x)
        x = layers.Activation("relu", name="fc1_relu")(x)
        x = layers.Dropout(self.dropout_rate, name="fc1_drop")(x)
        outputs = layers.Dense(self.num_classes, activation="softmax", name="predictions")(x)

        model = models.Model(inputs, outputs, name="MobileNetV2_Transfer")
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Keep a reference to the backbone so we can unfreeze it later
        self._backbone = backbone
        self._model    = model
        return model

    def unfreeze(self, n_layers: int = 30, learning_rate: float = 1e-5) -> None:
        """
        Unfreeze the last `n_layers` of the MobileNetV2 backbone for
        fine-tuning (phase 2).  Recompiles at a lower learning rate.

        Parameters
        ----------
        n_layers      : how many backbone layers to unfreeze from the top
        learning_rate : typically 10-100× smaller than phase-1 LR
        """
        self._backbone.trainable = True

        # Freeze all layers except the last n_layers
        for layer in self._backbone.layers[:-n_layers]:
            layer.trainable = False

        trainable_count = sum(1 for l in self._backbone.layers if l.trainable)
        print(f"[unfreeze] {trainable_count} backbone layers now trainable.")

        self._model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )


# ─────────────────────────────────────────────────────────────
# Quick sanity-check
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== CustomCNN ===")
    cnn = CustomCNN(input_shape=(32, 32, 3), num_classes=10).build()
    cnn.summary()

    print("\n=== MobileNetV2 Transfer Model ===")
    transfer = TransferModel(input_shape=(96, 96, 3), num_classes=10)
    m = transfer.build()
    m.summary()
