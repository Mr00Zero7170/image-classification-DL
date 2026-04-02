"""
predict.py
----------
Run inference on a single image (file path or URL) using a saved model.
Optionally launches a simple Gradio web UI for interactive demo.

Usage
─────
# Predict from a local file:
  python src/predict.py --image path/to/cat.jpg --model models/cnn_final.keras

# Predict from a URL:
  python src/predict.py --image https://example.com/dog.jpg --model models/mobilenet_final.keras

# Launch an interactive web UI (requires: pip install gradio):
  python src/predict.py --ui --model models/cnn_final.keras
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import CIFAR10_CLASSES


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Image classification inference")
    p.add_argument("--model", required=True, help="Path to a saved .keras model file")
    p.add_argument("--image", default=None,  help="Path or URL of the image to classify")
    p.add_argument("--top_k", type=int, default=3, help="Show top-k predictions")
    p.add_argument(
        "--ui",
        action="store_true",
        help="Launch a Gradio web UI instead of running a single prediction",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Image loading & preprocessing
# ─────────────────────────────────────────────────────────────

def load_image_from_path(path: str) -> Image.Image:
    """Load an image from a local path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def load_image_from_url(url: str) -> Image.Image:
    """Download and load an image from a URL."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def load_image(source: str) -> Image.Image:
    """Auto-detect whether `source` is a URL or a file path."""
    if source.startswith("http://") or source.startswith("https://"):
        return load_image_from_url(source)
    return load_image_from_path(source)


def preprocess_image(img: Image.Image, target_size: tuple) -> np.ndarray:
    """
    Resize, normalise, and add the batch dimension.

    Returns
    -------
    np.ndarray of shape (1, H, W, 3) with float32 values in [0, 1].
    """
    img_resized = img.resize(target_size, Image.LANCZOS)
    arr = np.array(img_resized, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)          # (H, W, 3) → (1, H, W, 3)


# ─────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────

def load_model_and_meta(model_path: str):
    """
    Load a Keras model and its companion metadata JSON (if present).

    The metadata JSON tells us:
      • input_shape : (H, W, C) that the model expects
      • classes     : list of class-name strings

    Falls back to CIFAR-10 defaults if JSON is not found.
    """
    print(f"[load] Loading model from {model_path} …")
    model = tf.keras.models.load_model(model_path)
    print("[load] Model loaded successfully.")

    # Try to read companion metadata
    meta_path = Path(model_path).with_suffix("").as_posix().replace(
        "_final", ""
    ) + "_metadata.json"

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        input_shape = tuple(meta["input_shape"])   # e.g. (32, 32, 3)
        class_names = meta["classes"]
        print(f"[load] Metadata loaded: {len(class_names)} classes, input {input_shape}")
    else:
        # Safe defaults
        input_shape = model.input_shape[1:]        # strip batch dim
        class_names = CIFAR10_CLASSES
        print("[load] No metadata file found – using CIFAR-10 defaults.")

    return model, input_shape, class_names


# ─────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────

def predict(model, image: Image.Image, input_shape: tuple, class_names: list, top_k: int = 3):
    """
    Run the model on one PIL Image and return a list of (class_name, probability)
    tuples for the top-k predictions.
    """
    h, w = input_shape[0], input_shape[1]
    x = preprocess_image(image, target_size=(w, h))

    probs = model.predict(x, verbose=0)[0]         # shape: (num_classes,)

    # Get top-k indices sorted by probability (descending)
    top_indices = np.argsort(probs)[::-1][:top_k]
    return [(class_names[i], float(probs[i])) for i in top_indices]


def print_predictions(results: list):
    """Pretty-print prediction results to stdout."""
    print("\n" + "─" * 40)
    print("  Prediction Results")
    print("─" * 40)
    for rank, (cls, prob) in enumerate(results, start=1):
        bar = "█" * int(prob * 30)
        print(f"  {rank}. {cls:<12}  {prob*100:5.1f}%  {bar}")
    print("─" * 40 + "\n")


# ─────────────────────────────────────────────────────────────
# Gradio web UI  (optional – only imported if --ui is passed)
# ─────────────────────────────────────────────────────────────

def launch_gradio_ui(model, input_shape: tuple, class_names: list):
    """
    Launch a simple Gradio interface for interactive predictions.
    Install with: pip install gradio
    """
    try:
        import gradio as gr
    except ImportError:
        print("Gradio is not installed. Run:  pip install gradio")
        sys.exit(1)

    def classify_image(pil_img):
        """Gradio callback: receives a PIL Image, returns a label dict."""
        if pil_img is None:
            return {}
        results = predict(model, pil_img, input_shape, class_names, top_k=5)
        return {cls: float(f"{prob:.4f}") for cls, prob in results}

    demo = gr.Interface(
        fn=classify_image,
        inputs=gr.Image(type="pil", label="Upload an image"),
        outputs=gr.Label(num_top_classes=5, label="Predictions"),
        title="CIFAR-10 Image Classifier",
        description=(
            "Upload any image and the model will predict which of the 10 CIFAR-10 "
            "categories it belongs to: airplane, automobile, bird, cat, deer, "
            "dog, frog, horse, ship, or truck."
        ),
        examples=[],      # add example image paths here if desired
        allow_flagging="never",
    )
    demo.launch(share=False)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    model, input_shape, class_names = load_model_and_meta(args.model)

    if args.ui:
        # ── Interactive web UI ────────────────────────────────
        launch_gradio_ui(model, input_shape, class_names)
    else:
        # ── Single-image prediction ───────────────────────────
        if args.image is None:
            print("Error: --image is required when not using --ui")
            sys.exit(1)

        print(f"[predict] Loading image: {args.image}")
        img = load_image(args.image)
        print(f"[predict] Image size: {img.size}  mode: {img.mode}")

        results = predict(model, img, input_shape, class_names, top_k=args.top_k)
        print_predictions(results)

        # Save a copy of the prediction for logging
        top_class, top_prob = results[0]
        print(f"Final answer: \"{top_class}\"  ({top_prob*100:.1f}% confidence)")


if __name__ == "__main__":
    main()
