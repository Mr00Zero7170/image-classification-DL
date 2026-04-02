# Image Classification using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A **production-ready image classification pipeline** built with TensorFlow/Keras, demonstrating
both a custom Convolutional Neural Network (CNN) trained from scratch and a MobileNetV2
transfer learning approach on the CIFAR-10 benchmark dataset.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Dataset](#dataset)
4. [Project Structure](#project-structure)
5. [Model Architecture](#model-architecture)
6. [Training Results](#training-results)
7. [How to Run](#how-to-run)
8. [Future Improvements](#future-improvements)
9. [Contributing](#contributing)

---

## Project Overview

This project builds an end-to-end deep learning system that can identify objects in images
across 10 categories. It covers every stage of a real ML workflow:

- **Data loading & exploration** — automated download, class distribution analysis
- **Preprocessing** — normalisation, optional resizing for transfer learning
- **Augmentation** — flips, crops, rotations and colour jitter to improve generalisation
- **Two model architectures** — a lightweight CNN and a fine-tuned MobileNetV2
- **Training** — early stopping, learning-rate scheduling, checkpointing
- **Evaluation** — accuracy/loss curves, confusion matrix, per-class F1 scores
- **Inference** — single-image CLI prediction and optional Gradio web UI

---

## Problem Statement

Classifying images is one of the foundational tasks in computer vision.
Given a 32 × 32 pixel photograph, the model must decide which of 10 object
categories the image belongs to.

This is a **10-class multi-class classification** problem:

```
Input: RGB image (32×32 or 96×96 after upscaling)
Output: probability distribution over 10 classes
Loss: Sparse Categorical Cross-Entropy
Metric: Top-1 Accuracy
```

---

## Dataset

**CIFAR-10** — 60,000 colour images (50,000 train / 10,000 test), perfectly balanced.

| Class      | Examples                        |
|------------|---------------------------------|
| airplane   | jets, propeller planes, gliders |
| automobile | cars, SUVs                      |
| bird       | sparrows, eagles, parrots       |
| cat        | tabby, persian, siamese         |
| deer       | white-tailed deer, elk          |
| dog        | golden retriever, husky         |
| frog       | tree frog, bull frog            |
| horse      | stallion, mare                  |
| ship       | cargo ships, sailboats          |
| truck      | semi trucks, pickup trucks      |

The dataset is automatically downloaded by Keras on first run.
See [`dataset/README.md`](dataset/README.md) for custom-dataset instructions.

---

## Project Structure

```
image-classification/
│
├── dataset/
│   └── README.md               ← Dataset download / custom dataset guide
│
├── src/
│   ├── model.py                ← CustomCNN and TransferModel (MobileNetV2) classes
│   ├── train.py                ← Training script (CLI with argparse)
│   ├── predict.py              ← Inference script (file / URL / Gradio UI)
│   └── utils.py                ← Data loading, augmentation, plotting helpers
│
├── notebooks/
│   └── exploration.ipynb       ← Interactive walkthrough of the full pipeline
│
├── models/                     ← Saved model files (auto-created after training)
├── results/                    ← Plots and reports (auto-created after training)
│
├── requirements.txt
└── README.md
```

---

## Model Architecture

### Architecture 1 — Custom CNN (built from scratch)

```
Input: 32×32×3
  └─ Conv Block 1 (Conv2D 32 filters → BN → ReLU → MaxPool → Dropout 0.3)
  └─ Conv Block 2 (Conv2D 64 filters → BN → ReLU → MaxPool → Dropout 0.3)
  └─ Conv Block 3 (Conv2D 128 filters → BN → ReLU → MaxPool → Dropout 0.3)
  └─ Flatten
  └─ Dense(256) → BN → ReLU → Dropout(0.5)
  └─ Dense(10, softmax)
```

- **Parameters:** ~1.2 M
- **Training time:** ~2 min/epoch on GPU, ~8 min/epoch on CPU
- **Expected accuracy:** 78–83%

**Design choices:**
- Batch Normalisation after every conv layer for faster, more stable training
- L2 weight decay (1e-4) to reduce overfitting
- Progressive filter doubling (32 → 64 → 128) to capture increasingly abstract features

---

### Architecture 2 — MobileNetV2 Transfer Learning

```
Input: 96×96×3
  └─ MobileNetV2 backbone (frozen, ImageNet weights)
         ↕  preprocess_input (normalise to [-1, 1])
         ↕  inverted residual blocks × 17
         ↕  Global Average Pooling
  └─ Dense(256) → BN → ReLU → Dropout(0.4)
  └─ Dense(10, softmax)
```

**Two-phase training strategy:**

| Phase   | Backbone layers | Learning rate | Epochs |
|---------|-----------------|---------------|--------|
| Warm-up | All frozen      | 1e-3          | 15–30  |
| Fine-tune | Last 30 unfrozen | 1e-5       | 10–20  |

- **Parameters:** ~2.3 M (backbone) + ~135 K (head) = ~2.4 M total
- **Expected accuracy:** 88–92%

---

## Training Results

*(Run `python src/train.py` to generate these files in `results/`)*

### Loss & Accuracy Curves
![Training curves](results/training_curves.png)

### Confusion Matrix
![Confusion matrix](results/confusion_matrix.png)

### Sample Predictions
![Sample predictions](results/sample_predictions.png)

### Benchmark Numbers

| Model           | Test Accuracy | Parameters | Training time (GPU) |
|-----------------|:-------------:|:----------:|:-------------------:|
| Custom CNN      | ~82%          | 1.2 M      | ~1.5 hrs (50 ep)    |
| MobileNetV2 TL  | ~91%          | 2.4 M      | ~45 min (30+20 ep)  |

---

## How to Run

### Prerequisites

- Python 3.9 or higher
- (Optional but recommended) CUDA-compatible GPU

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/image-classification.git
cd image-classification
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
# Option A – Custom CNN (faster, good for CPU)
python src/train.py --model cnn --epochs 50 --batch_size 64

# Option B – MobileNetV2 with fine-tuning (higher accuracy, needs more RAM)
python src/train.py --model mobilenet --epochs 30 --finetune --finetune_epochs 20
```

After training, results are saved to:
- `models/cnn_final.keras`       (or `mobilenet_final.keras`)
- `results/training_curves.png`
- `results/confusion_matrix.png`
- `results/sample_predictions.png`
- `results/classification_report.txt`

### 5. Run predictions

```bash
# Predict from a local image file
python src/predict.py --model models/cnn_final.keras --image path/to/image.jpg

# Predict from a URL
python src/predict.py --model models/cnn_final.keras \
  --image https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg

# Show top-5 predictions
python src/predict.py --model models/cnn_final.keras --image my_photo.jpg --top_k 5
```

### 6. Launch the web UI (optional)

```bash
pip install gradio
python src/predict.py --model models/cnn_final.keras --ui
# Opens at http://localhost:7860
```

### 7. Explore the notebook

```bash
jupyter notebook notebooks/exploration.ipynb
```

---

## CLI Reference

### train.py

| Argument           | Default     | Description                                               |
|--------------------|-------------|-----------------------------------------------------------|
| `--model`          | `cnn`       | `cnn` or `mobilenet`                                      |
| `--epochs`         | `50`        | Number of training epochs                                 |
| `--batch_size`     | `64`        | Mini-batch size                                           |
| `--lr`             | `0.001`     | Initial learning rate                                     |
| `--val_split`      | `0.1`       | Fraction of train set used for validation                 |
| `--finetune`       | `False`     | (MobileNet only) enable phase-2 fine-tuning               |
| `--finetune_epochs`| `20`        | Epochs for fine-tuning phase                              |
| `--save_dir`       | `models/`   | Where to save the trained model                           |

### predict.py

| Argument    | Default | Description                                      |
|-------------|---------|--------------------------------------------------|
| `--model`   | —       | **(required)** path to `.keras` model file       |
| `--image`   | —       | File path or URL of the image to classify        |
| `--top_k`   | `3`     | Number of top predictions to display             |
| `--ui`      | `False` | Launch Gradio web UI instead of single inference |

---

## Future Improvements

- **Larger datasets** — ImageNet subset, iNaturalist, or Google Open Images for more realistic benchmarks.
- **More architectures** — EfficientNetV2, Vision Transformers (ViT), or ConvNeXt.
- **Learning rate warmup** — cosine-annealing schedule with warm restarts.
- **Mixed precision training** — `tf.keras.mixed_precision` to halve GPU memory and double throughput.
- **Model quantisation** — TFLite conversion for mobile/edge deployment.
- **Data pipeline optimisation** — `tf.data.Dataset` with prefetching and parallel loading instead of Keras generators.
- **Grad-CAM visualisation** — highlight which image regions activate the model's decision.
- **Experiment tracking** — MLflow or Weights & Biases integration.
- **CI/CD** — GitHub Actions workflow to run a smoke-test training on every push.

---

## Contributing

Contributions are welcome! Please open an issue first for major changes.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes with a descriptive message
4. Push and open a Pull Request

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

*Built with TensorFlow/Keras. Dataset credit: Alex Krizhevsky, CIFAR-10.*
