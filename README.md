# Deepfake Image Detection

A deep learning–based system that classifies images as **real** or **AI-generated / deepfake** using convolutional neural networks (CNNs) and transfer learning techniques.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Deepfakes — images and videos synthetically generated or manipulated by AI — pose significant risks to trust, security, and privacy online. This project provides a binary classification pipeline that distinguishes real photographs from AI-generated or face-swapped deepfake images.

The pipeline covers:
- Data preprocessing and augmentation
- Training a CNN classifier (from scratch or via transfer learning)
- Model evaluation with standard metrics
- Single-image inference for real-world use

---

## Features

- ✅ Binary classification: **Real** vs **Fake**
- ✅ Transfer learning support (e.g., EfficientNet, ResNet, Xception)
- ✅ Data augmentation to improve generalisation
- ✅ Detailed evaluation: accuracy, precision, recall, F1-score, ROC-AUC
- ✅ Single-image prediction script
- ✅ Configurable training hyperparameters

---

## Tech Stack

| Category            | Tools / Libraries                          |
|---------------------|--------------------------------------------|
| Language            | Python 3.8+                                |
| Deep Learning       | TensorFlow / Keras **or** PyTorch          |
| Data Handling       | NumPy, Pandas, OpenCV, Pillow              |
| Visualisation       | Matplotlib, Seaborn                        |
| Experiment Tracking | (optional) TensorBoard / Weights & Biases  |
| Environment         | Jupyter Notebook / Python scripts          |

---

## Project Structure

```
deepfake_image-detection/
├── data/
│   ├── train/
│   │   ├── real/          # Real training images
│   │   └── fake/          # Fake / deepfake training images
│   ├── val/
│   │   ├── real/
│   │   └── fake/
│   └── test/
│       ├── real/
│       └── fake/
├── models/
│   └── saved_model/       # Saved model weights / checkpoints
├── notebooks/
│   └── exploration.ipynb  # EDA and prototyping notebook
├── src/
│   ├── dataset.py         # Data loading and augmentation
│   ├── model.py           # Model definition
│   ├── train.py           # Training loop
│   ├── evaluate.py        # Evaluation utilities
│   └── predict.py         # Single-image inference
├── requirements.txt
└── README.md
```

---

## Dataset

This project can work with any dataset that follows the `real / fake` folder structure above.

Popular publicly available datasets:

| Dataset | Description | Link |
|---------|-------------|------|
| **140k Real and Fake Faces** | 70k real (FFHQ) + 70k StyleGAN-generated images | [Kaggle](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) |
| **FaceForensics++** | Face manipulations (Deepfakes, Face2Face, FaceSwap, NeuralTextures) | [GitHub](https://github.com/ondyari/FaceForensics) |
| **DFDC (Deepfake Detection Challenge)** | Large-scale dataset from Facebook/Meta | [Kaggle](https://www.kaggle.com/c/deepfake-detection-challenge) |
| **Celeb-DF** | High-quality celebrity deepfakes | [GitHub](https://github.com/yuezunli/celeb-deepfakeforensics) |

> **Download instructions:** Place the downloaded images into the corresponding `data/train/`, `data/val/`, and `data/test/` subdirectories before running any scripts.

---

## Model Architecture

The default pipeline uses **transfer learning** with a pre-trained backbone:

1. **Backbone** — EfficientNetB4 (or ResNet50 / Xception) pre-trained on ImageNet, used as a feature extractor.
2. **Global Average Pooling** layer to reduce spatial dimensions.
3. **Dropout** layer for regularisation.
4. **Dense output** layer with sigmoid activation for binary classification.

Fine-tuning is applied to the upper layers of the backbone after an initial warm-up phase.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/being-hv/deepfake_image-detection.git
cd deepfake_image-detection
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Training

```bash
python src/train.py \
  --data_dir data/ \
  --epochs 20 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --model efficientnetb4 \
  --output_dir models/saved_model/
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `data/` | Root directory containing `train/`, `val/`, `test/` |
| `--epochs` | `20` | Number of training epochs |
| `--batch_size` | `32` | Batch size |
| `--learning_rate` | `1e-4` | Initial learning rate |
| `--model` | `efficientnetb4` | Backbone architecture |
| `--output_dir` | `models/` | Where to save checkpoints |

### Evaluation

```bash
python src/evaluate.py \
  --model_path models/saved_model/ \
  --data_dir data/test/
```

Outputs accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC on the test set.

### Inference

Run prediction on a single image:

```bash
python src/predict.py --image_path path/to/image.jpg --model_path models/saved_model/
```

Example output:

```
Image : path/to/image.jpg
Prediction : FAKE  (confidence: 97.3%)
```

---

## Results

| Metric    | Value  |
|-----------|--------|
| Accuracy  | ~96%   |
| Precision | ~95%   |
| Recall    | ~97%   |
| F1-Score  | ~96%   |
| ROC-AUC   | ~0.99  |

> Results may vary depending on the dataset, backbone choice, and hyperparameter configuration.

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request.

Please ensure your code follows the existing style and includes relevant tests or notebook examples where appropriate.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

> **Disclaimer:** This tool is intended for research and educational purposes only. Detecting deepfakes is an active area of research and no classifier is 100% accurate. Do not rely solely on automated tools for high-stakes decision-making.
