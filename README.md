<div align="center">

![Banner](./assets/banner.png)

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![PennyLane](https://img.shields.io/badge/PennyLane-Quantum-8B5CF6?style=for-the-badge&logo=quantum&logoColor=white)](https://pennylane.ai)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*An advanced deep learning approach combining **InceptionV3** and **ResNet-152** architectures with **Quantum Computing** layers for accurate diabetic retinopathy classification.*

[Features](#-key-features) • [Results](#-model-performance) • [Installation](#-getting-started) • [Models](#-pretrained-models)

---

</div>

## 🎯 About The Project

Diabetic Retinopathy (DR) is a leading cause of blindness worldwide. Early detection is crucial for preventing vision loss. This project leverages the power of **Quantum Transfer Learning** to classify retinal images as **Normal** or **Abnormal** with state-of-the-art accuracy.

### 🏆 Best Model Achievement
```
┌─────────────────────────────────────────────────────────┐
│                    BEST ACCURACY: 79.13%                │
│                     AUC Score: 0.86                     │
│            Model: InceptionV3 + Fine-tuning             │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Model Performance

![Performance Comparison](./assets/performance.png)

### Accuracy Comparison Across Experiments

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|:------|:--------:|:---------:|:------:|:--------:|:---:|
| **InceptionV3 (Best)** | **79.13%** | **0.80** | **0.79** | **0.79** | **0.86** |
| InceptionV3 v2 | 77.50% | 0.78 | 0.77 | 0.77 | 0.86 |
| InceptionV3 v3 | 77.00% | 0.78 | 0.77 | 0.76 | - |
| InceptionV3 v4 | 76.00% | 0.78 | 0.76 | 0.75 | - |
| InceptionV3 v5 | 74.00% | 0.79 | 0.74 | 0.73 | - |
| ResNet-152 + Quantum | 77.31% | - | - | - | - |

### 📈 Best Model Classification Report

```
              precision    recall  f1-score   support

    Abnormal     0.7154    0.8161    0.7624       348
      Normal     0.8581    0.7740    0.8139       500

    accuracy                         0.7913       848
   macro avg     0.7867    0.7950    0.7881       848
weighted avg     0.7995    0.7913    0.7928       848
```

### 🔢 Confusion Matrix (Best Model)

```
                 Predicted
              ┌─────────┬─────────┐
              │Abnormal │ Normal  │
┌─────────────┼─────────┼─────────┤
│ Abnormal    │   284   │    64   │  (81.6% recall)
├─────────────┼─────────┼─────────┤
│ Normal      │   113   │   387   │  (77.4% recall)
└─────────────┴─────────┴─────────┘
```

---

## 🔬 Architecture

![Architecture Diagram](./assets/architecture.png)

<table>
<tr>
<td width="50%">

### 🧠 Model Pipeline
- **Transfer Learning** with InceptionV3 & ResNet-152
- **Quantum Layers** via PennyLane (6 qubits, 8 layers)
- **Two-stage Training** (frozen backbone + fine-tuning)

</td>
<td width="50%">

### 🛡️ Robust Training
- **MixUp Augmentation** (α = 0.12)
- **Label Smoothing** (0.015)
- **Early Stopping** with patience
- **Learning Rate Scheduling**

</td>
</tr>
</table>

---

## ⚛️ Quantum Computing Layer

![Quantum Circuit](./assets/quantum.png)

The quantum layer is configured with the following parameters:

| Parameter | Value | Description |
|:----------|:-----:|:------------|
| `n_qubits` | 6 | Number of quantum bits |
| `q_depth` | 8 | Quantum circuit depth |
| `q_delta` | 0.005 | Quantum gradient step |

**How it Works:**
1. **Encoding**: Classical CNN features → Quantum states via rotation gates
2. **Entanglement**: CNOT gates create quantum correlations
3. **Measurement**: Quantum states → Classical features
4. **Training**: Backpropagation via PennyLane's autodiff

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🔬 Advanced Architecture
- **Transfer Learning** with InceptionV3 & ResNet-152
- **Quantum Layers** via PennyLane (6 qubits, 8 layers)
- **Two-stage Training** (frozen backbone + fine-tuning)

</td>
<td width="50%">

### 🛡️ Robust Training
- **MixUp Augmentation** (α = 0.12)
- **Label Smoothing** (0.015)
- **Early Stopping** with patience
- **Learning Rate Scheduling**

</td>
</tr>
<tr>
<td width="50%">

### 📊 Comprehensive Evaluation
- Confusion Matrix visualization
- ROC Curves with AUC scores
- Classification Reports (JSON/CSV/TXT)
- Training metrics tracking

</td>
<td width="50%">

### ⚡ Flexibility
- Works on **CPU** and **GPU**
- Configurable hyperparameters
- Optional quantum layer integration
- Automatic data augmentation

</td>
</tr>
</table>

---

## 🛠️ Tech Stack

<div align="center">

| Category | Technologies |
|:--------:|:------------|
| **Deep Learning** | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white) |
| **Quantum Computing** | ![PennyLane](https://img.shields.io/badge/PennyLane-8B5CF6?style=flat&logoColor=white) |
| **Data Science** | ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white) |
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat) |

</div>

---

## 📂 Project Structure

```
Diabetic-Retinopathy-Using-Quantum-Transfer-Learning/
│
├── 📁 assets/                 # 🖼️ Images for README
│   ├── banner.png
│   ├── architecture.png
│   ├── performance.png
│   └── quantum.png
│
├── 📁 inception_79%/          # 🏆 Best model (79.13% accuracy)
│   ├── 1.py                   # Training script
│   ├── classification_report.txt
│   ├── confusion_matrix.json
│   └── confusion_matrix.npy
│
├── 📁 inception_77.5%/        # InceptionV3 variant
│   ├── training_metrics.json
│   └── roc_curve_data.json
│
├── 📁 resnet_152_/            # ResNet-152 + Quantum
│   ├── results_summary.json
│   └── training_history.json
│
├── 📁 inception_*%/           # Other experiment variants
│
├── 📄 4.py                    # Utility scripts
├── 📄 5.py
└── 📄 README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.8+ required
python --version
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Diabetic-Retinopathy-Using-Quantum-Transfer-Learning.git
cd Diabetic-Retinopathy-Using-Quantum-Transfer-Learning

# Install dependencies
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn

# Optional: For quantum features
pip install pennylane pennylane-qiskit
```

### Dataset Preparation

```
im1/
├── train/
│   ├── Normal/       # Normal retina images
│   └── Abnormal/     # Diabetic retinopathy images
│
└── val/
    ├── Normal/
    └── Abnormal/
```

### Training

```bash
# Train the best model (InceptionV3)
cd inception_79%
python 1.py

# Or train with quantum layers (ResNet-152)
cd resnet_152_
python train.py
```

---

## 🔗 Pretrained Models

<div align="center">

[![Download Models](https://img.shields.io/badge/Download%20Models-Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1BYcqTBkt3Zvd_mHEggkrZZmQNrRDGAm9?usp=sharing)

</div>

The models save the following artifacts for evaluation:
- ✅ Trained model weights (`.keras` / `.pth`)
- ✅ Confusion matrix (JSON & NPY)
- ✅ Classification reports
- ✅ ROC curve data
- ✅ Training metrics (JSON & CSV)

---

## 📈 Training Hyperparameters

```python
# Best performing configuration
BATCH_SIZE = 8
NUM_EPOCHS = 51
MIXUP_ALPHA = 0.12
LABEL_SMOOTHING = 0.015
DROPOUT_RATE = 0.45
WEIGHT_DECAY = 1e-4
DENSE_AFTER_Q = 512
EARLYSTOP_PATIENCE = 15
FINE_TUNE_EPOCHS = 15
```

---

## 👤 Creator

- 💼 **Created by**: Kshama Mishra

---

<div align="center">


Created by Kshama Mishra

</div>
