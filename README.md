# Quantum-Enhanced Diabetic Retinopathy Classifier

A final-year AI/ML project for diabetic retinopathy screening using deep learning and transfer learning. The project includes model training, evaluation, explainability, deployment, and a user-friendly web interface.

## Live Demo

Streamlit App: https://quantum-enhanced-diabetic-retinopathy-classifier-aiml-project.streamlit.app

## Problem Statement

Diabetic retinopathy is a serious diabetes-related eye disease that can lead to vision loss if not detected early. This project aims to classify retinal fundus images as Normal or Abnormal using AI-assisted screening.

## Key Features

- Retinal image upload
- Normal/Abnormal classification
- Confidence score
- Probability bars
- Grad-CAM visual explanation
- Streamlit web deployment
- Google Drive model loading
- Docker support for containerized deployment

## Technology Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow
- Google Drive model storage
- Docker

## Model Summary

The deployed model uses a trained Inception-based Keras model.

The project also includes experiments with:
- InceptionV3
- ResNet-152
- Quantum-inspired model experiments

## Evaluation Metrics

The project evaluates models using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC curve
- AUC score

Best reported performance:

| Model | Accuracy | AUC |
|---|---:|---:|
| InceptionV3 Fine-tuned | 79.13% | 0.86 |
| ResNet-152 + Quantum | 77.31% | - |

## How To Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

