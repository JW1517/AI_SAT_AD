# 🛰️ AI SAT AD · Satellite Anomaly Detection

> **Industrial-grade Machine Learning Pipeline for Anomaly Detection in Satellite Telemetry**

---

## 📖 Overview

**AI SAT AD** is a complete pipeline for time series anomaly detection on satellite telemetry data.

The system combines:

- High-quality **feature engineering** on raw telemetry streams.
- Both **traditional machine learning** and **deep learning** models.
- A **Streamlit-based app** for seamless interaction and industrial deployment.
- Fully automated **model versioning** and **scaler management**, both local and cloud-based.
- Integration with **Google Cloud Platform** (GCS).

---

## 🎯 Project Objectives

- Model and detect subtle anomalies in highly heterogeneous satellite telemetry streams.
- Provide accurate, robust, and interpretable predictions for industrial use.
- Support **real-time visualisation** and interactive detection in production environments.
- Implement a fully modular and reproducible ML pipeline.

---

## 🗂️ Project Structure

## 🗂️ Project Structure

```plaintext
AI_SAT_AD/
├── aisatad/
│   ├── __init__.py
│   ├── __pycache__/
│   │   ├── __init__.cpython-312.pyc
│   │   └── params.cpython-312.pyc
│   ├── ano_detector/
│   │   ├── __init__.py
│   │   └── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── __pycache__/
│   │   │   ├── __init__.cpython-312.pyc
│   │   │   └── fast.cpython-312.pyc
│   │   └── fast.py
│   ├── function_files/
│   │   ├── data.py                 # Data fetching & caching
│   │   ├── preprocessor.py         # Feature engineering pipeline
│   │   ├── model.py                # Model training functions
│   │   ├── plot.py                 # Visualisation utilities
│   ├── params.py                   # Global project parameters
├── training_outputs/               # Registry of saved models, scalers, metrics
├── app.py                          # Streamlit app (production frontend)
├── README.md                       # Project documentation
└── requirements.txt                # Dependencies
```

---

## 📚 Data Specification

### Raw Data Format (example: [`segments.csv`](./segments.csv)):

| timestamp           | segment | value  | anomaly | train | channel | sampling |
|---------------------|---------|--------|---------|-------|---------|----------|
| 2025-01-01 00:00:00 | 0       | -0.435 | 0       | 1     | 2       | 10       |

- **timestamp**: Measurement timestamp.
- **segment**: Unique segment identifier.
- **value**: Telemetry value.
- **anomaly**: Label (1 = anomaly, 0 = normal).
- **train**: Flag for train/test split.
- **channel**: Measurement channel identifier.
- **sampling**: Sampling frequency.

---

## ⚙️ Preprocessing Pipeline

### Feature Engineering

#### Array-level Transformations:

- Length, mean, variance, standard deviation.
- Kurtosis, skewness.
- Peak counting (raw, smoothed at multiple scales, 1st and 2nd derivatives).
- Variance of first and second derivatives.

#### Segment-level (DataFrame) Transformations:

- Segment duration.
- Squared sum of time gaps between measurements.
- Duration of constant regions (zero derivative stretches).

### Feature Dataset Generation

- The `generate_dataset()` function builds a fully structured feature DataFrame.
- Scaling is performed via `StandardScaler`, saved with versioning.

---

## 🧠 Models

### Classical Machine Learning

- **Logistic Regression**
- **Support Vector Classifier** (linear kernel)
- **AdaBoostClassifier** with `DecisionTreeClassifier` base learner
- **XGBoostClassifier**
- **Model Stacking** (meta-model: Logistic Regression over an ensemble)

## 🧠 Deep Learning Models

### 🔸 Dense Neural Network

A fully connected architecture operating on engineered static features:

```text
Input
 → Dense(128, activation='gelu')
 → BatchNorm
 → Dropout(0.3)
 → Dense(64, activation='gelu')
 → BatchNorm
 → Dropout(0.3)
 → Dense(32, activation='gelu')
 → BatchNorm
 → Dropout(0.3)
 → Dense(1, activation='sigmoid')
```

#### RNN (LSTM-based)

```text
Input
→ Masking(mask_value=-10.0)
→ Bidirectional LSTM(64, activation='tanh', return_sequences=True)
→ LSTM(32, activation='tanh')
→ Dense(64, activation='relu') → Dropout(0.3)
→ Dense(1, activation='sigmoid')
```

### Unsupervised Anomaly Detection

- **K-Means** clustering on engineered features.
- Distance-based thresholding for anomaly detection.

---

## 📊 Visualisation

- Segment-wise signal plotting via `plot_seg()` and `plot_seg_ax()`.
- Comparative **ROC curves** for all classifiers: `plot_roc_curves_model_staking()`.

---

## 🏗️ Deployment Architecture

### Versioning & Registry

- **Model saving:** `save_model()`
- **Scaler saving:** `save_scaler()`
- Registry maintained under:

```plaintext
training_outputs/
├── models/
├── scalers/
├── metrics/
└── params/
```

- Models and scalers can be uploaded to **GCS** for cloud-based production.

### Loading for Inference

- **Models:** `load_model()` (local or GCS).
- **Scalers:** `load_scaler()` (local or GCS).

---

## 🌐 Streamlit App

Located in `app.py`.

### Features:

✅ Upload CSV file.  
✅ Visualise telemetry signals by segment.  
✅ Interactive prediction using deployed API endpoint (`/predict`).  
✅ Instant display of anomaly detection results.

### Deployment:

- Streamlit frontend deployable on **Streamlit Sharing** or **GCP Cloud Run**.
- Backend API served via **GCP Cloud Run**.

---

## 🛠️ Installation

```bash
git clone https://github.com/YOUR_GITHUB/AI_SAT_AD.git
cd AI_SAT_AD
pip install -r requirements.txt
```

## 🚀 Usage Guide

🧹 Preprocessing
```bash
from aisatad.function_files.data import get_data_with_cache
df = get_data_with_cache()

from aisatad.function_files.preprocessor import preprocess
X_train, X_test, y_train, y_test, scaler = preprocess(df)
```

🏋️‍♂️ Model Training Example

```bash
from aisatad.function_files.model import logistic_regression
logistic_regression(X_train, X_test, y_train, y_test)
```

🧠 Deep Learning Example

```bash
from aisatad.function_files.model import Dense_Neural_model
Dense_Neural_model(X_train, X_test, y_train, y_test)
```

💾 Saving Artifacts

```bash
from aisatad.function_files.registry import save_model, save_scaler
save_model(model)
save_scaler(scaler)
```

### 👥 **Collaborators**
- [![Arsène Géry](https://img.shields.io/badge/GitHub-Arsène_Géry-blue?logo=github)](https://github.com/Arsene-Gery)  
- [![Jocelyn Tête](https://img.shields.io/badge/GitHub-Jocelyn_Tête-blue?logo=github)](https://github.com/JTetePro) 
- [![Juan Wang](https://img.shields.io/badge/GitHub-Juan_Wang-blue?logo=github)](https://github.com/JW1517)  
- [![Yves Sporri](https://img.shields.io/badge/GitHub-Yves_Sporri-blue?logo=github)](https://github.com/Cohy)
