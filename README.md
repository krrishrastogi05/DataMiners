# 🏥 Fall Detection — Time Series Classification Pipeline

> **Academic Submission | Data Mining / Machine Learning Course**  
> **Team:** DataMiners  
> **Dataset:** IMU Wearable Sensor Data @ 100 Hz

---

## 📋 Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset](#2-dataset)
3. [Pipeline Overview](#3-pipeline-overview)
4. [Preprocessing & Feature Engineering](#4-preprocessing--feature-engineering)
5. [Class Imbalance Strategy](#5-class-imbalance-strategy)
6. [Models & Hyperparameter Tuning](#6-models--hyperparameter-tuning)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Results Summary](#8-results-summary)
9. [Project Structure](#9-project-structure)
10. [How to Run](#10-how-to-run)
11. [Dependencies](#11-dependencies)

---

## 1. Problem Statement

This project develops a machine learning pipeline for **binary fall detection** using data from body-mounted Inertial Measurement Unit (IMU) sensors. The task is:

- **Input:** Raw 9-channel IMU signals — `AccX`, `AccY`, `AccZ`, `GyrX`, `GyrY`, `GyrZ`, `EulerX`, `EulerY`, `EulerZ`
- **Output:** Binary classification — `Fall (1)` vs `No-Fall (0)`

A central challenge is **extreme class imbalance**: routine activity (No-Fall) overwhelmingly outnumbers fall events in real-world sensor recordings (~7:1 ratio), causing naive models to collapse into majority-class prediction and entirely miss falls.

---

## 2. Dataset

| Split    | Subjects | Raw Rows   | After Windowing |
|----------|----------|------------|-----------------|
| Training | 17       | ~1,470,000 | 49,290 windows  |
| Test     | 6        | ~287,813   | 11,505 windows  |

**Sensor Channels (9 total):**

| Type          | Channels               |
|---------------|------------------------|
| Accelerometer | AccX, AccY, AccZ       |
| Gyroscope     | GyrX, GyrY, GyrZ       |
| Euler Angles  | EulerX, EulerY, EulerZ |

**Class Distribution (Training):**

| Class     | Count  | Proportion |
|-----------|--------|------------|
| No-Fall (0) | 43,160 | 87.6%    |
| Fall (1)    |  6,130 | 12.4%    |
| **Ratio**   | **7.04 : 1** |        |

---

## 3. Pipeline Overview

```
Raw CSV Files (per subject)
        │
        ▼
  Data Loading & Validation
        │
        ▼
  Sliding Window Segmentation   ← Window: 50 steps | Step: 25 | Overlap: 50%
        │
        ▼
  Statistical Feature Extraction ← 15 stats × 9 channels = 135 features
        │
        ▼
  ┌─────────────────────────────────────────────────────┐
  │  3 Dataset Variants                                 │
  │  ① Imbalanced Baseline (raw 7:1)                    │
  │  ② Class-Weighted (algorithmic balance)             │
  │  ③ Data-Balanced (random undersampling, 1:1)        │
  └─────────────────────────────────────────────────────┘
        │
        ▼
  Nested Cross-Validation + Hyperparameter Search
  (Inner 3-Fold RandomizedSearchCV → Outer 3-Fold CV)
        │
        ▼
  ┌────────────────────────────────────────────┐
  │  4 Classical Classifiers                   │
  │  • Random Forest      • Logistic Regression│
  │  • SVM                • XGBoost            │
  └────────────────────────────────────────────┘
        │
        ▼
  Plan B: LSTM Deep Learning Baseline
  (Raw windows (N, 50, 9) → 2-layer stacked LSTM)
        │
        ▼
  Evaluation & Comparison Report
```

---

## 4. Preprocessing & Feature Engineering

### 4.1 Sliding Window Segmentation

Raw 100 Hz sensor streams lack temporal context when processed point-by-point. Windows of **50 timesteps (0.5 seconds)** are extracted with a step of **25 timesteps (50% overlap)**.

**Label Thresholding:** A window is labelled Fall (`1`) if more than **40%** of its timesteps are fall events; otherwise No-Fall (`0`).

### 4.2 Vectorised Statistical Feature Extraction

For each window, **15 hand-crafted statistics** are computed per sensor channel — fully implemented from scratch using NumPy (no scikit-learn statistics calls):

| Category               | Statistics                                      |
|------------------------|-------------------------------------------------|
| Central Tendency       | Mean, Median (p50)                              |
| Dispersion             | Std, Variance, IQR, p25, p75, MAD               |
| Range & Energy         | Min, Max, Range, RMS, Energy (sum of squares)   |
| Signal Shape           | Skewness, Excess Kurtosis, Zero-Crossing Rate   |

**15 statistics × 9 channels = 135 features per window.**

This dimensionality reduction converts a raw `(50, 9)` window into a compact `(135,)` descriptor, enabling classical ML classifiers to operate on meaningful, physics-aware features.

---

## 5. Class Imbalance Strategy

Three dataset variants are used to explicitly study the effect of balancing:

### Variant 1 — Imbalanced Baseline
Raw 7:1 data, no correction. Demonstrates the collapse in minority-class recall when imbalance is ignored.

### Variant 2 — Class-Weighted (Algorithmic Balance)
Data physically unchanged. Each classifier receives a `class_weight="balanced"` parameter (or `scale_pos_weight=7.04` for XGBoost), mathematically penalising misclassification of Fall samples ~7× more during loss computation.

### Variant 3 — Data-Balanced (Random Undersampling)
Dataset physically forced to 1:1 by randomly removing majority-class samples.

> **Architectural constraint enforced:** 100% of all minority (Fall) samples are retained. Only majority-class samples are drawn down. This preserves the full ground-truth variation of actual fall events.

> **SVM-specific constraint:** SVM operates at O(n²) complexity. Training is capped at 15,000 samples (stratified), with all minority samples retained, to prevent memory stalls.

---

## 6. Models & Hyperparameter Tuning

All classical models run through a **Nested Cross-Validation** harness:

- **Inner loop:** `RandomizedSearchCV` (3-Fold Stratified) optimising **Balanced Accuracy**
- **Outer loop:** 3-Fold Stratified K-Fold for generalisation validation
- **Final evaluation:** Refit best model on full training set → test on held-out test set

| Model               | Type      | Search Type | Iterations | Class Balance Support |
|---------------------|-----------|-------------|------------|-----------------------|
| Random Forest       | Ensemble  | Random      | 15         | `class_weight`        |
| SVM                 | Shallow   | Random      | 10         | `class_weight`        |
| Logistic Regression | Shallow   | Random      | 10         | `class_weight`        |
| XGBoost             | Ensemble  | Random      | 20         | `scale_pos_weight`    |

**Plan B — LSTM Deep Learning Baseline:**  
A 2-layer stacked LSTM processes raw `(N, 50, 9)` windows directly (no hand-crafted features). Imbalance handled via `class_weight`. Provides a classical-vs-DL comparison. Supports both TensorFlow/Keras and PyTorch backends.

| Parameter   | Value |
|-------------|-------|
| LSTM Units  | 64 → 32 (stacked) |
| Dropout     | 0.3 per layer |
| Epochs      | 10 |
| Batch Size  | 256 |
| Optimizer   | Adam (lr=1e-3) |

---

## 7. Evaluation Metrics

Standard accuracy is **deliberately excluded** as the primary metric. A model that predicts No-Fall for every window would achieve ~87% accuracy on this dataset while detecting zero falls.

Primary metrics:

| Metric                        | Formula                             | Why It Matters                                      |
|-------------------------------|-------------------------------------|-----------------------------------------------------|
| **Balanced Accuracy**         | (TPR + TNR) / 2                     | Corrects for imbalance; primary ranking metric      |
| **Recall / Sensitivity (TPR)**| TP / (TP + FN)                      | Critical — every missed fall is a safety failure    |
| **Specificity (TNR)**         | TN / (TN + FP)                      | Avoids excessive false alarms                       |
| **Precision (PPV)**           | TP / (TP + FP)                      | Quality of fall alerts                              |
| **F1 Macro**                  | Harmonic mean of F1 per class       | Combined performance across both classes            |
| **F1 (Class 1 — Fall)**       | 2·P·R / (P + R)                     | Fall-specific detection quality                     |

Confusion matrix values (TP, TN, FP, FN) are computed from scratch without scikit-learn.

---

## 8. Results Summary

Best observed performance on the held-out test set (11,505 windows):

| Model               | Dataset Variant         | Balanced Acc | F1 Macro | Recall (Fall) |
|---------------------|------------------------|:------------:|:--------:|:-------------:|
| Logistic Regression | Class-Weighted          | 0.9344       | ~0.88    | 0.9305        |
| Random Forest       | Data-Balanced           | —            | —        | —             |
| XGBoost             | Class-Weighted          | —            | —        | —             |
| SVM                 | Data-Balanced           | —            | —        | —             |
| LSTM (Keras)        | Full + class_weight     | —            | —        | —             |

> Run `python main2.py` to reproduce all result tables, confusion matrices, and feature importance plots.

---

## 9. Project Structure

```
DataMiners/
│
├── main2.py                  # ← MAIN SUBMISSION FILE
│                               Full pipeline: load → window → extract → train → evaluate
│
├── evaluate_new_data.py      # Inference script for new/hidden datasets
│                               Loads saved .pkl models and reports all metrics
│
├── data/
│   ├── Sample_Training/      # Training data (per-subject subdirectories of CSVs)
│   │   └── Subject_XX/
│   │       └── *.csv
│   │
│   ├── Sample_Test/          # Held-out test data (structure mirrors training)
│   │   └── Subject_XX/
│   │       └── *.csv
│   │
│   └── Balanced/             # Generated outputs (created on first run)
│       ├── full_training_set.csv
│       ├── undersampled_training_set.csv
│       ├── full_test_set.csv
│       └── saved_models/     # .pkl and .keras model files
│           ├── RandomForest_*.pkl
│           ├── SVM_*.pkl
│           ├── LogisticRegression_*.pkl
│           ├── XGBoost_*.pkl
│           └── LSTM_Keras_class_weighted.keras
│
└── README.md                 # This file
```

---

## 10. How to Run

### Step 1 — Train all models

```bash
python main2.py
```

This will:
- Load training & test data from `data/Sample_Training/` and `data/Sample_Test/`
- Apply sliding window segmentation and extract 135 features per window
- Train 4 classifiers × 3 dataset variants (12 models total) with nested CV
- Train an LSTM baseline (if TensorFlow or PyTorch is available)
- Save all trained models to `data/Balanced/saved_models/`
- Generate confusion matrices, metric comparison charts, and feature importance plots
- Log all output to a timestamped `run_output_YYYYMMDD_HHMMSS.txt` file

### Step 2 — Evaluate on new / hidden data

```bash
python evaluate_new_data.py <path_to_dataset_directory>
```

**Example:**
```bash
python evaluate_new_data.py data/Sample_Test
```

The script expects the directory to contain per-subject subdirectories of CSV files (same format as training data). It will:
1. Load and preprocess the new dataset
2. Extract features using the same 135-feature pipeline
3. Load all saved `.pkl` (and optionally `.keras`) models
4. Print a full evaluation report per model including:
   - Confusion matrix (TP / TN / FP / FN)
   - Balanced Accuracy
   - Precision, Recall (Sensitivity), Specificity
   - F1 Score per class + F1 Macro

### Google Colab

The pipeline auto-detects the Colab environment. Clone the repository to `/content/DataMiners` and run normally:

```bash
!git clone https://github.com/krrishrastogi05/DataMiners /content/DataMiners
%cd /content/DataMiners
!python main2.py
```

---

## 11. Dependencies

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn joblib
```

**Optional (for LSTM Plan B):**
```bash
pip install tensorflow        # Keras backend (preferred)
# OR
pip install torch             # PyTorch backend (fallback)
```

| Package      | Version Tested | Purpose                              |
|--------------|:--------------:|--------------------------------------|
| numpy        | ≥ 1.24         | Vectorised feature extraction        |
| pandas       | ≥ 2.0          | Data loading and tabular output      |
| scikit-learn | ≥ 1.3          | ML models, CV, pipelines             |
| xgboost      | ≥ 2.0          | XGBoost classifier                   |
| matplotlib   | ≥ 3.7          | Visualisations (saved to PNG)        |
| seaborn      | ≥ 0.12         | Heatmap styling                      |
| joblib       | ≥ 1.3          | Model serialisation (.pkl)           |
| tensorflow   | ≥ 2.13         | LSTM Keras backend *(optional)*      |
| torch        | ≥ 2.0          | LSTM PyTorch backend *(optional)*    |

---

*All statistical formulas (skewness, kurtosis, zero-crossing rate, confusion matrix values, F1, balanced accuracy) are implemented from scratch using NumPy — no scikit-learn metric functions are used in the core evaluation.*
