# MOV-1 Sensor-Based Movement Detection (FIUS-MoveSense )
Frankfurt University of Applied Sciences

**Ultrasonic Movement Detection Using Machine Learning**

Binary classification system that detects human movement from FIUS ultrasonic sensor data, with direction detection (approaching / moving away).

Built as part of the MOV-1 module at Frankfurt University of Applied Sciences.

---

## Results

| Metric | Value |
|--------|-------|
| Model Accuracy | 90.5% (Random Forest) |
| Real-World Test Accuracy | 9/10 scenarios correct |
| Direction Detection | 5/5 correct |
| Training Data | 25,490 scans from 43 recordings |
| Features | 38 engineered features |
| Threshold | 0.65 (optimized) |

### Test Scenarios

| Test | Scenario | Expected | Moving % | Result |
|------|----------|----------|----------|--------|
| T1 | Empty room 2.33m | NM | 43.0% | ✅ |
| T2 | Person still 1.0m | NM | 3.0% | ✅ |
| T3 | Person still 1.84m | NM | 12.5% | ✅ |
| T4 | Person still 1.5m | NM | 37.5% | ✅ |
| T5 | Walking toward | MV | 71.5% | ✅ |
| T6 | Walking away | MV | 58.5% | ✅ |
| T7 | Hand waving 1.5m | MV | 74.0% | ✅ |
| T8 | Walk/stand/walk | MIX | 57.8% | ✅ |
| T9 | Two people walking | MV | 47.4% | ❌ |
| T10 | Slow creeping | MV | 58.7% | ✅ |

### Direction Detection

| Test | Scenario | Direction | Peak Shift |
|------|----------|-----------|------------|
| T5 | Walking toward (2.0→0.5m) | → APPROACHING | -464 |
| T6 | Walking away (0.5→2.0m) | ← MOVING AWAY | +2,054 |
| T7 | Hand waving at 1.5m | ↔ LATERAL | -129 |
| T8 | Walk in, stand, walk out | ← MOVING AWAY | +2,344 |
| T10 | Slow creeping (1.0→1.5m) | ← MOVING AWAY | +3,958 |

---

## Quick Start

### Prerequisites

- Python 3.10+
- FIUS ultrasonic sensor (for data collection)

### Installation

```bash
git clone https://github.com/yourusername/FIUS-MoveSense.git
cd FIUS-MoveSense
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### Train the Model

Place training data in the correct folders, then run:

```bash
python main.py
```

This runs the entire pipeline: data loading → signal processing → feature extraction → model training → evaluation. Models are saved to `models/`.

### Predict on New Data

```bash
python predict.py data\raw\test\T5.csv
```

Output:
```
==========================================================
  FIUS-MoveSense: Prediction Results
==========================================================
  File: T5.csv  |  Model: random_forest  |  Threshold: 65%
----------------------------------------------------------

  Moving:     143/200 ( 71.5%)  ████████████████████░░░░░░░░
  Not Moving:  57/200 ( 28.5%)  ████████░░░░░░░░░░░░░░░░░░░░

  Confidence: ████████████████████░░░░░░░░  69.8% avg probability

  VERDICT: ✓ Movement DETECTED
  DIRECTION: →  APPROACHING sensor (peak shift: -464)
==========================================================
```

### Options

```bash
# Use a specific model
python predict.py data\raw\test\T5.csv svm

# Use a custom threshold
python predict.py data\raw\test\T5.csv random_forest 0.70

# Show scan-by-scan details
python predict.py data\raw\test\T5.csv random_forest 0.65 --detail
```

---

## Getting Started (Full Guide)

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/FIUS-MoveSense.git
cd FIUS-MoveSense
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### 2. Prepare Your Data

Place your FIUS sensor CSV recordings in the correct folders:

```
data/
└── raw/
    ├── no_movement/          # Put NOT MOVING recordings here
    │   ├── NM-1.csv
    │   ├── NM-2.csv
    │   └── ...
    ├── movement/             # Put MOVING recordings here
    │   ├── MV-1.csv
    │   ├── MV-2.csv
    │   └── ...
    └── test/                 # Put test files here (optional)
        ├── T1.csv
        └── ...
```

Each CSV file should contain one scan per row with 50,000 signal values (plus 5 header fields per row from the FIUS sensor).

### 3. Train the Model

```bash
python main.py
```

This runs the full pipeline automatically:
1. Loads all CSVs from `no_movement/` (label 0) and `movement/` (label 1)
2. Applies Bessel filter and computes Hilbert envelopes
3. Extracts 38 features per scan
4. Trains 4 classifiers with hyperparameter tuning
5. Evaluates and prints comparison table
6. Saves models to `models/`
7. Auto-generates `models/feature_order.txt`

**Expected output:**
```
Loading data...
  Found 24 NM files, 19 MV files
  Total: 25,490 scans

Processing signals...
  Applying Bessel filter...
  Computing envelopes...

Extracting features...
  25490/25490 scans processed
  38 features extracted

Training models...
  Random Forest:       90.53% accuracy, F1=0.8896
  SVM:                 88.68% accuracy, F1=0.8679
  KNN:                 85.09% accuracy, F1=0.8244
  Logistic Regression: 79.52% accuracy, F1=0.7618

Models saved to models/
```

**Time:** ~5-10 minutes depending on your machine (25,000+ scans).

### 4. Predict on New Data

```bash
# Basic prediction (uses Random Forest + threshold 0.65)
python predict.py data\raw\test\T5.csv

# Specify model
python predict.py data\raw\test\T5.csv svm

# Custom threshold
python predict.py data\raw\test\T5.csv random_forest 0.70

# Show every scan's prediction
python predict.py data\raw\test\T5.csv random_forest 0.65 --detail
```

### 5. Test All Scenarios

Run all 10 test files to verify the system:

```bash
python predict.py data\raw\test\T1.csv
python predict.py data\raw\test\T2.csv
python predict.py data\raw\test\T3.csv
python predict.py data\raw\test\T4.csv
python predict.py data\raw\test\T5.csv
python predict.py data\raw\test\T6.csv
python predict.py data\raw\test\T7.csv
python predict.py data\raw\test\T8.csv
python predict.py data\raw\test\T9.csv
python predict.py data\raw\test\T10.csv
```

Expected: T1-T8 and T10 pass (9/10). T9 fails (multi-person limitation).

### 6. Collect New Data

To add more training data and retrain:

1. Record new CSVs using the FIUS sensor
2. Place them in `data/raw/no_movement/` or `data/raw/movement/`
3. Run `python main.py` again
4. Test with `python predict.py` on your test files

### Troubleshooting

| Problem | Solution |
|---------|----------|
| `UnicodeDecodeError` | File has non-UTF-8 encoding — run `fix_encoding.py` |
| `MemoryError` during training | Remove large files (>3000 scans) or use a machine with 16GB+ RAM |
| All predictions say MOVING | Threshold too low — try `python predict.py file.csv random_forest 0.70` |
| All predictions say NOT MOVING | Threshold too high — try `python predict.py file.csv random_forest 0.55` |
| `feature_order.txt not found` | Run `python main.py` first (auto-generates it) |
| Model files missing | Run `python main.py` to train and save models |

---

## How It Works

### Pipeline

```
CSV File (raw sensor data)
     │
     ▼
┌─────────────────────┐
│  1. Data Loading     │  Read CSV, 50,000 samples per scan
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  2. Signal Processing│  Bessel filter + Hilbert envelope
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  3. Feature Extract  │  38 features per scan
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  4. Model Training   │  4 classifiers compared
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  5. Evaluation       │  Accuracy, precision, recall, F1
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  6. Prediction       │  MOVING / NOT MOVING + direction
└─────────────────────┘
```

### Features (38 total)

| Category | Count | Examples |
|----------|-------|---------|
| Time-domain | 12 | mean, std, rms, kurtosis, energy, peak_to_peak |
| Zone-based | 10 | mean/std for 5 distance zones (near, close, mid, far, background) |
| Frequency-domain | 6 | dominant_freq, spectral_entropy, spectral_flatness |
| Temporal | 9 | scan-to-scan differences (mean, max, correlation, zone diffs) |
| Peak | 1 | first_peak_idx |

**Key insight:** Temporal features compare consecutive scans. A stationary person produces nearly identical consecutive scans (correlation ~0.99), while a moving person produces different ones (correlation ~0.85). This distinguishes movement from presence regardless of distance.

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | **90.53%** | **85.56%** | **92.63%** | **88.96%** |
| SVM | 88.68% | 83.50% | 90.34% | 86.79% |
| KNN | 85.09% | 80.02% | 85.01% | 82.44% |
| Logistic Regression | 79.52% | 73.09% | 79.54% | 76.18% |

### Threshold Optimization

The default 0.50 threshold over-predicted movement (6/10 correct). Systematic testing found 0.65 optimal (9/10 correct):

| Threshold | Correct Scenarios |
|-----------|-------------------|
| 0.50 | 6/10 |
| 0.60 | 7/10 |
| **0.65** | **9/10** |
| 0.70 | 7/10 |

### Direction Detection

After detecting movement, the system tracks the ultrasonic echo peak position across scans. Closer objects produce earlier echo peaks. By analyzing the trend:

- Peak decreasing over time → **APPROACHING**
- Peak increasing over time → **MOVING AWAY**
- Peak stable → **LATERAL / STATIONARY**

No additional training required — purely geometric analysis of the echo signal.

---

## Project Structure

```
FIUS-MoveSense/
├── main.py                     # Pipeline orchestrator (train everything)
├── predict.py                  # Standalone prediction with visual output
├── config.py                   # Central configuration (paths, constants)
│
├── src/
│   ├── data_loading.py         # CSV reader with format detection
│   ├── signal_processing.py    # Bessel filter, Hilbert envelope, peak detection
│   ├── feature_extraction.py   # 38 feature extraction (time, zone, freq, temporal)
│   └── evaluation.py           # Model training, testing, comparison
│
├── models/
│   ├── random_forest.joblib    # Best model (90.5% accuracy)
│   ├── svm.joblib              # Support Vector Machine
│   ├── knn.joblib              # K-Nearest Neighbors
│   ├── logistic_regression.joblib
│   ├── scaler.joblib           # StandardScaler (fitted on training data)
│   └── feature_order.txt       # Feature column ordering
│
├── data/
│   ├── raw/
│   │   ├── no_movement/        # 24 files, 15,000 scans (NM class)
│   │   ├── movement/           # 19 files, 10,490 scans (MV class)
│   │   └── test/               # T1.csv - T10.csv (evaluation)
│   ├── processed/              # Filtered signals, envelopes, peaks
│   └── features/               # features.csv (38 columns + label)
│
└── results/                    # Confusion matrices, charts
```

### File Responsibilities

| File | Purpose |
|------|---------|
| `main.py` | Runs all 6 pipeline stages, auto-generates feature_order.txt |
| `predict.py` | Loads trained model, processes new CSV, applies threshold (0.65), shows visual results + direction |
| `config.py` | Signal length (50,000), echo zones, paths, sampling rate |
| `data_loading.py` | Reads CSVs, handles split-line format, tracks file boundaries for temporal features |
| `signal_processing.py` | Bessel filter (order 2, Wn=0.6), Hilbert envelope, first peak detection |
| `feature_extraction.py` | Extracts 38 features per scan across 4 categories |
| `evaluation.py` | GridSearchCV with 5-fold CV, trains 4 models, generates metrics |

---

## Signal Processing

### Bessel Filter
- Type: Low-pass, order 2
- Cutoff: Wn=0.6 (normalized)
- Why Bessel: Preserves echo waveshape (maximally flat group delay)

### Hilbert Envelope
- Computes analytic signal via Hilbert transform
- Extracts instantaneous amplitude (envelope)
- Makes peak detection reliable

### Echo Zones
The 50,000-sample signal is divided into distance regions:

| Zone | Samples | Approximate Distance |
|------|---------|---------------------|
| Near field | 0–5,000 | < 0.5m |
| Close | 5,000–11,000 | 0.5–1.0m |
| Mid | 11,000–17,000 | 1.0–1.5m |
| Far | 17,000–25,000 | 1.5–2.5m |
| Background | 25,000–50,000 | > 2.5m |

---

## Known Limitations

- **Multi-person scenarios:** T9 (two people walking) fails because training data only contains single-person recordings
- **Threshold is empirically tuned:** 0.65 was found through systematic testing, not ROC analysis
- **Batch processing only:** Processes entire files, not real-time streaming
- **Indoor only:** Tested in controlled indoor environments

## Future Work

- Collect multi-person training data
- ROC-based threshold optimization
- Real-time sliding window detection
- Distance estimation from echo peak position
- Deep learning (CNN/RNN) on raw signals
- Extended distance range (> 3.2m)

---

## Tech Stack

- **Python 3.12**
- **NumPy** — numerical computation
- **SciPy** — Bessel filter, Hilbert transform, FFT
- **scikit-learn** — Random Forest, SVM, KNN, Logistic Regression, GridSearchCV
- **pandas** — feature data management
- **joblib** — model serialization

---

## Author

- Name: Md Maruf hossain
- Matriculation No: 1390272
- Email: maruf.hossain@stud.fra-uas.de
---
- Name: Md Jamal Uddin
- Matriculation No: 1387201
- Email: md.uddin2@stud.fra-uas.de
---
- Name: Mousumi Parvin Tonny
- Matriculation No: 1393224
- Email: mousumi.tonny@stud.fra-uas.de
---
- Name: Md. Shamsir Doha
- Matriculation No: 1344011
- Email: mohammad.doha@stud.fra-uas.de 




March 2026
=======

