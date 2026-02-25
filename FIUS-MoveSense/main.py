"""
FIUS-MoveSense: Main Pipeline
===============================
Run the complete ML pipeline with a single command:
    python main.py

Pipeline steps:
  1. Load raw sensor data
  2. Apply signal processing (Bessel filter, envelope, peak detection)
  3. Extract features (time-domain, zone-based, frequency-domain)
  4. Train 4 ML models (Random Forest, Logistic Regression, KNN, SVM)
  5. Evaluate all models and generate results

To switch between sample and final data:
  - Edit config.py → DATA_MODE = "sample" or "final"
"""

import os
import sys
import time

# Ensure imports work from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_loading import load_data
from src.signal_processing import process_signals
from src.feature_extraction import build_feature_matrix
from src.model_training import train_all_models
from src.evaluation import evaluate_all_models


def main():
    print("=" * 60)
    print("  FIUS-MoveSense: Ultrasonic Movement Detection")
    print(f"  Data Mode: {config.DATA_MODE.upper()}")
    print("=" * 60)

    start_time = time.time()

    # Step 1: Load data
    print("\n[STEP 1/5] Loading raw sensor data...")
    signals, labels = load_data()

    # Step 2: Signal processing
    print("\n[STEP 2/5] Signal processing...")
    filtered, envelopes, first_peaks = process_signals(signals)

    # Step 3: Feature extraction
    print("\n[STEP 3/5] Feature extraction...")
    df = build_feature_matrix(filtered, first_peaks, labels)

    # Step 4: Model training
    print("\n[STEP 4/5] Model training...")
    models = train_all_models(df)

    # Step 5: Evaluation
    print("\n[STEP 5/5] Evaluation...")
    results = evaluate_all_models()

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  PIPELINE COMPLETE in {elapsed:.1f} seconds")
    print(f"  Results saved to: {config.RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
