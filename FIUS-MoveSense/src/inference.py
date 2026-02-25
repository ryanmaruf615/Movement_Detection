"""
FIUS-MoveSense: Inference Module
==================================
Load trained models and predict on new unseen signals.
"""

import os
import sys
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.signal_processing import apply_bessel_filter, compute_envelope, detect_first_peak
from src.feature_extraction import extract_all_features


def load_model(model_name="svm"):
    """Load a saved model and scaler."""
    name_map = {
        "rf": "random_forest.joblib",
        "random_forest": "random_forest.joblib",
        "lr": "logistic_regression.joblib",
        "logistic_regression": "logistic_regression.joblib",
        "knn": "knn.joblib",
        "svm": "svm.joblib",
    }
    fname = name_map.get(model_name.lower(), f"{model_name}.joblib")
    model = joblib.load(os.path.join(config.MODELS_DIR, fname))
    scaler = joblib.load(os.path.join(config.MODELS_DIR, "scaler.joblib"))
    return model, scaler


def predict_signal(signal, model, scaler):
    """Predict a single raw signal (1D numpy array).
    
    Returns: label (0=Not Moving, 1=Moving), feature dict
    """
    # Reshape if needed
    sig = signal.reshape(1, -1)

    # Process
    filtered = apply_bessel_filter(sig)
    envelope = compute_envelope(filtered)
    first_peak = detect_first_peak(filtered[0], envelope[0])

    # Extract features
    features = extract_all_features(filtered[0], first_peak)
    feature_names = sorted(features.keys())
    X = np.array([[features[k] for k in feature_names]])

    # Scale and predict
    X_scaled = scaler.transform(X)
    label = model.predict(X_scaled)[0]

    return label, features


def predict_batch(signals, model_name="svm"):
    """Predict a batch of raw signals.
    
    Args:
        signals: numpy array of shape (n_scans, signal_length)
        model_name: which model to use

    Returns:
        predictions: array of labels
    """
    model, scaler = load_model(model_name)
    predictions = []
    for i in range(signals.shape[0]):
        label, _ = predict_signal(signals[i], model, scaler)
        predictions.append(label)
    return np.array(predictions)


if __name__ == "__main__":
    # Quick demo: predict on a few test samples
    signals = np.load(os.path.join(config.PROCESSED_DIR, "signals.npy"))
    labels = np.load(os.path.join(config.PROCESSED_DIR, "labels.npy"))

    model, scaler = load_model("svm")

    print("Inference Demo (first 5 samples):")
    for i in range(min(5, len(signals))):
        pred, _ = predict_signal(signals[i], model, scaler)
        actual = labels[i]
        status = "✓" if pred == actual else "✗"
        print(f"  Sample {i}: Predicted={config.LABEL_NAMES[pred]}, Actual={config.LABEL_NAMES[actual]} {status}")
