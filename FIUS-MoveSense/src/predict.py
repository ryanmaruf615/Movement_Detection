# FIUS-MoveSense: Predict on New Data
# =====================================
# Usage:
#     python predict.py path/to/new_data.csv
#     python predict.py path/to/new_data.csv svm
#
# Models available: random_forest (default), svm, knn, logistic_regression

import os
import sys
import re
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.signal_processing import apply_bessel_filter, compute_envelope, detect_first_peak
from src.feature_extraction import extract_all_features


def load_feature_order():
    """Load the correct feature order used during training."""
    order_file = os.path.join(config.MODELS_DIR, "feature_order.txt")
    with open(order_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_new_csv(filepath):
    """Load a CSV file - auto-detects clean or split format."""
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        first_line = f.readline().strip()

    col_count = first_line.count(',') + 1

    if col_count >= 50000:
        print("  Format: Clean (one scan per line)")
        return _parse_clean(filepath)
    else:
        print("  Format: Split lines (reassembling scans...)")
        return _parse_split(filepath)


def _parse_clean(filepath):
    """Parse clean CSV where each row is one complete scan."""
    signals = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = line.split(',')
            if len(values) < config.HEADER_FIELDS + 100:
                continue
            try:
                sig = [float(v) for v in values[config.HEADER_FIELDS:config.HEADER_FIELDS + config.EXPECTED_SIGNAL_LENGTH]]
            except ValueError:
                continue
            if len(sig) < config.EXPECTED_SIGNAL_LENGTH:
                sig.extend([0.0] * (config.EXPECTED_SIGNAL_LENGTH - len(sig)))
            sig = sig[:config.EXPECTED_SIGNAL_LENGTH]
            signals.append(sig)
    if len(signals) == 0:
        return np.array([]).reshape(0, config.EXPECTED_SIGNAL_LENGTH)
    return np.array(signals, dtype=np.float64)


def _parse_split(filepath):
    """Parse CSV with lines split at 32KB boundaries."""
    full_text = ""
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for line in f:
            full_text += line.rstrip('\n').rstrip('\r')

    full_text = re.sub(r'(\d)-', r'\1,-', full_text)
    full_text = re.sub(r'\.0(\d)', r'.0,\1', full_text)

    HEADER_CONSUMED = 5
    HEADER_REMAINING = config.HEADER_FIELDS - HEADER_CONSUMED

    parts = full_text.split(config.HEADER_PATTERN)
    parts = [p for p in parts if p.strip()]

    signals = []
    min_signal_samples = config.EXPECTED_SIGNAL_LENGTH - 1000

    for part in parts:
        values = part.split(',')
        values = [v.strip() for v in values if v.strip()]

        total_signal = len(values) - HEADER_REMAINING
        if total_signal < min_signal_samples:
            continue

        try:
            sig = [float(v) for v in values[HEADER_REMAINING:HEADER_REMAINING + config.EXPECTED_SIGNAL_LENGTH]]
        except (ValueError, IndexError):
            continue

        if len(sig) < config.EXPECTED_SIGNAL_LENGTH:
            sig.extend([0.0] * (config.EXPECTED_SIGNAL_LENGTH - len(sig)))
        sig = sig[:config.EXPECTED_SIGNAL_LENGTH]
        signals.append(sig)

    if len(signals) == 0:
        return np.array([]).reshape(0, config.EXPECTED_SIGNAL_LENGTH)
    return np.array(signals, dtype=np.float64)


def predict(filepath, model_name="random_forest"):
    print("=" * 50)
    print("FIUS-MoveSense: Prediction")
    print("=" * 50)
    print("File: {}".format(filepath))

    signals = load_new_csv(filepath)
    print("Scans loaded: {}\n".format(signals.shape[0]))

    if signals.shape[0] == 0:
        print("ERROR: No valid scans found in the file.")
        print("Check that the file is a valid FIUS sensor CSV.")
        return

    # Load trained model, scaler, and feature order
    model = joblib.load(os.path.join(config.MODELS_DIR, "{}.joblib".format(model_name)))
    scaler = joblib.load(os.path.join(config.MODELS_DIR, "scaler.joblib"))
    feature_order = load_feature_order()
    print("Model: {}\n".format(model_name))

    # Process signals
    print("Processing signals...")
    filtered = apply_bessel_filter(signals)
    envelopes = compute_envelope(filtered)

    # Extract features in the CORRECT order (same as training)
    # V3: Pass temporal context (previous and next scan) for each scan
    print("Extracting features...")
    all_features = []
    n = filtered.shape[0]
    for i in range(n):
        first_peak = detect_first_peak(filtered[i], envelopes[i])
        prev_sig = filtered[i - 1] if i > 0 else None
        next_sig = filtered[i + 1] if i < n - 1 else None
        feats = extract_all_features(filtered[i], first_peak,
                                     prev_signal=prev_sig, next_signal=next_sig)
        # Use training feature order, NOT sorted
        all_features.append([feats[k] for k in feature_order])

        if (i + 1) % 200 == 0:
            print("  {}/{} scans processed".format(i + 1, signals.shape[0]))

    X = np.array(all_features)
    X_scaled = scaler.transform(X)

    # Predict
    predictions = model.predict(X_scaled)

    # Print results
    print("")
    print("=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)

    for i, pred in enumerate(predictions):
        label = "MOVING" if pred == 1 else "NOT MOVING"
        marker = ">>>" if pred == 1 else "   "
        print("  {} Scan {:4d}: {}".format(marker, i + 1, label))

    # Summary
    moving = int(np.sum(predictions == 1))
    not_moving = int(np.sum(predictions == 0))
    total = len(predictions)

    print("")
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print("  Total scans:  {}".format(total))
    print("  Moving:       {} ({:.1f}%)".format(moving, moving / total * 100))
    print("  Not Moving:   {} ({:.1f}%)".format(not_moving, not_moving / total * 100))

    if moving > not_moving:
        print("\n  VERDICT: Movement DETECTED in this recording")
    else:
        print("\n  VERDICT: No significant movement detected")

    print("=" * 50)
    return predictions


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_csv_file> [model_name]")
        print("")
        print("Models: random_forest (default), svm, knn, logistic_regression")
        print("")
        print("Example: python predict.py data/raw/new_recording.csv")
        sys.exit(1)

    filepath = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "random_forest"

    if not os.path.exists(filepath):
        print("ERROR: File not found: {}".format(filepath))
        sys.exit(1)

    predict(filepath, model_name)
