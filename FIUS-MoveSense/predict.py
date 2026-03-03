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




def detect_direction(filtered_signals, envelopes, predictions):
    """
    Detect movement direction by tracking envelope peak position in echo zone.
    
    The echo zone (samples 0-15000) contains the reflected signal from objects.
    When someone walks TOWARD the sensor, the echo peak shifts to earlier samples.
    When someone walks AWAY, the echo peak shifts to later samples.
    
    Uses envelope maximum in 0-15000 range (not first_peak_idx which is noisy).
    """
    ECHO_RANGE = 15000  # Only look at echo zone
    
    # Get envelope peak position for each scan in echo zone
    peak_positions = []
    for i in range(envelopes.shape[0]):
        echo_env = envelopes[i][:ECHO_RANGE]
        peak_pos = np.argmax(echo_env)
        peak_positions.append(peak_pos)
    
    peak_positions = np.array(peak_positions, dtype=float)
    
    # Only analyze scans predicted as MOVING
    moving_mask = predictions == 1
    if np.sum(moving_mask) < 5:
        return "UNKNOWN", 0.0, peak_positions
    
    moving_indices = np.where(moving_mask)[0]
    moving_peaks = peak_positions[moving_mask]
    
    # Smooth the peaks to reduce noise (moving average, window=10)
    window = min(10, len(moving_peaks))
    if window > 1:
        kernel = np.ones(window) / window
        smoothed = np.convolve(moving_peaks, kernel, mode='valid')
    else:
        smoothed = moving_peaks
    
    # Calculate overall trend: compare first quarter vs last quarter
    n = len(smoothed)
    quarter = max(1, n // 4)
    first_quarter_mean = np.mean(smoothed[:quarter])
    last_quarter_mean = np.mean(smoothed[-quarter:])
    
    total_shift = last_quarter_mean - first_quarter_mean
    
    # Linear regression for slope
    x = np.arange(len(smoothed), dtype=float)
    y = smoothed
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    slope = numerator / denominator if denominator > 0 else 0
    
    # Direction threshold: need at least 200 samples shift to be meaningful
    # (T5 showed ~1400 shift, T6 showed ~4600 shift, T7 should be <200)
    min_shift = 200
    
    if abs(total_shift) < min_shift:
        direction = "LATERAL / STATIONARY"
    elif total_shift < 0:
        direction = "APPROACHING sensor"
    else:
        direction = "MOVING AWAY from sensor"
    
    return direction, total_shift, peak_positions


def predict(filepath, model_name="random_forest", threshold=0.65):
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

    # Predict using probability threshold
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(X_scaled)
        # Column 1 = probability of MOVING (class 1)
        move_proba = probas[:, 1]
        predictions = (move_proba >= threshold).astype(int)
        print("\nUsing confidence threshold: {:.0f}%".format(threshold * 100))
        print("  (Higher threshold = fewer false MOVING predictions)")
        print("  Avg movement probability: {:.1f}%".format(move_proba.mean() * 100))
    else:
        predictions = model.predict(X_scaled)

    # Calculate results
    moving = int(np.sum(predictions == 1))
    not_moving = int(np.sum(predictions == 0))
    total = len(predictions)
    moving_pct = moving / total * 100
    not_moving_pct = not_moving / total * 100

    # Calculate average probability if available
    avg_prob = move_proba.mean() * 100 if 'move_proba' in dir() else moving_pct

    # Build visual bars (28 chars wide)
    bar_width = 28
    mv_bars = int(round(moving_pct / 100 * bar_width))
    nm_bars = int(round(not_moving_pct / 100 * bar_width))
    conf_bars = int(round(avg_prob / 100 * bar_width))

    mv_bar_str = chr(9608) * mv_bars + chr(9617) * (bar_width - mv_bars)
    nm_bar_str = chr(9608) * nm_bars + chr(9617) * (bar_width - nm_bars)
    conf_bar_str = chr(9608) * conf_bars + chr(9617) * (bar_width - conf_bars)

    # Determine verdict
    is_moving = moving > not_moving
    verdict_symbol = chr(10003) if is_moving else chr(10005)

    # Get filename only
    fname = os.path.basename(filepath)

    # Print clean visual output
    print("")
    print("=" * 58)
    print("  FIUS-MoveSense: Prediction Results")
    print("=" * 58)
    print("  File: {}  |  Model: {}  |  Threshold: {:.0f}%".format(
        fname, model_name, threshold * 100))
    print("-" * 58)
    print("")
    print("  Moving:     {:>3}/{:<3} ({:>5.1f}%)  {}".format(
        moving, total, moving_pct, mv_bar_str))
    print("  Not Moving: {:>3}/{:<3} ({:>5.1f}%)  {}".format(
        not_moving, total, not_moving_pct, nm_bar_str))
    print("")
    print("  Confidence: {}  {:.1f}% avg probability".format(
        conf_bar_str, avg_prob))
    print("")

    if is_moving:
        print("  VERDICT: {} Movement DETECTED".format(verdict_symbol))
        
        # Direction detection
        direction, peak_shift, peak_indices = detect_direction(
            filtered, envelopes, predictions)
        
        if "APPROACHING" in direction:
            arrow = chr(8594) + " "  # right arrow
            print("  DIRECTION: {} {} (peak shift: {:.0f})".format(
                arrow, direction, peak_shift))
        elif "AWAY" in direction:
            arrow = chr(8592) + " "  # left arrow
            print("  DIRECTION: {} {} (peak shift: {:.0f})".format(
                arrow, direction, peak_shift))
        else:
            arrow = chr(8596) + " "  # left-right arrow
            print("  DIRECTION: {} {} (peak shift: {:.0f})".format(
                arrow, direction, peak_shift))
    else:
        print("  VERDICT: {} No significant movement detected".format(verdict_symbol))

    print("=" * 58)

    # Show scan-by-scan details only if --detail flag or few scans
    if "--detail" in sys.argv or total <= 20:
        print("")
        print("  Scan-by-scan details:")
        print("  " + "-" * 40)
        for i, pred in enumerate(predictions):
            label = "MOVING" if pred == 1 else "NOT MOVING"
            marker = ">>>" if pred == 1 else "   "
            print("  {} Scan {:4d}: {}".format(marker, i + 1, label))

    return predictions


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_csv_file> [model_name] [threshold]")
        print("")
        print("Models: random_forest (default), svm, knn, logistic_regression")
        print("Threshold: 0.0-1.0 (default 0.5, higher = fewer false MOVING)")
        print("")
        print("Example: python predict.py data/raw/new_recording.csv")
        print("Example: python predict.py data/raw/test/T1.csv random_forest 0.65")
        sys.exit(1)

    filepath = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "random_forest"
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.65

    if not os.path.exists(filepath):
        print("ERROR: File not found: {}".format(filepath))
        sys.exit(1)

    predict(filepath, model_name, threshold)
