"""
FIUS-MoveSense: Fix Direction Detection
=========================================
Uses envelope peak position in echo zone (0-15000) to detect direction.
Replaces the old first_peak_idx approach which was too noisy.

Usage:
    python fix_direction_v2.py
    python predict.py data\raw\test\T5.csv
"""

import os

BASE = os.path.dirname(os.path.abspath(__file__))
pred_path = os.path.join(BASE, "predict.py")

with open(pred_path, 'r') as f:
    code = f.read()

# Replace the old detect_direction function with the new one
old_func_start = "\ndef detect_direction("
old_func_end = "    return direction, total_shift, peak_indices\n"

start_idx = code.find(old_func_start)
end_idx = code.find(old_func_end)

if start_idx == -1:
    print("[FAILED] Could not find detect_direction function")
    print("  Run fix_direction.py first")
    exit()

end_idx += len(old_func_end)

new_function = '''
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
'''

code = code[:start_idx] + new_function + code[end_idx:]

with open(pred_path, 'w') as f:
    f.write(code)

print("[OK] Updated detect_direction() with envelope peak method")
print("")
print("Test with:")
print("  python predict.py data\\raw\\test\\T5.csv    (APPROACHING)")
print("  python predict.py data\\raw\\test\\T6.csv    (MOVING AWAY)")
print("  python predict.py data\\raw\\test\\T7.csv    (LATERAL)")
