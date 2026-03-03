"""
FIUS-MoveSense: Add Direction Detection to predict.py
======================================================
After detecting movement, analyzes the echo peak trend
to determine if the person is APPROACHING or MOVING AWAY.

Usage:
    python fix_direction.py

Then test:
    python predict.py data\raw\test\T5.csv   (should show APPROACHING)
    python predict.py data\raw\test\T6.csv   (should show MOVING AWAY)
"""

import os

BASE = os.path.dirname(os.path.abspath(__file__))
pred_path = os.path.join(BASE, "predict.py")

with open(pred_path, 'r') as f:
    code = f.read()

if "DIRECTION" in code:
    print("predict.py already has direction detection - skipping")
else:
    # ============================================================
    # 1. Add the direction detection function after imports
    # ============================================================
    direction_function = '''

def detect_direction(filtered_signals, envelopes, predictions):
    """
    Detect movement direction by analyzing first_peak_idx trend.
    
    When someone walks TOWARD the sensor, echo arrives earlier (peak index decreases).
    When someone walks AWAY, echo arrives later (peak index increases).
    
    Only analyzes scans predicted as MOVING for cleaner signal.
    """
    from src.signal_processing import detect_first_peak
    
    # Get peak indices for ALL scans
    peak_indices = []
    for i in range(filtered_signals.shape[0]):
        peak = detect_first_peak(filtered_signals[i], envelopes[i])
        peak_indices.append(peak)
    
    peak_indices = np.array(peak_indices, dtype=float)
    
    # Only look at scans predicted as MOVING
    moving_mask = predictions == 1
    if np.sum(moving_mask) < 5:
        return "UNKNOWN", 0.0, peak_indices
    
    # Get moving scan indices and their peak positions
    moving_indices = np.where(moving_mask)[0]
    moving_peaks = peak_indices[moving_mask]
    
    # Calculate trend using linear regression (simple slope)
    n = len(moving_indices)
    x = moving_indices.astype(float)
    y = moving_peaks
    
    # Remove any NaN or zero peaks
    valid = (y > 0) & ~np.isnan(y)
    if np.sum(valid) < 5:
        return "UNKNOWN", 0.0, peak_indices
    
    x = x[valid]
    y = y[valid]
    
    # Linear regression: slope = direction
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    if denominator == 0:
        return "UNKNOWN", 0.0, peak_indices
    
    slope = numerator / denominator
    
    # Calculate R-squared for confidence
    y_pred = slope * (x - x_mean) + y_mean
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Total peak shift (first moving scan to last moving scan)
    total_shift = y[-1] - y[0]
    
    # Determine direction based on slope
    # Negative slope = peak index decreasing = object getting closer = APPROACHING
    # Positive slope = peak index increasing = object getting farther = MOVING AWAY
    
    # Use threshold to avoid classifying noise as direction
    slope_threshold = 0.5  # samples per scan minimum trend
    
    if abs(slope) < slope_threshold:
        direction = "LATERAL / STATIONARY"
    elif slope < 0:
        direction = "APPROACHING sensor"
    else:
        direction = "MOVING AWAY from sensor"
    
    return direction, total_shift, peak_indices

'''

    # Insert after the imports section - find a good insertion point
    # Look for the predict function definition
    insert_marker = "def predict(filepath"
    insert_idx = code.find(insert_marker)
    
    if insert_idx != -1:
        code = code[:insert_idx] + direction_function + "\n" + code[insert_idx:]
        print("[OK] Added detect_direction() function")
    else:
        print("[FAILED] Could not find predict function to insert before")

    # ============================================================
    # 2. Add direction detection call and display in output
    # ============================================================
    
    # Find the verdict section and add direction after it
    old_verdict = '''    if is_moving:
        print("  VERDICT: {} Movement DETECTED".format(verdict_symbol))
    else:
        print("  VERDICT: {} No significant movement detected".format(verdict_symbol))

    print("=" * 58)'''
    
    new_verdict = '''    if is_moving:
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

    print("=" * 58)'''
    
    if old_verdict in code:
        code = code.replace(old_verdict, new_verdict)
        print("[OK] Added direction display to output")
    else:
        print("[WARNING] Could not find verdict section for direction display")
        print("  Trying alternate approach...")
        
        # Try finding just the VERDICT line
        if 'VERDICT: {} Movement DETECTED' in code:
            old_line = '        print("  VERDICT: {} Movement DETECTED".format(verdict_symbol))'
            new_line = '''        print("  VERDICT: {} Movement DETECTED".format(verdict_symbol))
        
        # Direction detection
        direction, peak_shift, peak_indices = detect_direction(
            filtered, envelopes, predictions)
        
        if "APPROACHING" in direction:
            arrow = chr(8594) + " "
            print("  DIRECTION: {} {} (peak shift: {:.0f})".format(
                arrow, direction, peak_shift))
        elif "AWAY" in direction:
            arrow = chr(8592) + " "
            print("  DIRECTION: {} {} (peak shift: {:.0f})".format(
                arrow, direction, peak_shift))
        else:
            arrow = chr(8596) + " "
            print("  DIRECTION: {} {} (peak shift: {:.0f})".format(
                arrow, direction, peak_shift))'''
            
            code = code.replace(old_line, new_line, 1)
            print("[OK] Added direction display (alternate method)")
        else:
            print("[FAILED] Could not add direction display")

    with open(pred_path, 'w') as f:
        f.write(code)

print("")
print("=" * 50)
print("Direction detection added!")
print("=" * 50)
print("")
print("Test with:")
print("  python predict.py data\\raw\\test\\T5.csv    (should say APPROACHING)")
print("  python predict.py data\\raw\\test\\T6.csv    (should say MOVING AWAY)")
print("  python predict.py data\\raw\\test\\T7.csv    (should say LATERAL)")
