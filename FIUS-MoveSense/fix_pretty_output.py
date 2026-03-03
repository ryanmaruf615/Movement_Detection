"""
FIUS-MoveSense: Upgrade predict.py output
==========================================
Replaces the verbose scan-by-scan output with a clean visual summary.

Usage:
    python fix_pretty_output.py

Then test:
    python predict.py data\raw\test\T5.csv
"""

import os

BASE = os.path.dirname(os.path.abspath(__file__))
pred_path = os.path.join(BASE, "predict.py")

with open(pred_path, 'r') as f:
    code = f.read()

# ============================================================
# Replace the PREDICTION RESULTS + SUMMARY section
# ============================================================

# Find and replace the results printing section
old_results = '''    # Print results
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
        print("\\n  VERDICT: Movement DETECTED in this recording")
    else:
        print("\\n  VERDICT: No significant movement detected")

    print("=" * 50)
    return predictions'''

new_results = '''    # Calculate results
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

    return predictions'''

if old_results in code:
    code = code.replace(old_results, new_results)
    print("[OK] Replaced output section with visual summary")
else:
    print("[WARNING] Could not find exact output section.")
    print("  Your predict.py may have been modified.")
    print("  Trying alternate approach...")
    
    # Try to find the section by key markers
    if "PREDICTION RESULTS" in code and "SUMMARY" in code:
        # Find the start: "# Print results"
        start_marker = "    # Print results"
        end_marker = "    return predictions"
        
        start_idx = code.find(start_marker)
        end_idx = code.find(end_marker)
        
        if start_idx != -1 and end_idx != -1:
            end_idx += len(end_marker)
            code = code[:start_idx] + new_results + code[end_idx:]
            print("[OK] Replaced output section (alternate method)")
        else:
            print("[FAILED] Could not locate output section boundaries")
            print("  You may need to manually update predict.py")
    else:
        print("[FAILED] predict.py structure not recognized")

with open(pred_path, 'w') as f:
    f.write(code)

print("")
print("Done! Now test:")
print("  python predict.py data\\raw\\test\\T5.csv")
print("")
print("For scan-by-scan details, add --detail:")
print("  python predict.py data\\raw\\test\\T5.csv random_forest 0.65 --detail")
