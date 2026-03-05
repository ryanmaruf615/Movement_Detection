"""
FIUS-MoveSense: Fix encoding error in predict.py
==================================================
Some CSV files use latin-1 or other encodings instead of UTF-8.
This fix tries multiple encodings so predict.py works with any file.

Usage:
    python fix_encoding.py
"""

import os

BASE = os.path.dirname(os.path.abspath(__file__))
pred_path = os.path.join(BASE, "predict.py")

with open(pred_path, 'r') as f:
    code = f.read()

# Replace all instances of: open(filepath, 'r', encoding='utf-8-sig')
# with a function that tries multiple encodings

# First, add an encoding detection helper after the imports
import_end = "from src.feature_extraction import extract_all_features"

if "def open_csv_safe" not in code:
    helper_function = '''

def open_csv_safe(filepath):
    """Open CSV file with auto-detected encoding."""
    encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for enc in encodings:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                f.readline()  # test read
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return 'latin-1'  # fallback - handles any byte
'''
    
    if import_end in code:
        code = code.replace(import_end, import_end + helper_function)
        print("[OK] Added open_csv_safe() helper function")
    else:
        print("[WARNING] Could not find import section, trying alternate")
        # Try inserting before detect_direction
        alt_marker = "\ndef detect_direction"
        if alt_marker in code:
            code = code.replace(alt_marker, helper_function + alt_marker)
            print("[OK] Added open_csv_safe() helper (alternate location)")
        else:
            print("[FAILED] Could not add helper function")

    # Now replace all hardcoded encoding opens in load_new_csv and _parse_clean and _parse_split
    # Replace in load_new_csv
    code = code.replace(
        "    with open(filepath, 'r', encoding='utf-8-sig') as f:\n        first_line = f.readline().strip()",
        "    enc = open_csv_safe(filepath)\n    with open(filepath, 'r', encoding=enc) as f:\n        first_line = f.readline().strip()"
    )
    print("[OK] Fixed encoding in load_new_csv()")

    # Replace in _parse_clean
    old_clean = "    with open(filepath, 'r', encoding='utf-8-sig') as f:\n        for line in f:"
    new_clean = "    enc = open_csv_safe(filepath)\n    with open(filepath, 'r', encoding=enc) as f:\n        for line in f:"
    
    # There might be multiple instances, replace all
    count = code.count(old_clean)
    if count > 0:
        code = code.replace(old_clean, new_clean)
        print("[OK] Fixed encoding in _parse_clean() ({} instances)".format(count))

    # Replace in _parse_split  
    old_split = "    with open(filepath, 'r', encoding='utf-8-sig') as f:\n        for line in f:\n            full_text += line"
    new_split = "    enc = open_csv_safe(filepath)\n    with open(filepath, 'r', encoding=enc) as f:\n        for line in f:\n            full_text += line"
    
    if old_split in code:
        code = code.replace(old_split, new_split)
        print("[OK] Fixed encoding in _parse_split()")
    else:
        print("[NOTE] _parse_split encoding not found (may already be fixed)")

    with open(pred_path, 'w') as f:
        f.write(code)

    print("")
    print("Done! Now test:")
    print("  python predict.py data\\raw\\movement\\signal_500__distance_2_0_to_0_5_meter_Person_walking_toward_sensor_MV.csv")
else:
    print("predict.py already has encoding fix - skipping")
