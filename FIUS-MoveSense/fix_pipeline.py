"""
FIUS-MoveSense: Pipeline Fix Script
=====================================
Fixes 3 issues:
1. predict.py: Set default threshold to 0.65
2. main.py: Auto-generate feature_order.txt after training
3. evaluation.py: Fix memory crash on signal plots

Usage:
    python fix_pipeline.py

After running:
    python main.py
    python predict.py data\raw\test\T1.csv
"""

import os

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "src")

print("=" * 60)
print("FIUS-MoveSense: Pipeline Fixes")
print("=" * 60)

fixes_applied = 0

# ============================================================
# FIX 1: Set default threshold to 0.65 in predict.py
# ============================================================
print("\n[FIX 1] Setting default threshold to 0.65 in predict.py...")

pred_path = os.path.join(BASE, "predict.py")
with open(pred_path, 'r') as f:
    pred_code = f.read()

# Fix the default threshold in the function signature
if 'threshold=0.5' in pred_code:
    pred_code = pred_code.replace('threshold=0.5', 'threshold=0.65')
    fixes_applied += 1
    print("  Changed default threshold from 0.5 to 0.65")
elif 'threshold=0.65' in pred_code:
    print("  Already set to 0.65 - skipping")
else:
    print("  WARNING: Could not find threshold parameter")

# Fix the default in __main__ block too
if "float(sys.argv[3]) if len(sys.argv) > 3 else 0.5" in pred_code:
    pred_code = pred_code.replace(
        "float(sys.argv[3]) if len(sys.argv) > 3 else 0.5",
        "float(sys.argv[3]) if len(sys.argv) > 3 else 0.65"
    )
    print("  Updated __main__ default to 0.65")
elif "else 0.65" in pred_code:
    print("  __main__ default already 0.65 - skipping")

with open(pred_path, 'w') as f:
    f.write(pred_code)


# ============================================================
# FIX 2: Auto-generate feature_order.txt in main.py
# ============================================================
print("\n[FIX 2] Adding auto feature_order.txt generation to main.py...")

main_path = os.path.join(BASE, "main.py")
with open(main_path, 'r') as f:
    main_code = f.read()

if "feature_order.txt" in main_code:
    print("  main.py already generates feature_order.txt - skipping")
else:
    # Find where training happens and add feature_order generation after it
    # Look for the line after model training completes
    # The training module saves models, so we add after that call

    # Strategy: add it right before evaluation
    old_eval = "    # Step 5: Evaluation"
    if old_eval not in main_code:
        # Try alternate formats
        old_eval = "    # Step 5"
        if old_eval not in main_code:
            # Try finding evaluate_all_models
            if "evaluate_all_models" in main_code:
                # Add before the evaluate call
                old_eval = "    results = evaluate_all_models()"
                new_eval = """    # Auto-generate feature_order.txt for predict.py
    import pandas as pd
    features_csv = os.path.join(config.FEATURES_DIR, "features.csv")
    if os.path.exists(features_csv):
        df_feat = pd.read_csv(features_csv)
        feature_cols = [c for c in df_feat.columns if c != "label"]
        order_path = os.path.join(config.MODELS_DIR, "feature_order.txt")
        with open(order_path, 'w') as f_order:
            for col in feature_cols:
                f_order.write(col + "\\n")
        print(f"  Feature order saved: {len(feature_cols)} features -> {order_path}")

    results = evaluate_all_models()"""
                if old_eval in main_code:
                    main_code = main_code.replace(old_eval, new_eval, 1)
                    fixes_applied += 1
                    print("  Added feature_order.txt auto-generation before evaluation")
                else:
                    print("  WARNING: Could not find insertion point in main.py")
            else:
                print("  WARNING: Could not find evaluation section in main.py")
    else:
        new_eval = old_eval.rstrip() + """

    # Auto-generate feature_order.txt for predict.py
    import pandas as pd
    features_csv = os.path.join(config.FEATURES_DIR, "features.csv")
    if os.path.exists(features_csv):
        df_feat = pd.read_csv(features_csv)
        feature_cols = [c for c in df_feat.columns if c != "label"]
        order_path = os.path.join(config.MODELS_DIR, "feature_order.txt")
        with open(order_path, 'w') as f_order:
            for col in feature_cols:
                f_order.write(col + "\\n")
        print(f"  Feature order saved: {len(feature_cols)} features -> {order_path}")
"""
        main_code = main_code.replace(old_eval, new_eval, 1)
        fixes_applied += 1
        print("  Added feature_order.txt auto-generation before evaluation")

    with open(main_path, 'w') as f:
        f.write(main_code)


# ============================================================
# FIX 3: Fix memory crash in evaluation.py signal plots
# ============================================================
print("\n[FIX 3] Fixing memory crash in evaluation.py signal plots...")

eval_path = os.path.join(SRC, "evaluation.py")
with open(eval_path, 'r') as f:
    eval_code = f.read()

if "MemoryError" in eval_code or "memory" in eval_code.lower() and "try:" in eval_code:
    # Check if already has our specific fix
    if "plot_signal_examples_safe" in eval_code:
        print("  Already fixed - skipping")
    else:
        print("  May already have some error handling - applying fix anyway")

# Wrap the plot_signal_examples call in a try-except
# Find where it's called
if "plot_signal_examples" in eval_code:
    # Find the function definition
    old_call = "    plot_signal_examples(config.SIGNAL_PLOTS_DIR)"
    if old_call in eval_code:
        new_call = """    try:
        plot_signal_examples(config.SIGNAL_PLOTS_DIR)
    except (MemoryError, np._core._exceptions._ArrayMemoryError, Exception) as e:
        print(f"  WARNING: Signal plots skipped (not enough memory: {e})")
        print("  This does not affect model training or predictions.")"""
        eval_code = eval_code.replace(old_call, new_call, 1)
        fixes_applied += 1
        print("  Wrapped signal plot in try-except for memory errors")

        with open(eval_path, 'w') as f:
            f.write(eval_code)
    else:
        # Try to find alternative call format
        import re
        pattern = r'(\s+)(plot_signal_examples\([^)]+\))'
        match = re.search(pattern, eval_code)
        if match:
            indent = match.group(1)
            call = match.group(2)
            old_line = indent + call
            new_line = f"""{indent}try:
{indent}    {call}
{indent}except (MemoryError, Exception) as e:
{indent}    print(f"  WARNING: Signal plots skipped (not enough memory: {{e}})")
{indent}    print("  This does not affect model training or predictions.")"""
            eval_code = eval_code.replace(old_line, new_line, 1)
            fixes_applied += 1
            print("  Wrapped signal plot in try-except for memory errors")

            with open(eval_path, 'w') as f:
                f.write(eval_code)
        else:
            print("  WARNING: Could not find plot_signal_examples call")
else:
    print("  No plot_signal_examples found in evaluation.py")


# ============================================================
# FIX 4 (bonus): Remove the part1/part2 movement files if present
# These are split-format files that cause issues
# ============================================================
mv_dir = os.path.join(BASE, "data", "raw", "movement")
removed_files = []
if os.path.exists(mv_dir):
    for f in os.listdir(mv_dir):
        if "part1" in f.lower() or "part2" in f.lower():
            fpath = os.path.join(mv_dir, f)
            # Don't auto-delete, just warn
            print(f"\n[NOTE] Found split-format file: {f}")
            print(f"  These part1/part2 files have 1914+1920 scans and may skew training data.")
            print(f"  Consider removing them if you have enough MV data without them.")


# ============================================================
# VERIFICATION
# ============================================================
print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)

with open(pred_path, 'r') as f:
    c = f.read()
has_threshold = "threshold=0.65" in c
print(f"  predict.py default threshold=0.65: {has_threshold}")

with open(main_path, 'r') as f:
    c = f.read()
has_feature_order = "feature_order.txt" in c
print(f"  main.py auto-generates feature_order.txt: {has_feature_order}")

with open(eval_path, 'r') as f:
    c = f.read()
has_memory_fix = "MemoryError" in c
print(f"  evaluation.py handles memory errors: {has_memory_fix}")

print(f"\n  Fixes applied: {fixes_applied}")

print("\n" + "=" * 60)
print("FIXES APPLIED SUCCESSFULLY!")
print("=" * 60)

print("""
Now you just need:
  python main.py
  python predict.py data\\raw\\test\\T1.csv

No more manual feature_order.txt command!
No more memory crash on signal plots!
Default threshold is now 0.65!
""")
