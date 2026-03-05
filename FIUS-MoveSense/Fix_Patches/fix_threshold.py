"""
FIUS-MoveSense: Add threshold tuning to predict.py
====================================================
Run this once, then use:
    python predict.py data\raw\test\T1.csv
    python predict.py data\raw\test\T1.csv random_forest 0.65

The third argument is the confidence threshold (default 0.5).
Higher = model needs more confidence to say MOVING.
"""

import os

BASE = os.path.dirname(os.path.abspath(__file__))
pred_path = os.path.join(BASE, "predict.py")

with open(pred_path, 'r') as f:
    code = f.read()

# Check if already patched
if "threshold" in code:
    print("predict.py already has threshold support - skipping")
else:
    # 1. Update the predict function signature
    code = code.replace(
        'def predict(filepath, model_name="random_forest"):',
        'def predict(filepath, model_name="random_forest", threshold=0.5):'
    )

    # 2. Replace the simple predict with threshold-based predict
    code = code.replace(
        '''    # Predict
    predictions = model.predict(X_scaled)''',
        '''    # Predict using probability threshold
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(X_scaled)
        # Column 1 = probability of MOVING (class 1)
        move_proba = probas[:, 1]
        predictions = (move_proba >= threshold).astype(int)
        print("\\nUsing confidence threshold: {:.0f}%".format(threshold * 100))
        print("  (Higher threshold = fewer false MOVING predictions)")
        print("  Avg movement probability: {:.1f}%".format(move_proba.mean() * 100))
    else:
        predictions = model.predict(X_scaled)'''
    )

    # 3. Update the __main__ block to accept threshold argument
    code = code.replace(
        '''    filepath = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "random_forest"

    if not os.path.exists(filepath):
        print("ERROR: File not found: {}".format(filepath))
        sys.exit(1)

    predict(filepath, model_name)''',
        '''    filepath = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "random_forest"
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    if not os.path.exists(filepath):
        print("ERROR: File not found: {}".format(filepath))
        sys.exit(1)

    predict(filepath, model_name, threshold)'''
    )

    # 4. Update usage text
    code = code.replace(
        '        print("Usage: python predict.py <path_to_csv_file> [model_name]")',
        '        print("Usage: python predict.py <path_to_csv_file> [model_name] [threshold]")'
    )
    code = code.replace(
        '        print("Models: random_forest (default), svm, knn, logistic_regression")',
        '        print("Models: random_forest (default), svm, knn, logistic_regression")\n        print("Threshold: 0.0-1.0 (default 0.5, higher = fewer false MOVING)")'
    )
    code = code.replace(
        '        print("Example: python predict.py data/raw/new_recording.csv")',
        '        print("Example: python predict.py data/raw/new_recording.csv")\n        print("Example: python predict.py data/raw/test/T1.csv random_forest 0.65")'
    )

    with open(pred_path, 'w') as f:
        f.write(code)

    print("predict.py updated with threshold support!")
    print()
    print("Usage:")
    print("  python predict.py <file> [model] [threshold]")
    print()
    print("Now test different thresholds on T1:")
    print("  python predict.py data\\raw\\test\\T1.csv random_forest 0.55")
    print("  python predict.py data\\raw\\test\\T1.csv random_forest 0.60")
    print("  python predict.py data\\raw\\test\\T1.csv random_forest 0.65")
    print("  python predict.py data\\raw\\test\\T1.csv random_forest 0.70")
