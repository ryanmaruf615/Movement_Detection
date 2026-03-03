"""
FIUS-MoveSense: Evaluation Module
===================================
Evaluates all 4 models, generates confusion matrices,
comparison charts, and a summary evaluation table.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a single model and return metrics."""
    y_pred = model.predict(X_test)

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
    }

    return metrics, y_pred


def plot_confusion_matrix(y_test, y_pred, model_name, save_path):
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Not Moving", "Moving"],
                yticklabels=["Not Moving", "Moving"],
                annot_kws={"size": 16}, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_model_comparison(results_df, save_path):
    """Generate a grouped bar chart comparing all 4 models."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    models = results_df["Model"].tolist()

    x = np.arange(len(metrics))
    width = 0.18
    offsets = np.arange(len(models)) - (len(models) - 1) / 2

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    for i, model_name in enumerate(models):
        values = [results_df.loc[results_df["Model"] == model_name, m].values[0] for m in metrics]
        bars = ax.bar(x + offsets[i] * width, values, width, label=model_name, color=colors[i])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_signal_examples(save_dir):
    """Plot example signals for movement vs no movement."""
    try:
        signals = np.load(os.path.join(config.PROCESSED_DIR, "signals.npy"))
        labels = np.load(os.path.join(config.PROCESSED_DIR, "labels.npy"))
        filtered = np.load(os.path.join(config.PROCESSED_DIR, "filtered.npy"))
    except FileNotFoundError:
        print("  Signal files not found, skipping signal plots.")
        return

    fs = config.SAMPLING_RATE
    t = np.arange(signals.shape[1]) / fs * 1000  # time in ms

    # Find one example of each class
    nm_idx = np.where(labels == 0)[0][0]
    mv_idx = np.where(labels == 1)[0][0]

    # Plot raw signals comparison
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(t, signals[nm_idx], color='#2196F3', linewidth=0.3)
    axes[0].set_title("No Movement (Raw Signal)", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("ADC Value")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, signals[mv_idx], color='#E91E63', linewidth=0.3)
    axes[1].set_title("Movement (Raw Signal)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("ADC Value")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "raw_signal_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot filtered vs raw
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(t, signals[nm_idx], color='gray', linewidth=0.3, alpha=0.5, label='Raw')
    axes[0].plot(t, filtered[nm_idx], color='#2196F3', linewidth=0.5, label='Filtered')
    axes[0].set_title("Raw vs Filtered Signal (No Movement)", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("ADC Value")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, signals[mv_idx], color='gray', linewidth=0.3, alpha=0.5, label='Raw')
    axes[1].plot(t, filtered[mv_idx], color='#E91E63', linewidth=0.5, label='Filtered')
    axes[1].set_title("Raw vs Filtered Signal (Movement)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("ADC Value")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "filtered_vs_raw.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print("  Signal example plots saved.")


def evaluate_all_models():
    """Evaluate all 4 trained models and generate all outputs."""
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    # Load test data
    X_test = np.load(os.path.join(config.PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(config.PROCESSED_DIR, "y_test.npy"))
    print(f"Test set: {X_test.shape[0]} samples")

    # Load all models
    model_files = {
        "Random Forest": "random_forest.joblib",
        "Logistic Regression": "logistic_regression.joblib",
        "KNN": "knn.joblib",
        "SVM": "svm.joblib",
    }

    all_results = []
    for name, filename in model_files.items():
        path = os.path.join(config.MODELS_DIR, filename)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping {name}")
            continue

        model = joblib.load(path)
        metrics, y_pred = evaluate_model(model, X_test, y_test, name)
        all_results.append(metrics)

        # Print metrics
        print(f"\n  {name}:")
        print(f"    Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"    Precision: {metrics['Precision']:.4f}")
        print(f"    Recall:    {metrics['Recall']:.4f}")
        print(f"    F1-Score:  {metrics['F1-Score']:.4f}")

        # Confusion matrix
        cm_path = os.path.join(config.CONFUSION_DIR, f"{filename.replace('.joblib', '_cm.png')}")
        plot_confusion_matrix(y_test, y_pred, name, cm_path)
        print(f"    Confusion matrix saved: {cm_path}")

    # Create comparison table
    results_df = pd.DataFrame(all_results)
    table_path = os.path.join(config.RESULTS_DIR, "evaluation_table.csv")
    results_df.to_csv(table_path, index=False)

    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))

    # Find best model
    best_idx = results_df["F1-Score"].idxmax()
    best_model = results_df.loc[best_idx, "Model"]
    best_f1 = results_df.loc[best_idx, "F1-Score"]
    print(f"\n  BEST MODEL: {best_model} (F1-Score: {best_f1:.4f})")

    # Generate comparison chart
    comparison_path = os.path.join(config.RESULTS_DIR, "model_comparison.png")
    plot_model_comparison(results_df, comparison_path)
    print(f"  Comparison chart saved: {comparison_path}")

    # Generate signal example plots
    print("\nGenerating signal plots...")
    try:
        try:
        plot_signal_examples(config.SIGNAL_PLOTS_DIR)
    except (MemoryError, np._core._exceptions._ArrayMemoryError, Exception) as e:
        print(f"  WARNING: Signal plots skipped (not enough memory: {e})")
        print("  This does not affect model training or predictions.")
    except (MemoryError, np._core._exceptions._ArrayMemoryError, Exception) as e:
        print(f"  WARNING: Signal plots skipped (not enough memory: {e})")
        print("  This does not affect model training or predictions.")

    print("\nEvaluation complete!")
    return results_df


if __name__ == "__main__":
    evaluate_all_models()
