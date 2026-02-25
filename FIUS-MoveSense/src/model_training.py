"""
FIUS-MoveSense: Model Training Module
=======================================
Trains 4 ML models (Random Forest, Logistic Regression, KNN, SVM)
with proper cross-validation, hyperparameter tuning, and scaler handling.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def prepare_data(df):
    """Split data into train/val/test with proper scaling.
    
    CRITICAL: Scaler is fit ONLY on training data.
    """
    X = df.drop("label", axis=1).values
    y = df["label"].values
    feature_names = [c for c in df.columns if c != "label"]

    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(config.TEST_SIZE + config.VAL_SIZE),
        random_state=config.RANDOM_STATE, stratify=y
    )

    # Second split: temp -> 50/50 val/test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5,
        random_state=config.RANDOM_STATE, stratify=y_temp
    )

    # Scale features - FIT ONLY ON TRAINING DATA
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Save the scaler
    scaler_path = os.path.join(config.MODELS_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)

    print(f"Data Split:")
    print(f"  Train: {X_train.shape[0]} samples ({np.sum(y_train==0)} NM / {np.sum(y_train==1)} MV)")
    print(f"  Val:   {X_val.shape[0]} samples ({np.sum(y_val==0)} NM / {np.sum(y_val==1)} MV)")
    print(f"  Test:  {X_test.shape[0]} samples ({np.sum(y_test==0)} NM / {np.sum(y_test==1)} MV)")
    print(f"  Scaler saved to {scaler_path}")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


def train_random_forest(X_train, y_train):
    """Train Random Forest with GridSearchCV."""
    print("\n  [1/4] Training Random Forest...")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
    }
    model = GridSearchCV(
        RandomForestClassifier(random_state=config.RANDOM_STATE),
        param_grid, cv=config.CV_FOLDS, scoring="f1", n_jobs=-1, verbose=0
    )
    model.fit(X_train, y_train)
    print(f"    Best params: {model.best_params_}")
    print(f"    Best CV F1:  {model.best_score_:.4f}")
    return model.best_estimator_


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression with GridSearchCV."""
    print("\n  [2/4] Training Logistic Regression...")
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
    }
    model = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE, l1_ratio=0),
        param_grid, cv=config.CV_FOLDS, scoring="f1", n_jobs=-1, verbose=0
    )
    model.fit(X_train, y_train)
    print(f"    Best params: {model.best_params_}")
    print(f"    Best CV F1:  {model.best_score_:.4f}")
    return model.best_estimator_


def train_knn(X_train, y_train):
    """Train K-Nearest Neighbors with GridSearchCV."""
    print("\n  [3/4] Training KNN...")
    param_grid = {
        "n_neighbors": [3, 5, 7, 11],
        "weights": ["uniform", "distance"],
    }
    model = GridSearchCV(
        KNeighborsClassifier(),
        param_grid, cv=config.CV_FOLDS, scoring="f1", n_jobs=-1, verbose=0
    )
    model.fit(X_train, y_train)
    print(f"    Best params: {model.best_params_}")
    print(f"    Best CV F1:  {model.best_score_:.4f}")
    return model.best_estimator_


def train_svm(X_train, y_train):
    """Train Support Vector Machine with GridSearchCV."""
    print("\n  [4/4] Training SVM...")
    param_grid = {
        "kernel": ["rbf"],
        "C": [1, 10, 100, 1000],
        "gamma": ["scale", 0.001, 0.01],
    }
    model = GridSearchCV(
        SVC(random_state=config.RANDOM_STATE),
        param_grid, cv=config.CV_FOLDS, scoring="f1", n_jobs=-1, verbose=0
    )
    model.fit(X_train, y_train)
    print(f"    Best params: {model.best_params_}")
    print(f"    Best CV F1:  {model.best_score_:.4f}")
    return model.best_estimator_


def train_all_models(df):
    """Train all 4 models and save them."""
    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(df)

    # Save test data for evaluation
    np.save(os.path.join(config.PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(config.PROCESSED_DIR, "y_test.npy"), y_test)
    np.save(os.path.join(config.PROCESSED_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(config.PROCESSED_DIR, "y_val.npy"), y_val)

    # Combine train+val for final model training (after hyperparameter tuning)
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    print(f"\nTraining on {X_train_full.shape[0]} samples, testing on {X_test.shape[0]}...")

    models = {}

    models["Random Forest"] = train_random_forest(X_train_full, y_train_full)
    models["Logistic Regression"] = train_logistic_regression(X_train_full, y_train_full)
    models["KNN"] = train_knn(X_train_full, y_train_full)
    models["SVM"] = train_svm(X_train_full, y_train_full)

    # Save all models
    model_filenames = {
        "Random Forest": "random_forest.joblib",
        "Logistic Regression": "logistic_regression.joblib",
        "KNN": "knn.joblib",
        "SVM": "svm.joblib",
    }
    for name, model in models.items():
        path = os.path.join(config.MODELS_DIR, model_filenames[name])
        joblib.dump(model, path)
        print(f"\n  Saved {name} -> {path}")

    # Save feature importance for Random Forest
    rf_model = models["Random Forest"]
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": rf_model.feature_importances_
    }).sort_values("importance", ascending=False)
    importance_df.to_csv(os.path.join(config.RESULTS_DIR, "feature_importance.csv"), index=False)
    print(f"\n  Top 10 most important features:")
    for _, row in importance_df.head(10).iterrows():
        print(f"    {row['feature']:30s} {row['importance']:.4f}")

    return models


if __name__ == "__main__":
    print("Loading features...")
    df = pd.read_csv(os.path.join(config.FEATURES_DIR, "features.csv"))
    models = train_all_models(df)
    print("\nAll models trained and saved!")
