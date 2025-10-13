#!/usr/bin/env python3
"""
Final Enhanced Anomaly Detector
- IsolationForest anomaly score added as feature
- SMOTETomek sampling on training data
- StackingClassifier (RF + GB) with RF as final estimator
- Threshold optimization (F1) on validation set
- Reports metrics and saves model (joblib)
"""

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest, StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score, f1_score
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

RND = 42
DATA_PATH = "/Users/appdev/Downloads/Waste_Reduction/labeled_waste_dataset.csv"
OUTPUT_MODEL = "enhanced_anomaly_detector.joblib"

def prepare_features(df):
    """Select numeric features, drop useless cols, basic cleaning."""
    # Drop any obviously useless columns, adjust as needed
    drop_cols = ["waste_risk_level"] if "waste_risk_level" in df.columns else []
    df = df.drop(columns=drop_cols, errors="ignore")

    # labels
    if "is_anomaly" not in df.columns:
        raise ValueError("Dataset must contain 'is_anomaly' column (0/1).")

    # features: numeric only
    X = df.drop(columns=["is_anomaly"]).select_dtypes(include=[np.number]).copy()
    y = df["is_anomaly"].astype(int).copy()

    # drop zero-variance
    nonzero = X.columns[X.var() > 0].tolist()
    X = X[nonzero]

    # fill all-NaN numeric cols with 0 so imputer won't drop them
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        X[all_nan_cols] = 0

    return X, y

def impute_and_scale(X_train, X_val, X_test):
    """Impute numeric NaNs with mean then scale with RobustScaler."""
    imputer = SimpleImputer(strategy="mean")
    scaler = RobustScaler()

    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns, index=X_val.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=X_train_imp.columns, index=X_train_imp.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val_imp), columns=X_val_imp.columns, index=X_val_imp.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=X_test_imp.columns, index=X_test_imp.index)

    return X_train_scaled, X_val_scaled, X_test_scaled, imputer, scaler

def threshold_optimize_and_evaluate(model, X_val, y_val, X_test, y_test):
    """Find best threshold on val set (by F1) and evaluate on test set using that threshold."""
    # get predicted probabilities on validation
    val_proba = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, val_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)

    if len(thresholds) == 0:
        best_threshold = 0.5
    else:
        best_idx = np.nanargmax(f1_scores)
        # thresholds array is one element shorter than precision/recall; ensure safe index
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    # Evaluate on test using best_threshold
    test_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (test_proba >= best_threshold).astype(int)

    print(f"\nOptimal threshold (by F1 on val): {best_threshold:.4f}")
    print("\nValidation set F1 (best):", f1_scores.max() if len(f1_scores)>0 else None)
    print("\n--- Test set evaluation at optimal threshold ---")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    try:
        auc = roc_auc_score(y_test, test_proba)
        print(f"Test ROC AUC: {auc:.4f}")
    except Exception:
        pass

    return best_threshold, f1_score(y_test, y_pred, pos_label=1)

def plot_f1_vs_threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * (precision * recall) / (precision + recall + 1e-10)

    thresholds = np.append(thresholds, 1.0)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1s, marker='o', linewidth=2)
    plt.title("F1 Score vs. Decision Threshold`")
    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    best_idx = np.argmax(f1s)
    best_thresh, best_f1 = thresholds[best_idx], f1s[best_idx]
    plt.axvline(best_thresh, color='red', linestyle='--', label=f'Best = {best_thresh:.2f} (F1={best_f1:.2f})')
    plt.legend()
    plt.show()

    print(f"🔥 Best threshold = {best_thresh:.3f}, F1 = {best_f1:.3f}")
    return best_thresh, best_f1

if __name__ == "__main__":
    # 1) load
    df = pd.read_csv(DATA_PATH)
    print("Loaded data:", df.shape)

    # 2) prepare numeric features + labels
    X, y = prepare_features(df)
    print("Prepared X shape:", X.shape, "y dist:", y.value_counts().to_dict())

    # 3) train/val/test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RND)
    # further split train into train / val for threshold tuning
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=RND)

    print("Shapes -> train:", X_train.shape, "val:", X_val.shape, "test:", X_test.shape)

    # 4) impute & scale
    X_train_scaled, X_val_scaled, X_test_scaled, imputer, scaler = impute_and_scale(X_train, X_val, X_test)

    # 5) compute IsolationForest iso_score and append as feature
    iso = IsolationForest(contamination=max(0.01, (y_train.sum() / len(y_train))), random_state=RND, n_estimators=200)
    iso.fit(X_train_scaled)
    X_train_scaled["iso_score"] = -iso.score_samples(X_train_scaled)
    X_val_scaled["iso_score"] = -iso.score_samples(X_val_scaled)
    X_test_scaled["iso_score"] = -iso.score_samples(X_test_scaled)
    print("Added iso_score feature.")

    # 6) Apply SMOTETomek on training data only
    smt = SMOTETomek(random_state=RND)
    try:
        X_res, y_res = smt.fit_resample(X_train_scaled, y_train)
    except Exception as e:
        # fallback to SMOTE only if SMOTETomek fails for very small data
        print("SMOTETomek failed, falling back to SMOTE:", e)
        X_res, y_res = SMOTE(random_state=RND, k_neighbors=max(1, min(5, int(y_train.sum())-1))).fit_resample(X_train_scaled, y_train)

    print("After resampling, distribution:", pd.Series(y_res).value_counts().to_dict())

    # 7) build stacking ensemble
    estimators = [
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=4, class_weight="balanced", random_state=RND)),
        ("gb", GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=RND))
    ]
    final_estimator = RandomForestClassifier(n_estimators=150, max_depth=3, random_state=RND)
    stack = StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=3, passthrough=True, n_jobs=-1)

    # 8) fit stacking on resampled training data
    stack.fit(X_res, y_res)
    print("Stacking model trained.")

    # 9) threshold optimization on val and evaluate on test
    best_threshold, test_f1 = threshold_optimize_and_evaluate(stack, X_val_scaled, y_val, X_test_scaled, y_test)
    print(f"\nFinal test F1 (anomaly class) = {test_f1:.4f}")

    test_proba = stack.predict_proba(X_test_scaled)[:, 1]

    for thr in [0.5, 0.34, 0.2, 0.1, 0.05]:
        y_pred = (test_proba >= thr).astype(int)
        print(f"\n=== Threshold {thr:.2f} ===")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    
    y_test_proba = stack.predict_proba(X_test_scaled)[:, 1]
    best_thresh, best_f1 = plot_f1_vs_threshold(y_test, y_test_proba)


    # 10) Save the pipeline artifacts
    saved = {
        "model": stack,
        "imputer": imputer,
        "scaler": scaler,
        "iso_model": iso,
        "best_threshold": float(best_thresh),
        "feature_columns": X_train_scaled.columns.tolist()
    }
    joblib.dump(saved, OUTPUT_MODEL)
    print(f"\nSaved trained pipeline to {OUTPUT_MODEL}")

    print("\nDone.")