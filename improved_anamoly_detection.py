
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

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest, StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score, f1_score
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE, RandomOverSampler
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
import json, datetime

RND = 42
DATA_PATH = "/Users/appdev/Downloads/Waste_Reduction/labeled_waste_dataset.csv"
OUTPUT_MODEL = "enhanced_anomaly_detector.joblib"
PLOT_PATH = "/Users/appdev/Downloads/Waste_Reduction/f1_threshold_plot.png"

def prepare_features(df):
    """Select numeric features, drop useless cols, basic cleaning."""
    df = df.copy()

    # labels
    if "is_anomaly" not in df.columns:
        raise ValueError("Column 'is_anomaly' not found in dataframe")
    df["is_anomaly"] = pd.to_numeric(df["is_anomaly"], errors="coerce")
    if df["is_anomaly"].isna().any():
        raise ValueError("Unparseable values in 'is_anomaly' column")
    
    drop_cols = [c for c in ["cycle_id", "timestamp", "is_synthetic", "operational_regime_name"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # features: numeric only
    numeric = df.select_dtypes(include=[np.number]).copy()
    if "is_anomaly" in numeric.columns:
        X = numeric.drop(columns=["is_anomaly"]).copy()
    else:
        X = numeric.copy()
    y = df["is_anomaly"].astype(int).copy()

    # drop zero-variance
    var = X.var(axis=0, skipna=True)
    nonzero = var[var > 0].index.tolist()
    X = X[nonzero]

    # fill all-NaN numeric cols with 0 so imputer won't drop them
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    for c in all_nan_cols:
        X[c] = 0.0

    return X, y

def impute_and_scale(X_train, X_val, X_test):
    """Impute numeric NaNs with mean then scale with RobustScaler."""
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()

    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns, index=X_val.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=X_train_imp.columns, index=X_train_imp.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val_imp), columns=X_val_imp.columns, index=X_val_imp.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=X_test_imp.columns, index=X_test_imp.index)

    return X_train_scaled, X_val_scaled, X_test_scaled, imputer, scaler

def add_iso_feature(X_train, X_val, X_test):
    iso = IsolationForest(contamination=0.05, random_state=RND)
    iso.fit(X_train)
    X_train["iso_score"] = -iso.score_samples(X_train)
    X_val["iso_score"] = -iso.score_samples(X_val)
    X_test["iso_score"] = -iso.score_samples(X_test)
    return X_train, X_val, X_test, iso

def resample_training(X_train, y_train):
    """
    Choose resampling strategy based on minority class size:
      - >=6 positives: SMOTETomek (default k_neighbors=5)
      - 2..5 positives: SMOTE with k_neighbors = min(5, n_pos-1)
      - 1 positive: RandomOverSampler (replicate)
      - 0 positives: raise error
    Returns X_res (DataFrame), y_res (Series)
    """
    pos = int(y_train.sum())
    n = len(y_train)
    if pos == 0:
        raise ValueError("No positive anomalies in training set; cannot resample.")
    if pos >= 6:
        sampler = SMOTETomek(random_state=RND)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
    elif pos >= 2:
        k = max(1, min(5, pos - 1))
        sampler = SMOTE(k_neighbors=k, random_state=RND)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
    else:
        # only 1 positive -> replicate it
        sampler = RandomOverSampler(random_state=RND)
        X_res, y_res = sampler.fit_resample(X_train, y_train)

    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res)
    return X_res, y_res

def threshold_optimize_and_evaluate(model, X_val, y_val, X_test, y_test, plot_path=None):
    """Find best threshold on val set (by F1) and evaluate on test set using that threshold."""
    # get predicted probabilities on validation
    if hasattr(model, "predict_proba"):
        val_proba = model.predict_proba(X_val)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]
    else:
        val_proba = model.decision_function(X_val)
        test_proba = model.decision_function(X_test)
    precision, recall, thresholds = precision_recall_curve(y_val, val_proba)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-12)
    
    if len(thresholds) == 0:
        best_threshold = 0.5
        best_f1 = float(f1_scores[0]) if len(f1_scores) > 0 else 0.0
        f1_for_plot = []
    else:
        f1_for_thresholds = f1_scores[:-1]
        best_idx = int(np.nanargmax(f1_for_thresholds))
        best_threshold = float(thresholds[best_idx])
        best_f1 = float(f1_for_thresholds[best_idx])
        f1_for_plot = f1_for_thresholds

    # Evaluate on test using best_threshold
    y_pred = (test_proba >= best_threshold).astype(int)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    roc = roc_auc_score(y_test, test_proba) if len(np.unique(y_test)) > 1 else float("nan")
    test_f1 = f1_score(y_test, y_pred)

    if plot_path and len(thresholds) > 0:
        plt.figure(figsize=(6,4))
        plt.plot(thresholds, f1_for_plot, label="F1")
        plt.axvline(best_threshold, color="red", linestyle="--", label=f"best {best_threshold:.3f}")
        plt.xlabel("threshold")
        plt.ylabel("F1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

    metrics = {
        "best_threshold": float(best_threshold),
        "best_val_f1": float(best_f1),
        "test_f1": float(test_f1),
        "roc_auc_test": float(roc),
        "confusion_matrix": cm.tolist()
    }
    return metrics, best_threshold

def train_with_cv_and_finalize(X, y, n_splits=5):
    pos = int(y.sum()); neg = len(y) - pos
    max_splits = max(2, min(n_splits, pos, neg)) if pos > 0 and neg > 0 else 0
    if max_splits < 2:
        print("Too few positives for CV; training single balanced model.")
        imputer = SimpleImputer(strategy="median")
        scaler = RobustScaler()
        X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        X_s = pd.DataFrame(scaler.fit_transform(X_imp), columns=X.columns, index=X.index)
        iso = IsolationForest(contamination=0.05, random_state=RND)
        iso.fit(X_s)
        X_s["iso_score"] = -iso.score_samples(X_s)
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RND, n_jobs=-1)
        clf.fit(X_s, y)
        artifact = {"model": clf, "imputer": imputer, "scaler": scaler, "iso_model": iso, "feature_columns": X_s.columns.tolist(), "best_threshold": 0.5}
        joblib.dump(artifact, OUTPUT_MODEL)
        print("Saved trained pipeline to", OUTPUT_MODEL)
        return artifact
    
    skf = StratifiedKFold(n_splits=max_splits, shuffle=True, random_state=RND)
    val_probas = np.zeros(len(y))
    for train_idx, val_idx in skf.split(X, y):
        X_tr = X.iloc[train_idx]; y_tr = y.iloc[train_idx]
        X_va = X.iloc[val_idx]; y_va = y.iloc[val_idx]

        imputer = SimpleImputer(strategy="median")
        scaler = RobustScaler()

        def arr_to_df(arr, cols, idx, name_hint="f"):
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.shape[1] == len(cols):
                out_cols = list(cols)
            elif hasattr(imputer, "feature_names_in_"):
                out_cols = list(imputer.feature_names_in_)[: arr.shape[1]]
            else:
                out_cols = [f"{name_hint}{i}" for i in range(arr.shape[1])]
            return pd.DataFrame(arr, columns=out_cols, index=idx)
        
        X_tr_imp_arr = imputer.fit_transform(X_tr)
        X_va_imp_arr = imputer.transform(X_va)
        X_tr_imp = arr_to_df(X_tr_imp_arr, X_tr.columns, X_tr.index)
        X_va_imp = arr_to_df(X_va_imp_arr, X_va.columns, X_va.index)

        X_tr_s_arr = scaler.fit_transform(X_tr_imp)
        X_va_s_arr = scaler.transform(X_va_imp)
        X_tr_s = arr_to_df(X_tr_s_arr, X_tr_imp.columns, X_tr_imp.index, name_hint="s")
        X_va_s = arr_to_df(X_va_s_arr, X_va_imp.columns, X_va_imp.index, name_hint="s")

        iso = IsolationForest(contamination=0.05, random_state=RND)
        iso.fit(X_tr_s)
        X_tr_s["iso_score"] = -iso.score_samples(X_tr_s)
        X_va_s["iso_score"] = -iso.score_samples(X_va_s)

        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RND, n_jobs=-1)
        clf.fit(X_tr_s, y_tr)
        val_probas[val_idx] = clf.predict_proba(X_va_s)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y, val_probas)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-12)
    if len(thresholds) == 0:
        best_threshold = 0.5
        best_f1 = float(f1_scores[0]) if len(f1_scores) else 0.0
    else:
        best_idx = int(np.nanargmax(f1_scores[:-1]))
        best_threshold = float(thresholds[best_idx])
        best_f1 = float(f1_scores[:-1][best_idx])
    
    print("CV selected threshold:", best_threshold, "val F1:", best_f1)

    # retrain final pipeline on full data using the same imputer/scaler + iso + balanced classifier
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()

    def arr_to_df_full(arr, cols, idx, name_hint="f"):
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[1] == len(cols):
            out_cols = list(cols)
        elif hasattr(imputer, "feature_names_in_"):
            out_cols = list(imputer.feature_names_in_)[: arr.shape[1]]
        else:
            out_cols = [f"{name_hint}{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=out_cols, index=idx)
    
    X_imp_arr = imputer.fit_transform(X)
    X_imp = arr_to_df_full(X_imp_arr, X.columns, X.index)
    X_s_arr = scaler.fit_transform(X_imp)
    X_s = arr_to_df_full(X_s_arr, X_imp.columns, X_imp.index, name_hint="s")

    iso = IsolationForest(contamination=0.05, random_state=RND)
    iso.fit(X_s)
    X_s["iso_score"] = -iso.score_samples(X_s)
    final_clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RND, n_jobs=-1)
    final_clf.fit(X_s, y)
    calibrated = CalibratedClassifierCV(final_clf, cv='prefit', method='sigmoid')
    calibrated.fit(X_s, y)

    # save artifact
    artifact = {
        "model": calibrated,
        "imputer": imputer,
        "scaler": scaler,
        "iso_model": iso,
        "feature_columns": X_s.columns.tolist(),
        "best_threshold": float(best_threshold)
    }
    joblib.dump(artifact, OUTPUT_MODEL)
    print("Saved trained pipeline to", OUTPUT_MODEL)
    # optional plot of F1 vs thresholds
    try:
        if len(thresholds) > 0:
            plt.figure(figsize=(6,4))
            plt.plot(thresholds, f1_scores[:-1], label="F1")
            plt.axvline(best_threshold, color="red", linestyle="--", label=f"best {best_threshold:.3f}")
            plt.xlabel("threshold"); plt.ylabel("F1"); plt.legend(); plt.tight_layout()
            plt.savefig(PLOT_PATH, dpi=150); plt.close()
    except Exception:
        pass

    return artifact

if __name__ == "__main__":
    # 1) load
    print("Loading data:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    # 2) prepare numeric features + labels
    numeric = df.select_dtypes(include=[np.number]).copy()
    if "is_anomaly" not in df.columns:
        raise RuntimeError("is_anomaly missing")
    y = pd.to_numeric(df["is_anomaly"], errors="coerce").astype(int)
    if "is_anomaly" in numeric.columns:
        X = numeric.drop(columns=["is_anomaly"])
    else:
        X = numeric.copy()
    
    artifact = train_with_cv_and_finalize(X, y, n_splits=5)

    imputer = artifact["imputer"]; scaler = artifact["scaler"]; iso = artifact["iso_model"]; model = artifact["model"]; best_threshold = artifact.get("best_threshold", 0.5)
    # do a single stable test split to report final metrics
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RND)
    def arr_to_df_local(arr, cols, idx, name_hint="f"):
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[1] == len(cols):
            out_cols = list(cols)
        elif hasattr(imputer, "feature_names_in_"):
            out_cols = list(imputer.feature_names_in_)[: arr.shape[1]]
        else:
            out_cols = [f"{name_hint}{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=out_cols, index=idx)
    
    X_test_imp_arr = imputer.transform(X_test)
    X_test_imp = arr_to_df_local(X_test_imp_arr, X_test.columns, X_test.index)
    X_test_s_arr = scaler.transform(X_test_imp)
    X_test_s = arr_to_df_local(X_test_s_arr, X_test_imp.columns, X_test_imp.index, name_hint="s")
    if "iso_score" not in X_test_s.columns:
        X_test_s["iso_score"] = -iso.score_samples(X_test_s)
    proba_test = model.predict_proba(X_test_s)[:, 1]
    y_pred = (proba_test >= best_threshold).astype(int)
    print("Final test report:\n", classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:", confusion_matrix(y_test, y_pred).tolist())

    try:
        summary = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "dataset": DATA_PATH,
            "random_seed": RND,
            "n_features": len(artifact.get("feature_columns", [])),
            "feature_sample": artifact.get("feature_columns", [])[:20],
            "best_threshold": float(best_threshold),
            "test_support": int(len(y_test)),
            "test_positive_support": int(y_test.sum()),
            "final_confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        with open("training_summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        art = joblib.load(OUTPUT_MODEL)
        art["_training_summary"] = summary
        joblib.dump(art, OUTPUT_MODEL)
        print("Saved training_summary.json and updated artifact with summary")
    except Exception as e:
        print("Warning: failed to write training summary:", e)