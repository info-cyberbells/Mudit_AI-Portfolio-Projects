import joblib, numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split

MODEL_PATH = "enhanced_anomaly_detector.joblib"
DATA_PATH = "labeled_waste_dataset.csv"

art = joblib.load(MODEL_PATH)
model = art["model"]
imputer = art["imputer"]
scaler = art["scaler"]
feature_cols = art["feature_columns"]
best_thr = art.get("best_threshold", 0.5)

df = pd.read_csv(DATA_PATH)
# prepare X,y same as train script

y = pd.to_numeric(df["is_anomaly"], errors="coerce").astype(int)

numeric = df.select_dtypes(include=[np.number]).copy()

if hasattr(imputer, "feature_names_in_"):
    imputer_cols = list(imputer.feature_names_in_)
else:
    # fallback: intersection between saved feature columns and current numeric columns
    imputer_cols = [c for c in feature_cols if c in numeric.columns]
    if not imputer_cols:
        imputer_cols = list(numeric.columns)

# ensure ordering and existence
imputer_cols = [c for c in imputer_cols if c in numeric.columns]
if not imputer_cols:
    raise RuntimeError("No numeric columns available for imputer transform")

# same impute+scale pipeline
X_for_impute = numeric[imputer_cols]
X_imp = pd.DataFrame(imputer.transform(X_for_impute), columns=imputer_cols, index=X_for_impute.index)
X_s = pd.DataFrame(scaler.transform(X_imp), columns=imputer_cols, index=X_imp.index)

# iso model if present
iso = art.get("iso_model", None)
if iso is not None:
    X_s["iso_score"] = -iso.score_samples(X_s)

available = [c for c in feature_cols if c in X_s.columns]
missing = [c for c in feature_cols if c not in X_s.columns]
if missing:
    print("Warning: some trained feature_columns missing in data, will use intersection:", missing)

X = X_s[available].copy()
# reproduce splits used in training for quick check
X_tmp, X_test, y_tmp, y_test = train_test_split(X_s, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=42)

def summarize(probas, name):
    print(f"--- {name} probas ---")
    print("min/max/mean:", np.min(probas), np.max(probas), np.mean(probas))
    for t in [0.5, best_thr, 0.3, 0.1]:
        preds = (probas >= t).astype(int)
        print(f"threshold {t}: positives={int(preds.sum())}, f1={f1_score(y_test, preds):.3f} (on test)")

# val probs
val_proba = model.predict_proba(X_val[available])[:,1] if hasattr(model, "predict_proba") else model.decision_function(X_val[available])
test_proba = model.predict_proba(X_test[available])[:,1] if hasattr(model, "predict_proba") else model.decision_function(X_test[available])

print("y counts -> train/val/test:", int(y_train.sum()), int(y_val.sum()), int(y_test.sum()))
summarize(val_proba, "VAL")
summarize(test_proba, "TEST")

# show small table of lowest/highest test probs with true label
tidx = np.argsort(test_proba)
print("\nlowest test prob rows (prob,label):")
for i in tidx[:10]:
    print(round(test_proba[i],4), int(y_test.iloc[i]))
print("\nhighest test prob rows (prob,label):")
for i in tidx[-10:]:
    print(round(test_proba[i],4), int(y_test.iloc[i]))

prec, rec, thr = precision_recall_curve(y_val, val_proba)
print("\nprecision/recall lengths:", len(prec), len(rec), len(thr))
print("best threshold used:", best_thr)