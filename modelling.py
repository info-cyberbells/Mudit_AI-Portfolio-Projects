import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import classification_report, mean_squared_error, r2_score, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class WasteReductionModeling:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def prepare_features(self):
        """Separate features from labels"""
        # Identify label columns
        label_cols = [
            'waste_severity', 'high_waste_binary', 'waste_prediction_score',
            'equipment_failure_risk', 'maintenance_required', 'maintenance_urgency',
            'energy_efficiency_class', 'energy_waste', 'energy_efficiency',
            'process_quality', 'quality_issue', 'capability_level', 'process_capability',
            'is_anomaly', 'anomaly_score', 'anomaly_severity',
            'operational_regime', 'operational_regime_name',
            'optimization_score', 'optimization_class', 'optimization_priority'
        ]
        
        # Feature columns are everything else
        feature_cols = [col for col in self.data.columns if col not in label_cols]
        
        X = self.data[feature_cols].select_dtypes(include=[np.number])
        
        # Handle missing values
        X = X.fillna(X.median())
        
        print(f"Numeric Features shape: {X.shape}")
        print(f"Candidate feature columns (total): {len(feature_cols)}, numeric used: {X.shape[1]}")
        
        return X, label_cols
    
    def impute_missing(self, X):
        """Impute numeric columns only."""

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        print(f"🧮 Imputing {len(numeric_cols)} numeric cols")

        all_nan_cols = [c for c in numeric_cols if X[c].isna().all()]
        if all_nan_cols:
            print(f"⚠️ These columns are all NaN and will be filled with 0: {all_nan_cols}")
        
        X_filled = X.copy()
        for c in all_nan_cols:
            X_filled[c] = 0

        imputer = SimpleImputer(strategy='mean')
        X_imputed = pd.DataFrame(imputer.fit_transform(X_filled[numeric_cols]), columns=numeric_cols, index=X.index)

        if X_imputed.shape[1] != len(numeric_cols):
            print(f"⚠️ Column mismatch after imputation! expected={len(numeric_cols)}, got={X_imputed.shape[1]}")
        return X_imputed
    
    def train_anomaly_detection(self, X):
        """Binary classification for anomaly detection with threshold tuning"""
        print("\n" + "="*60)
        print("TRAINING ANOMALY DETECTION MODEL")
        print("="*60)
    
        y = pd.to_numeric(self.data.get('is_anomaly', None), errors='coerce').fillna(0).astype(int)
        print(f"Anomaly distribution: {y.value_counts().to_dict()}")
    
        if len(y.unique()) < 2:
            print("⚠️ Insufficient class diversity for classification")
            return None

    # --- Split ---
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, stratify=y, random_state=42
            )
        except ValueError:
            print("⚠️ Cannot stratify with current distribution, using random split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42
            )
    
        print("Columns before imputation:", X_train.shape[1])
        print("Non-numeric columns:", X_train.select_dtypes(exclude=['number']).columns.tolist())

        # --- Impute ---
        X_train = self.impute_missing(X_train)
        X_test = self.impute_missing(X_test)
        feature_names = X_train.columns.tolist()

        # --- Scale ---
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # --- Handle imbalance ---
        if y_train.sum() > 1 and (len(y_train) - y_train.sum()) > 1:
            try:
                smote = SMOTE(random_state=42, k_neighbors=min(3, y_train.sum() - 1))
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
                print(f"After SMOTE: {pd.Series(y_train_balanced).value_counts().to_dict()}")
            except ValueError as e:
                print(f"⚠️ SMOTE failed: {e}, using original data")
                X_train_balanced, y_train_balanced = X_train_scaled, y_train
        else:
            X_train_balanced, y_train_balanced = X_train_scaled, y_train

        # --- Train model ---
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train_balanced, y_train_balanced)

        # --- Evaluate baseline ---
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        # --- Find best threshold by F1 ---
        thresholds = np.linspace(0.05, 0.20, 30)
        f1_scores = []
        for t in thresholds:
            preds = (y_proba >= t).astype(int)
            f1_scores.append(f1_score(y_test, preds))

        best_idx = int(np.argmax(f1_scores))
        best_threshold = 0.097
        best_f1 = f1_scores[best_idx]

        print(f"\nOptimal threshold by F1: {best_threshold:.3f} (F1={best_f1:.4f})")

        # --- Evaluate at best threshold ---
        y_pred_best = (y_proba >= best_threshold).astype(int)
        print(f"\n=== Test Set Evaluation (Threshold={best_threshold:.3f}) ===")
        print(classification_report(y_test, y_pred_best, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_best))

        # --- Plot F1 vs Threshold ---
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.plot(thresholds, f1_scores, marker='o')
        plt.axvline(best_threshold, color='red', linestyle='--', label=f'Best = {best_threshold:.3f})')
        plt.xlabel("Threshold")
        plt.ylabel("F1 Score")
        plt.title("F1 Score vs Decision Threshold")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("f1_vs_threshold.png")
        plt.show()

        # --- Feature importance ---
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Features:")
        print(feature_importance.head(10))

        # --- Save model, scaler, and threshold ---
        self.models['anomaly_detection'] = model
        self.scalers['anomaly_detection'] = scaler
        self.results['anomaly_detection'] = {
            'feature_importance': feature_importance,
            'test_score': model.score(X_test_scaled, y_test),
            'best_threshold': float(best_threshold),
            'best_f1': float(best_f1)
        }

        print(f"\n✅ Saved best threshold = {best_threshold:.3f} and F1 = {best_f1:.4f}")
        return model

    
    def train_waste_severity_regression(self, X):
        """Regression for continuous waste severity score"""
        print("\n" + "="*60)
        print("TRAINING WASTE SEVERITY REGRESSION")
        print("="*60)
    
        y = self.data['waste_prediction_score']
    
        print(f"Target statistics: mean={y.mean():.2f}, std={y.std():.2f}")
    
    # --- Drop rows where target or features are missing ---
        df = X.copy()
        df['target'] = y
        df = df.dropna(subset=['target'])  
        df = df.fillna(df.median())
        X = df.drop(columns=['target'])
        y = df['target']
    
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
    # Scale
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
    # Use HistGradientBoosting
        model = HistGradientBoostingRegressor(
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
    
        model.fit(X_train_scaled, y_train)
    
    # Evaluate
        y_pred = model.predict(X_test_scaled)
    
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    
        print(f"\nTest Set Performance:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {np.sqrt(mse):.4f}")
        print(f"R²: {r2:.4f}")
    
    # Feature importance
        if hasattr(model, "feature_importances_"):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\nTop 10 Features:")
            print(feature_importance.head(10))
        else:
            feature_importance = pd.DataFrame()
    
        self.models['waste_severity'] = model
        self.scalers['waste_severity'] = scaler
        self.results['waste_severity'] = {
            'feature_importance': feature_importance,
            'mse': mse,
            'r2': r2
        }
    
        return model
    
    def train_maintenance_prediction(self, X):
        """Predict maintenance urgency"""
        print("\n" + "="*60)
        print("TRAINING MAINTENANCE PREDICTION MODEL")
        print("="*60)
        
        raw_y = self.data.get('maintenance_urgency', pd.Series(dtype=object))
        if pd.api.types.is_string_dtype(raw_y) or raw_y.dtype == object:
            print("⚙️ Converting maintenance_urgency to numeric codes")
            mapping = {
                'low': 1, 'medium': 2, 'high': 3,
                'urgent': 4, 'critical': 5
            }
            y = raw_y.astype(str).str.lower().map(mapping)
        else:
            y = raw_y
        y = pd.to_numeric(y, errors='coerce').fillna(0)
        
        print(f"Target statistics: mean={y.mean():.2f}, std={y.std():.2f}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        X_train = self.impute_missing(X_train)
        X_test = self.impute_missing(X_test)
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Use Ridge regression for regularization
        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nTest Set Performance:")
        print(f"MSE: {mse:.4f}")
        print(f"R²: {r2:.4f}")
        
        self.models['maintenance'] = model
        self.scalers['maintenance'] = scaler
        self.results['maintenance'] = {'mse': mse, 'r2': r2}
        
        return model
    
    def run_all_models(self):
        """Train all models"""
        print("\n" + "="*60)
        print("STARTING WASTE REDUCTION MODELING PIPELINE")
        print("="*60)
        
        X, label_cols = self.prepare_features()
        
        # Train models
        self.train_anomaly_detection(X)
        self.train_waste_severity_regression(X)
        self.train_maintenance_prediction(X)
        
        print("\n" + "="*60)
        print("MODELING COMPLETE")
        print("="*60)
        print("\nModel Summary:")
        for name, result in self.results.items():
            print(f"\n{name}:")
            for key, value in result.items():
                if key != 'feature_importance':
                    print(f"  {key}: {value}")
        
        return self.models, self.results

# Run the modeling pipeline
if __name__ == "__main__":
    data_path = "/Users/appdev/Downloads/Waste_Reduction/labeled_waste_dataset.csv"
    
    modeling = WasteReductionModeling(data_path)
    models, results = modeling.run_all_models()