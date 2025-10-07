import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

class WasteLabelDefinition:
    def __init__(self):
        self.scaler = StandardScaler()
        self.cluster_models = {}
        
    def load_engineered_data(self, data_path):
        """Load the engineered dataset"""
        print("Loading engineered dataset...")
        self.df = pd.read_csv(data_path)
        print(f"Loaded dataset with shape: {self.df.shape}")
        return self.df
    
    def define_waste_severity_labels(self, df):
        """Define waste severity labels based on multiple factors"""
        print("Defining waste severity labels...")
        
        # Create composite waste score if not already present
        if 'waste_prediction_score' not in df.columns:
            waste_components = []
            weights = []
            
            # Tool wear contribution
            if 'tool_wear' in df.columns:
                normalized_wear = df['tool_wear'] / 160
                waste_components.append(normalized_wear)
                weights.append(0.3)
            
            # Vibration contribution
            if 'vibration_level' in df.columns:
                normalized_vibration = df['vibration_level'] / 2.0
                waste_components.append(normalized_vibration)
                weights.append(0.25)
            
            # Energy inefficiency contribution
            if 'energy_efficiency' in df.columns:
                inefficiency = 1 - df['energy_efficiency']
                waste_components.append(inefficiency)
                weights.append(0.2)
            
            # Process instability contribution
            if 'overall_process_stability' in df.columns:
                instability = 1 - df['overall_process_stability']
                waste_components.append(instability)
                weights.append(0.25)
            
            if waste_components:
                # Calculate weighted waste score
                total_weight = sum(weights)
                weighted_components = [comp * weight for comp, weight in zip(waste_components, weights)]
                df['waste_prediction_score'] = sum(weighted_components) / total_weight
            else:
                # Fallback: use synthetic waste score
                np.random.seed(42)
                df['waste_prediction_score'] = np.random.beta(2, 5, len(df))
        
        # Define waste severity levels
        df['waste_severity'] = pd.cut(df['waste_prediction_score'],
                                    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                    labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        
        # Binary classification label
        df['high_waste_binary'] = (df['waste_prediction_score'] > 0.5).astype(int)
        
        print("Waste severity distribution:")
        print(df['waste_severity'].value_counts().sort_index())
        print(f"High waste cycles: {df['high_waste_binary'].sum()} ({df['high_waste_binary'].mean():.1%})")
        
        return df

    def define_equipment_failure_labels(self, df):
        """Define equipment failure and maintenance labels"""
        print("Defining equipment failure labels...")
        
        failure_indicators = []
        
        # Temperature-based failure indicators
        if 'max_vfd_temperature' in df.columns:
            temp_threshold = df['max_vfd_temperature'].quantile(0.85)
            temp_failure = (df['max_vfd_temperature'] > temp_threshold).astype(int)
            failure_indicators.append(temp_failure)
            print(f"Temperature failures: {temp_failure.sum()}")
        
        # Vibration-based failure indicators
        if 'vibration_level' in df.columns:
            vib_threshold = df['vibration_level'].quantile(0.85)
            vib_failure = (df['vibration_level'] > vib_threshold).astype(int)
            failure_indicators.append(vib_failure)
            print(f"Vibration failures: {vib_failure.sum()}")
        
        # Tool wear failure indicators
        if 'tool_wear' in df.columns:
            wear_threshold = df['tool_wear'].quantile(0.85)
            wear_failure = (df['tool_wear'] > wear_threshold).astype(int)
            failure_indicators.append(wear_failure)
            print(f"Tool wear failures: {wear_failure.sum()}")
        
        # Gripper imbalance failure
        if 'gripper_load_imbalance' in df.columns:
            imbalance_threshold = df['gripper_load_imbalance'].quantile(0.85)
            imbalance_failure = (df['gripper_load_imbalance'] > imbalance_threshold).astype(int)
            failure_indicators.append(imbalance_failure)
            print(f"Gripper imbalance failures: {imbalance_failure.sum()}")
        
        # Composite failure label
        if failure_indicators:
            df['equipment_failure_risk'] = sum(failure_indicators) / len(failure_indicators)
            df['maintenance_required'] = (df['equipment_failure_risk'] > 0.5).astype(int)
            df['maintenance_urgency'] = pd.cut(df['equipment_failure_risk'],
                                             bins=[0, 0.3, 0.6, 1.0],
                                             labels=['low', 'medium', 'high'])
        else:
            # Synthetic failure risk
            np.random.seed(42)
            df['equipment_failure_risk'] = np.random.beta(2, 8, len(df))
            df['maintenance_required'] = (df['equipment_failure_risk'] > 0.3).astype(int)
        
        print(f"Maintenance required cycles: {df['maintenance_required'].sum()} ({df['maintenance_required'].mean():.1%})")
        
        return df

    def define_energy_efficiency_labels(self, df):
        """Define energy efficiency classification labels"""
        print("Defining energy efficiency labels...")
        
        if 'energy_efficiency' not in df.columns:
            # Calculate energy efficiency from available features
            efficiency_components = []
            
            if 'ff_avg_vfd_temperature_mean' in df.columns:
                # Lower temperature = higher efficiency
                temp_efficiency = 1 - (df['ff_avg_vfd_temperature_mean'] / df['ff_avg_vfd_temperature_mean'].max())
                efficiency_components.append(temp_efficiency)
            
            if 'total_kinetic_energy' in df.columns and 'total_heat_generation' in df.columns:
                # Higher work-to-heat ratio = higher efficiency
                work_heat_ratio = df['total_kinetic_energy'] / (df['total_heat_generation'] + 1e-6)
                normalized_ratio = work_heat_ratio / work_heat_ratio.max()
                efficiency_components.append(normalized_ratio)
            
            if efficiency_components:
                df['energy_efficiency'] = sum(efficiency_components) / len(efficiency_components)
            else:
                np.random.seed(42)
                df['energy_efficiency'] = np.random.beta(6, 2, len(df))
        
        # Energy efficiency classification
        df['energy_efficiency_class'] = pd.cut(df['energy_efficiency'],
                                             bins=[0, 0.4, 0.7, 1.0],
                                             labels=['low', 'medium', 'high'])
        
        # Energy waste binary label
        df['energy_waste'] = (df['energy_efficiency'] < 0.5).astype(int)
        
        print("Energy efficiency distribution:")
        print(df['energy_efficiency_class'].value_counts().sort_index())
        print(f"Energy waste cycles: {df['energy_waste'].sum()} ({df['energy_waste'].mean():.1%})")
        
        return df

    def define_process_quality_labels(self, df):
        """Define process quality and stability labels"""
        print("Defining process quality labels...")
        
        if 'overall_process_stability' not in df.columns:
            # Calculate process stability from variability metrics
            stability_components = []
            
            # Speed stability
            speed_cols = [col for col in df.columns if 'Speed' in col and 'std' in col]
            if speed_cols:
                speed_stability = 1 / (1 + df[speed_cols[0]])
                stability_components.append(speed_stability)
            
            # Load stability
            load_cols = [col for col in df.columns if 'Load' in col and 'std' in col]
            if load_cols:
                load_stability = 1 / (1 + df[load_cols[0]])
                stability_components.append(load_stability)
            
            if stability_components:
                df['overall_process_stability'] = sum(stability_components) / len(stability_components)
            else:
                np.random.seed(42)
                df['overall_process_stability'] = np.random.beta(5, 2, len(df))
        
        # Process quality classification
        df['process_quality'] = pd.cut(df['overall_process_stability'],
                                     bins=[0, 0.6, 0.8, 1.0],
                                     labels=['poor', 'acceptable', 'excellent'])
        
        # Quality issues binary label
        df['quality_issue'] = (df['overall_process_stability'] < 0.7).astype(int)
        
        # Process capability labels
        if 'process_capability' in df.columns:
            df['capability_level'] = pd.cut(df['process_capability'],
                                          bins=[0, 1.0, 1.33, 2.0],
                                          labels=['incapable', 'marginally_capable', 'capable'])
        else:
            # Synthetic process capability
            np.random.seed(42)
            df['process_capability'] = np.random.uniform(0.8, 2.0, len(df))
            df['capability_level'] = pd.cut(df['process_capability'],
                                          bins=[0, 1.0, 1.33, 2.0],
                                          labels=['incapable', 'marginally_capable', 'capable'])
        
        print("Process quality distribution:")
        print(df['process_quality'].value_counts().sort_index())
        print(f"Quality issues: {df['quality_issue'].sum()} ({df['quality_issue'].mean():.1%})")
        
        return df

    def define_anomaly_labels(self, df):
        """Define anomaly detection labels using unsupervised learning"""
        print("Defining anomaly labels...")
        
        # Select features for anomaly detection
        feature_cols = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if any(keyword in col for keyword in ['temperature', 'vibration', 'wear', 'load', 'speed']):
                if 'std' in col or 'mean' in col:
                    feature_cols.append(col)
        
        # Limit to top 10 features to avoid dimensionality issues
        feature_cols = feature_cols[:10]
        
        if len(feature_cols) >= 3:
            # Prepare data for anomaly detection
            X = df[feature_cols].fillna(df[feature_cols].mean())
            X_scaled = self.scaler.fit_transform(X)
            
            # Use Isolation Forest for anomaly detection
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(X_scaled)
            
            # Convert to binary
            df['is_anomaly'] = (anomaly_labels == -1).astype(int)
            df['anomaly_score'] = iso_forest.decision_function(X_scaled)
            
            print(f"Detected anomalies: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean():.1%})")
        else:
            # Synthetic anomalies based on extreme values
            np.random.seed(42)
            df['is_anomaly'] = (np.random.random(len(df)) < 0.1).astype(int)
            df['anomaly_score'] = np.random.normal(0, 1, len(df))
            print("Using synthetic anomaly labels")
        
        # Define anomaly severity
        df['anomaly_severity'] = pd.cut(df['anomaly_score'],
                                      bins=[df['anomaly_score'].min(), -0.1, 0.1, df['anomaly_score'].max()],
                                      labels=['severe', 'moderate', 'normal'])
        
        return df

    def define_operational_regimes(self, df):
        """Define operational regime clusters"""
        print("Defining operational regimes...")
        
        # Features for regime clustering
        regime_features = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if any(keyword in col for keyword in ['efficiency', 'stability', 'wear', 'vibration', 'temperature']):
                if 'mean' in col or col in ['energy_efficiency', 'process_stability']:
                    regime_features.append(col)
        
        # Limit to 8 features for clustering
        regime_features = regime_features[:8]
        
        if len(regime_features) >= 3:
            X = df[regime_features].fillna(df[regime_features].mean())
            X_scaled = self.scaler.fit_transform(X)
            
            # Use K-means clustering for operational regimes
            kmeans = KMeans(n_clusters=4, random_state=42)
            df['operational_regime'] = kmeans.fit_predict(X_scaled)
            
            # Name the regimes based on cluster characteristics
            regime_names = {}
            for cluster_id in range(4):
                cluster_data = df[df['operational_regime'] == cluster_id]
                
                # Characterize cluster
                avg_efficiency = cluster_data.get('energy_efficiency', 0.5).mean()
                avg_stability = cluster_data.get('overall_process_stability', 0.5).mean()
                avg_waste = cluster_data.get('waste_prediction_score', 0.5).mean()
                
                if avg_efficiency > 0.7 and avg_stability > 0.7:
                    regime_names[cluster_id] = 'optimal'
                elif avg_waste > 0.6:
                    regime_names[cluster_id] = 'high_waste'
                elif avg_efficiency < 0.4:
                    regime_names[cluster_id] = 'inefficient'
                else:
                    regime_names[cluster_id] = 'normal'
            
            df['operational_regime_name'] = df['operational_regime'].map(regime_names)
            
            print("Operational regime distribution:")
            print(df['operational_regime_name'].value_counts())
        else:
            # Synthetic regimes
            np.random.seed(42)
            df['operational_regime'] = np.random.randint(0, 4, len(df))
            regime_map = {0: 'optimal', 1: 'normal', 2: 'inefficient', 3: 'high_waste'}
            df['operational_regime_name'] = df['operational_regime'].map(regime_map)
            print("Using synthetic operational regimes")
        
        return df

    def define_optimization_targets(self, df):
        """Define optimization targets for waste reduction"""
        print("Defining optimization targets...")
        
        # Overall optimization score
        optimization_components = []
        weights = []
        
        if 'energy_efficiency' in df.columns:
            optimization_components.append(df['energy_efficiency'])
            weights.append(0.3)
        
        if 'overall_process_stability' in df.columns:
            optimization_components.append(df['overall_process_stability'])
            weights.append(0.4)
        
        if 'waste_prediction_score' in df.columns:
            # We want to minimize waste, so use inverse
            optimization_components.append(1 - df['waste_prediction_score'])
            weights.append(0.3)
        
        if optimization_components:
            total_weight = sum(weights)
            weighted_components = [comp * weight for comp, weight in zip(optimization_components, weights)]
            df['optimization_score'] = sum(weighted_components) / total_weight
        else:
            np.random.seed(42)
            df['optimization_score'] = np.random.beta(4, 2, len(df))
        
        # Optimization priority
        improvement_areas = []
        
        if 'energy_efficiency' in df.columns and df['energy_efficiency'].mean() < 0.6:
            improvement_areas.append('energy_efficiency')
        
        if 'overall_process_stability' in df.columns and df['overall_process_stability'].mean() < 0.7:
            improvement_areas.append('process_stability')
        
        if 'waste_prediction_score' in df.columns and df['waste_prediction_score'].mean() > 0.4:
            improvement_areas.append('waste_reduction')
        
        if improvement_areas:
            df['optimization_priority'] = ', '.join(improvement_areas)
        else:
            df['optimization_priority'] = 'balanced_improvement'
        
        # Define optimization classes
        df['optimization_class'] = pd.cut(df['optimization_score'],
                                        bins=[0, 0.5, 0.7, 0.9, 1.0],
                                        labels=['poor', 'fair', 'good', 'excellent'])
        
        print("Optimization target distribution:")
        print(df['optimization_class'].value_counts().sort_index())
        print(f"Primary optimization areas: {df['optimization_priority'].mode().iloc[0]}")
        
        return df

    def define_all_labels(self, data_path):
        """Complete label definition pipeline"""
        print("Starting comprehensive label definition...")
        print("=" * 60)
        
        # Load data
        df = self.load_engineered_data(data_path)
        
        # Apply all label definition steps
        df = self.define_waste_severity_labels(df)
        df = self.define_equipment_failure_labels(df)
        df = self.define_energy_efficiency_labels(df)
        df = self.define_process_quality_labels(df)
        df = self.define_anomaly_labels(df)
        df = self.define_operational_regimes(df)
        df = self.define_optimization_targets(df)
        
        print("=" * 60)
        print("Label definition completed!")
        
        return df

    def generate_label_report(self, df):
        """Generate comprehensive report of defined labels"""
        print("\n" + "=" * 60)
        print("LABEL DEFINITION REPORT")
        print("=" * 60)
        
        label_categories = {
            'Waste Severity Labels': [
                'waste_severity', 'high_waste_binary', 'waste_prediction_score'
            ],
            'Equipment & Maintenance Labels': [
                'equipment_failure_risk', 'maintenance_required', 'maintenance_urgency'
            ],
            'Energy Efficiency Labels': [
                'energy_efficiency_class', 'energy_waste', 'energy_efficiency'
            ],
            'Process Quality Labels': [
                'process_quality', 'quality_issue', 'capability_level', 'process_capability'
            ],
            'Anomaly Detection Labels': [
                'is_anomaly', 'anomaly_score', 'anomaly_severity'
            ],
            'Operational Regimes': [
                'operational_regime', 'operational_regime_name'
            ],
            'Optimization Targets': [
                'optimization_score', 'optimization_class', 'optimization_priority'
            ]
        }
        
        print("\nDefined Label Categories:")
        for category, labels in label_categories.items():
            available_labels = [label for label in labels if label in df.columns]
            if available_labels:
                print(f"  {category}: {len(available_labels)} labels")
                for label in available_labels:
                    if label in df.columns:
                        unique_vals = df[label].nunique() if df[label].dtype == 'object' else 'continuous'
                        print(f"    - {label} ({unique_vals})")
        
        # Label statistics
        print(f"\nLabel Statistics:")
        binary_labels = [col for col in df.columns if df[col].dtype in [np.int64, np.int32] and df[col].nunique() == 2]
        categorical_labels = [col for col in df.columns if df[col].dtype == 'object']
        continuous_labels = [col for col in df.columns if df[col].dtype in [np.float64, np.float32] and col not in binary_labels]
        
        print(f"  Binary labels: {len(binary_labels)}")
        print(f"  Categorical labels: {len(categorical_labels)}")
        print(f"  Continuous labels: {len(continuous_labels)}")
        print(f"  Total labels defined: {len(binary_labels) + len(categorical_labels) + len(continuous_labels)}")
        
        # Show distribution of key binary labels
        print(f"\nKey Binary Label Distributions:")
        for label in binary_labels[:5]:
            if label in df.columns:
                positive_count = df[label].sum()
                positive_pct = df[label].mean() * 100
                print(f"  {label}: {positive_count} positive ({positive_pct:.1f}%)")
        
        return label_categories

# -----------------------------
# Main Execution
# -----------------------------
def main():
    # Initialize label definition
    label_definer = WasteLabelDefinition()
    
    # Path to engineered dataset
    data_path = "/Users/appdev/Downloads/Waste_Reduction/engineered_waste_dataset.csv"
    
    # Define all labels
    labeled_df = label_definer.define_all_labels(data_path)
    
    # Generate report
    label_categories = label_definer.generate_label_report(labeled_df)
    
    # Save labeled dataset
    output_path = "/Users/appdev/Downloads/Waste_Reduction/labeled_waste_dataset.csv"
    labeled_df.to_csv(output_path, index=False)
    print(f"\nLabeled dataset saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS FOR MODEL DEVELOPMENT:")
    print("=" * 60)
    print("1. Use labeled data for supervised learning models")
    print("2. Classification models for:")
    print("   - Waste severity prediction")
    print("   - Maintenance requirement detection")
    print("   - Energy efficiency classification")
    print("3. Regression models for:")
    print("   - Waste score prediction")
    print("   - Optimization score prediction")
    print("4. Anomaly detection for real-time monitoring")
    print("5. Multi-output models for comprehensive waste reduction")

if __name__ == "__main__":
    main()