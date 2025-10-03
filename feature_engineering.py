import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class WasteReductionFeatureEngineer:
    def __init__(self):
        self.scalers = {}
        self.feature_selector = None
        self.pca = None
        self.cluster_model = None
        
    def load_data(self, data_path):
        """Load the merged dataset"""
        print("Loading merged dataset...")
        self.df = pd.read_csv(data_path)
        print(f"Loaded dataset with shape: {self.df.shape}")
        return self.df
    
    def create_temporal_features(self, df):
        """Create time-based features"""
        print("Creating temporal features...")
        
        # If timestamp exists, extract time features
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
                df['shift_type'] = df['hour'].apply(
                    lambda x: 'morning' if 6 <= x < 14 else 'afternoon' if 14 <= x < 22 else 'night'
                )
            except:
                print("Could not parse timestamp, skipping temporal features")
        
        return df

    def create_equipment_health_features(self, df):
        """Create features related to equipment health and wear"""
        print("Creating equipment health features...")
        
        # Motor and drive health indicators
        vfd_temp_cols = [col for col in df.columns if 'VFD' in col and 'Temperature' in col]
        if vfd_temp_cols:
            df['max_vfd_temperature'] = df[vfd_temp_cols].max(axis=1)
            df['vfd_temperature_range'] = df[vfd_temp_cols].max(axis=1) - df[vfd_temp_cols].min(axis=1)
            df['vfd_temperature_anomaly'] = (df[vfd_temp_cols].std(axis=1) > df[vfd_temp_cols].std(axis=1).mean()).astype(int)
        
        # Conveyor system health
        conv_speed_cols = [col for col in df.columns if 'Conv' in col and 'Speed' in col]
        if conv_speed_cols:
            df['conv_speed_variance'] = df[conv_speed_cols].var(axis=1)
            df['conv_speed_imbalance'] = df[conv_speed_cols].std(axis=1)
            df['total_conveyor_power'] = df[conv_speed_cols].sum(axis=1)
        
        # Gripper system health
        gripper_load_cols = [col for col in df.columns if 'Gripper_Load' in col]
        if gripper_load_cols:
            df['gripper_load_asymmetry'] = (df[gripper_load_cols].max(axis=1) - df[gripper_load_cols].min(axis=1))
            df['gripper_health_score'] = 1 - (df['gripper_load_asymmetry'] / df[gripper_load_cols].max(axis=1).max())
            df['gripper_overload_flag'] = (df[gripper_load_cols].max(axis=1) > df[gripper_load_cols].mean().mean()).astype(int)
        
        # Robot joint health
        joint_angle_cols = [col for col in df.columns if 'JointAngle' in col]
        if joint_angle_cols:
            df['joint_angle_variance'] = df[joint_angle_cols].var(axis=1)
            df['robot_stability'] = 1 / (1 + df[joint_angle_variance])
        
        return df

    def create_energy_efficiency_features(self, df):
        """Create energy consumption and efficiency features"""
        print("Creating energy efficiency features...")
        
        # Energy consumption patterns
        temp_cols = [col for col in df.columns if 'Temperature' in col]
        if temp_cols:
            df['total_heat_generation'] = df[temp_cols].sum(axis=1)
            df['heat_dissipation_efficiency'] = 1 / (1 + df['total_heat_generation'])
        
        # Power consumption proxies
        speed_cols = [col for col in df.columns if 'Speed' in col]
        if speed_cols:
            df['total_kinetic_energy'] = df[speed_cols].sum(axis=1)
            df['energy_efficiency_ratio'] = df['total_kinetic_energy'] / (df['total_heat_generation'] + 1e-6)
        
        # Operational efficiency
        load_cols = [col for col in df.columns if 'Load' in col]
        if load_cols and speed_cols:
            df['work_performance_index'] = df[load_cols].mean(axis=1) * df[speed_cols].mean(axis=1)
            df['operational_efficiency'] = df['work_performance_index'] / (df['total_heat_generation'] + 1e-6)
        
        return df

    def create_process_stability_features(self, df):
        """Create features measuring process stability and consistency"""
        print("Creating process stability features...")
        
        # Variability metrics across different sensor groups
        sensor_groups = {
            'temperature': [col for col in df.columns if 'Temperature' in col],
            'speed': [col for col in df.columns if 'Speed' in col],
            'load': [col for col in df.columns if 'Load' in col],
            'angle': [col for col in df.columns if 'Angle' in col]
        }
        
        for group_name, group_cols in sensor_groups.items():
            if group_cols:
                df[f'{group_name}_stability'] = 1 / (1 + df[group_cols].std(axis=1))
                df[f'{group_name}_consistency'] = 1 - (df[group_cols].var(axis=1) / df[group_cols].var(axis=1).max())
        
        # Overall process stability composite
        stability_cols = [col for col in df.columns if 'stability' in col.lower()]
        if stability_cols:
            df['overall_process_stability'] = df[stability_cols].mean(axis=1)
        
        # Process capability indices
        if 'ff_Q_Cell_CycleCount_mean' in df.columns:
            cycle_mean = df['ff_Q_Cell_CycleCount_mean']
            cycle_std = df['ff_Q_Cell_CycleCount_std']
            df['process_capability'] = (cycle_mean.max() - cycle_mean.min()) / (6 * cycle_std + 1e-6)
        
        return df

    def create_waste_prediction_features(self, df):
        """Create features specifically for waste prediction"""
        print("Creating waste prediction features...")
        
        # Tool wear progression
        if 'tool_wear' in df.columns:
            df['tool_wear_rate'] = df['tool_wear'].diff().fillna(0)
            df['cumulative_tool_wear'] = df['tool_wear'].cumsum()
            df['tool_wear_severity'] = pd.cut(df['tool_wear'], 
                                            bins=[0, 30, 70, 120, 160],
                                            labels=[1, 2, 3, 4]).astype(float)
        
        # Vibration analysis
        if 'vibration_level' in df.columns:
            df['vibration_severity'] = np.where(df['vibration_level'] < 0.5, 'low',
                                              np.where(df['vibration_level'] < 1.0, 'medium', 'high'))
            df['vibration_risk_score'] = df['vibration_level'] * df.get('tool_wear', 1)
        
        # Material waste risk composite
        risk_factors = []
        if 'tool_wear' in df.columns:
            risk_factors.append(df['tool_wear'] / 160)  # Normalized tool wear
        if 'vibration_level' in df.columns:
            risk_factors.append(df['vibration_level'] / 2.0)  # Normalized vibration
        if 'gripper_load_imbalance' in df.columns:
            risk_factors.append(df['gripper_load_imbalance'] / 50)  # Normalized imbalance
        
        if risk_factors:
            df['material_waste_risk'] = sum(risk_factors) / len(risk_factors)
            df['waste_risk_category'] = pd.cut(df['material_waste_risk'], 
                                             bins=[0, 0.3, 0.6, 1.0],
                                             labels=['low', 'medium', 'high'])
        
        # Predictive maintenance indicators
        maintenance_indicators = []
        if 'max_vfd_temperature' in df.columns:
            temp_indicator = (df['max_vfd_temperature'] > df['max_vfd_temperature'].quantile(0.8)).astype(int)
            maintenance_indicators.append(temp_indicator)
        
        if 'vibration_level' in df.columns:
            vib_indicator = (df['vibration_level'] > df['vibration_level'].quantile(0.8)).astype(int)
            maintenance_indicators.append(vib_indicator)
        
        if maintenance_indicators:
            df['maintenance_urgency'] = sum(maintenance_indicators) / len(maintenance_indicators)
        
        return df

    def create_interaction_features(self, df):
        """Create interaction features between different sensor groups"""
        print("Creating interaction features...")
        
        # Temperature-Speed interactions
        if 'ff_avg_vfd_temperature_mean' in df.columns and 'ff_M_Conv1_Speed_mmps_mean' in df.columns:
            df['temp_speed_interaction'] = df['ff_avg_vfd_temperature_mean'] * df['ff_M_Conv1_Speed_mmps_mean']
            df['efficiency_temperature_ratio'] = df['ff_M_Conv1_Speed_mmps_mean'] / (df['ff_avg_vfd_temperature_mean'] + 1e-6)
        
        # Load-Energy interactions
        gripper_load_mean = [col for col in df.columns if 'Gripper_Load_mean' in col and 'ff_' in col]
        if gripper_load_mean and 'total_heat_generation' in df.columns:
            df['load_energy_ratio'] = df[gripper_load_mean[0]] / (df['total_heat_generation'] + 1e-6)
        
        # Wear-Vibration interactions
        if 'tool_wear' in df.columns and 'vibration_level' in df.columns:
            df['wear_vibration_product'] = df['tool_wear'] * df['vibration_level']
            df['stability_wear_index'] = df.get('overall_process_stability', 1) / (df['tool_wear'] + 1e-6)
        
        # Cross-equipment correlations
        equipment_pairs = [
            ('VFD', 'Conv'),
            ('Gripper', 'Joint'),
            ('Temperature', 'Load')
        ]
        
        for term1, term2 in equipment_pairs:
            cols1 = [col for col in df.columns if term1 in col and 'mean' in col]
            cols2 = [col for col in df.columns if term2 in col and 'mean' in col]
            if cols1 and cols2:
                df[f'{term1}_{term2}_correlation'] = df[cols1[0]] * df[cols2[0]]
        
        return df

    def create_aggregate_statistics(self, df):
        """Create rolling statistics and aggregate features"""
        print("Creating aggregate statistics...")
        
        # Sort by cycle_id if available
        if 'cycle_id' in df.columns:
            df = df.sort_values('cycle_id').reset_index(drop=True)
            
            # Rolling means for trend analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols[:10]:  
                if col != 'cycle_id':
                    df[f'{col}_rolling_mean_3'] = df[col].rolling(window=3, min_periods=1).mean()
                    df[f'{col}_rolling_std_3'] = df[col].rolling(window=3, min_periods=1).std()
                    df[f'{col}_trend'] = df[col].diff().fillna(0)
        
        # Z-score based anomaly detection
        key_metrics = ['ff_avg_vfd_temperature_mean', 'vibration_level', 'tool_wear']
        for metric in key_metrics:
            if metric in df.columns:
                z_score = (df[metric] - df[metric].mean()) / (df[metric].std() + 1e-6)
                df[f'{metric}_anomaly'] = (abs(z_score) > 2).astype(int)
        
        return df

    def create_dimensionality_reduction_features(self, df):
        """Create features using dimensionality reduction techniques"""
        print("Creating dimensionality reduction features...")
        
        # Select numeric columns for PCA
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove highly correlated columns and target variables
        exclude_keywords = ['anomaly', 'risk', 'severity', 'category', 'flag', 'predicted', 'efficiency', 'stability']
        feature_cols = [col for col in numeric_cols if not any(keyword in col.lower() for keyword in exclude_keywords)]
        
        if len(feature_cols) > 5:  
            # Handle missing values
            X = df[feature_cols].fillna(df[feature_cols].mean())
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply PCA
            self.pca = PCA(n_components=min(5, len(feature_cols)))
            pca_features = self.pca.fit_transform(X_scaled)
            
            for i in range(pca_features.shape[1]):
                df[f'pca_component_{i+1}'] = pca_features[:, i]
            
            print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_}")
        
        # Create equipment clusters
        equipment_features = [col for col in feature_cols if any(term in col for term in ['VFD', 'Conv', 'Gripper', 'Joint'])]
        if len(equipment_features) > 3:
            X_equip = df[equipment_features].fillna(df[equipment_features].mean())
            self.cluster_model = KMeans(n_clusters=3, random_state=42)
            df['equipment_cluster'] = self.cluster_model.fit_predict(X_equip)
        
        return df

    def perform_feature_selection(self, df, target_column='predicted_material_waste'):
        """Select most important features for waste prediction"""
        print("Performing feature selection...")
        
        if target_column not in df.columns:
            print(f"Target column {target_column} not found. Using first numeric column.")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            target_column = numeric_cols[0] if len(numeric_cols) > 0 else None
        
        if target_column:
            # Prepare features and target
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in feature_cols if col != target_column]
            
            X = df[feature_cols].fillna(df[feature_cols].mean())
            y = df[target_column]
            
            # Use mutual information for feature selection
            self.feature_selector = SelectKBest(score_func=mutual_info_regression, k=min(20, len(feature_cols)))
            X_selected = self.feature_selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_mask = self.feature_selector.get_support()
            selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selected_mask[i]]
            
            print(f"Selected top {len(selected_features)} features:")
            for i, feature in enumerate(selected_features[:10]):
                print(f"  {i+1}. {feature}")
            
            # Add selection indicator to dataframe
            df['is_selected_feature'] = True  
            
            return df, selected_features
        else:
            return df, []

    def create_final_waste_features(self, df):
        """Create final composite waste prediction features"""
        print("Creating final waste prediction features...")
        
        # Waste prediction score 
        waste_factors = []
        weights = []
        
        if 'material_waste_risk' in df.columns:
            waste_factors.append(df['material_waste_risk'])
            weights.append(0.4)
        
        if 'tool_wear_severity' in df.columns:
            waste_factors.append(df['tool_wear_severity'] / 4.0)  
            weights.append(0.3)
        
        if 'vibration_risk_score' in df.columns:
            waste_factors.append(df['vibration_risk_score'] / 2.0)  
            weights.append(0.2)
        
        if 'maintenance_urgency' in df.columns:
            waste_factors.append(df['maintenance_urgency'])
            weights.append(0.1)
        
        if waste_factors:
            # Weighted average
            total_weight = sum(weights)
            weighted_factors = [factor * weight for factor, weight in zip(waste_factors, weights)]
            df['waste_prediction_score'] = sum(weighted_factors) / total_weight
            
            # Classification based on score
            df['waste_risk_level'] = pd.cut(df['waste_prediction_score'],
                                          bins=[0, 0.3, 0.6, 1.0],
                                          labels=['low', 'medium', 'high'])
        
        # Optimization potential features
        if 'energy_efficiency' in df.columns and 'process_stability' in df.columns:
            df['optimization_potential'] = 1 - (df['energy_efficiency'] * df['process_stability'])
            df['improvement_priority'] = pd.cut(df['optimization_potential'],
                                              bins=[0, 0.2, 0.5, 1.0],
                                              labels=['low', 'medium', 'high'])
        
        return df

    def engineer_all_features(self, data_path):
        """Complete feature engineering pipeline"""
        print("Starting comprehensive feature engineering...")
        print("=" * 60)
        
        # Load data
        df = self.load_data(data_path)
        
        # Apply all feature engineering steps
        df = self.create_temporal_features(df)
        df = self.create_equipment_health_features(df)
        df = self.create_energy_efficiency_features(df)
        df = self.create_process_stability_features(df)
        df = self.create_waste_prediction_features(df)
        df = self.create_interaction_features(df)
        df = self.create_aggregate_statistics(df)
        df = self.create_dimensionality_reduction_features(df)
        df = self.create_final_waste_features(df)
        
        # Perform feature selection
        df, selected_features = self.perform_feature_selection(df)
        
        print("=" * 60)
        print("Feature engineering completed!")
        print(f"Original features: {len(self.df.columns)}")
        print(f"Engineered features: {len(df.columns)}")
        print(f"Total features after engineering: {len(df.columns)}")
        
        return df, selected_features

    def generate_feature_report(self, df, selected_features):
        """Generate a comprehensive report of engineered features"""
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING REPORT")
        print("=" * 60)
        
        # Feature categories
        categories = {
            'Temporal Features': [col for col in df.columns if any(x in col for x in ['hour', 'day', 'shift', 'weekend'])],
            'Equipment Health': [col for col in df.columns if any(x in col for x in ['health', 'wear', 'maintenance', 'anomaly', 'stability'])],
            'Energy Efficiency': [col for col in df.columns if any(x in col for x in ['energy', 'efficiency', 'power', 'heat'])],
            'Process Stability': [col for col in df.columns if any(x in col for x in ['stability', 'consistency', 'capability', 'variance'])],
            'Waste Prediction': [col for col in df.columns if any(x in col for x in ['waste', 'risk', 'severity', 'prediction'])],
            'Interaction Features': [col for col in df.columns if any(x in col for x in ['interaction', 'ratio', 'product', 'correlation'])],
            'Statistical Features': [col for col in df.columns if any(x in col for x in ['rolling', 'trend', 'zscore'])],
            'Dimensionality Reduction': [col for col in df.columns if any(x in col for x in ['pca', 'cluster'])]
        }
        
        print("\nFeature Categories:")
        for category, features in categories.items():
            if features:
                print(f"  {category}: {len(features)} features")
        
        print(f"\nSelected Top Features for Waste Prediction: {len(selected_features)}")
        for i, feature in enumerate(selected_features[:15]):
            print(f"  {i+1:2d}. {feature}")
        
        # Data quality summary
        print(f"\nData Quality Summary:")
        print(f"  Total records: {len(df):,}")
        print(f"  Total features: {len(df.columns):,}")
        print(f"  Missing values: {df.isnull().sum().sum():,}")
        print(f"  Data completeness: {df.notnull().mean().mean():.1%}")
        
        return categories

# -----------------------------
# Main Execution
# -----------------------------
def main():
    # Initialize feature engineer
    feature_engineer = WasteReductionFeatureEngineer()
    
    # Path to the merged dataset
    data_path = "/Users/appdev/Downloads/Waste_Reduction/improved_merged_production_dataset.csv"
    
    # Perform feature engineering
    engineered_df, selected_features = feature_engineer.engineer_all_features(data_path)
    
    # Generate report
    categories = feature_engineer.generate_feature_report(engineered_df, selected_features)
    
    # Save engineered dataset
    output_path = "/Users/appdev/Downloads/Waste_Reduction/engineered_waste_dataset.csv"
    engineered_df.to_csv(output_path, index=False)
    print(f"\nEngineered dataset saved to: {output_path}")
    
    # Save selected features
    selected_features_path = "/Users/appdev/Downloads/Waste_Reduction/selected_features.txt"
    with open(selected_features_path, 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    print(f"Selected features saved to: {selected_features_path}")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS FOR WASTE REDUCTION AI:")
    print("=" * 60)
    print("1. Use engineered features for machine learning models")
    print("2. Train predictive models for waste classification")
    print("3. Implement optimization algorithms for process improvement")
    print("4. Set up real-time monitoring with the selected features")
    print("5. Create dashboards for waste reduction insights")

if __name__ == "__main__":
    main()