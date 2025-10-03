import pandas as pd
import numpy as np
import os
import h5py
import ijson
from decimal import Decimal
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Utility function to handle decimal conversion
# -----------------------------
def convert_decimal_to_float(obj):
    """Recursively convert Decimal objects to float for pandas compatibility"""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_decimal_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimal_to_float(item) for item in obj]
    else:
        return obj

# -----------------------------
# Fixed Future Factories Loader - Better Cycle Detection
# -----------------------------
def load_future_factories_enhanced(base_dir, limit=5000):
    json_path = os.path.join(base_dir, "data_31000.json")
    records = []

    print("Loading Future Factories data...")
    with open(json_path, "r") as f:
        parser = ijson.kvitems(f, "")
        for idx, (timestamp, content) in enumerate(parser):
            # Convert Decimal values to float
            content = convert_decimal_to_float(content)
            
            row = {"timestamp": timestamp}
            
            # Parse timestamp to datetime
            try:
                row["datetime"] = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                row["datetime"] = pd.NaT
            
            # Flatten sensor values
            if "Sensor_values" in content:
                sensor_data = content["Sensor_values"]
                # Ensure all sensor values are numeric
                for key, value in sensor_data.items():
                    if isinstance(value, Decimal):
                        sensor_data[key] = float(value)
                row.update(sensor_data)
            
            # Better cycle detection - use timestamp or index if cycle count doesn't change
            cycle_count = content.get("Sensor_values", {}).get("Q_Cell_CycleCount", 0)
            
            # If cycle count is not changing, use timestamp-based cycles
            if idx == 0:
                row["cycle_id"] = 0
            else:
                # Use minute-based cycles to create more cycles for analysis
                try:
                    time_diff = (datetime.fromisoformat(timestamp.replace('Z', '+00:00')) - 
                                datetime.fromisoformat(records[0]['timestamp'].replace('Z', '+00:00')))
                    row["cycle_id"] = int(time_diff.total_seconds() // 60)  # New cycle every minute
                except:
                    row["cycle_id"] = idx  # Fallback to index
            
            # Keep image paths for potential computer vision analysis
            if "Images" in content:
                row["images"] = content["Images"]
                row["has_images"] = len(content["Images"]) > 0
            
            records.append(row)

            if limit and idx >= limit:
                break

    df = pd.DataFrame(records)
    
    # Convert all potential numeric columns to float
    for col in df.columns:
        if col not in ['timestamp', 'datetime', 'images', 'cycle_id']:
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
    
    print(f"Loaded {len(df)} records with {len(df['cycle_id'].unique())} unique cycles")
    
    # Feature engineering for waste indicators
    if len(df) > 0:
        # Calculate energy consumption patterns
        vfd_cols = [c for c in df.columns if 'VFD' in c and 'Temperature' in c]
        if vfd_cols:
            print(f"Processing {len(vfd_cols)} VFD temperature columns")
            df['avg_vfd_temperature'] = df[vfd_cols].apply(
                lambda row: np.mean([x for x in row if pd.notna(x)]), 
                axis=1
            )
            df['vfd_temperature_variance'] = df[vfd_cols].apply(
                lambda row: np.var([x for x in row if pd.notna(x)]), 
                axis=1
            )
        
        # Gripper load patterns (potential wear indicators)
        gripper_load_cols = [c for c in df.columns if 'Gripper_Load' in c]
        if gripper_load_cols:
            print(f"Processing {len(gripper_load_cols)} gripper load columns")
            df['total_gripper_load'] = df[gripper_load_cols].apply(
                lambda row: np.sum([x for x in row if pd.notna(x)]), 
                axis=1
            )
            df['gripper_load_imbalance'] = df[gripper_load_cols].apply(
                lambda row: np.std([x for x in row if pd.notna(x)]), 
                axis=1
            )
    
    return df

# -----------------------------
# Fixed CNC Repository Loader
# -----------------------------
def load_cnc_repository_enhanced(base_dir):
    excel_path = os.path.join(base_dir, "CNC Machining Data Respository.xlsx")
    
    try:
        print("Loading CNC repository...")
        # Try reading without assuming headers first
        cnc_df = pd.read_excel(excel_path, sheet_name=None, header=None)
        
        target_sheet = None
        best_row_count = 0
        
        for sheet_name, sheet_data in cnc_df.items():
            print(f"Sheet: {sheet_name}, Shape: {sheet_data.shape}")
            
            # Look for sheets with substantial data
            if sheet_data.shape[0] > best_row_count:
                target_sheet = sheet_data
                best_row_count = sheet_data.shape[0]
        
        if target_sheet is not None:
            print(f"Using sheet with {best_row_count} rows")
            
            # Clean the data - remove completely empty rows and columns
            target_sheet = target_sheet.dropna(how='all')
            
            # Remove completely empty columns
            target_sheet = target_sheet.dropna(axis=1, how='all')
            
            # Use generic column names to avoid string accessor issues
            target_sheet.columns = [f"cnc_param_{i}" for i in range(len(target_sheet.columns))]
            
            # Convert to numeric where possible
            for col in target_sheet.columns:
                target_sheet[col] = pd.to_numeric(target_sheet[col], errors='ignore')
            
            # Remove rows that are mostly text/headers
            numeric_mask = target_sheet.apply(lambda row: row.apply(lambda x: isinstance(x, (int, float))).sum(), axis=1)
            if len(numeric_mask) > 0:
                numeric_threshold = len(target_sheet.columns) * 0.5  # 50% numeric values
                target_sheet = target_sheet[numeric_mask >= numeric_threshold]
            
            print(f"Final CNC data shape: {target_sheet.shape}")
            if not target_sheet.empty:
                print(f"CNC numeric columns: {len(target_sheet.select_dtypes(include=[np.number]).columns)}")
            return target_sheet
        else:
            print("No suitable sheet found in CNC data")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error loading CNC data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

# -----------------------------
# Improved Milling Data Loader
# -----------------------------
def load_milling_enhanced(base_dir):
    print("Loading Milling/Tool Wear data...")
    
    # Strategy 1: Look for filelist.csv
    filelist_path = os.path.join(base_dir, "filelist.csv")
    if os.path.exists(filelist_path):
        try:
            filelist = pd.read_csv(filelist_path)
            data = []
            
            for idx, (fname, wear_label) in enumerate(zip(filelist["filename"], filelist["wear"])):
                fpath = os.path.join(base_dir, fname)
                
                if not os.path.exists(fpath):
                    print(f"File not found: {fpath}")
                    continue
                    
                try:
                    if fname.endswith(".csv"):
                        df = pd.read_csv(fpath)
                        # Add source information
                        df['tool_wear_source_file'] = fname
                        df['tool_wear_label'] = wear_label
                        df['milling_cycle_id'] = idx
                        
                        # Extract vibration/force patterns as waste indicators
                        vibration_cols = [c for c in df.columns if any(x in c.lower() for x in ['vib', 'accel', 'force'])]
                        if vibration_cols:
                            df['vibration_magnitude'] = df[vibration_cols].std(axis=1)
                            df['peak_vibration'] = df[vibration_cols].max(axis=1)
                        
                        data.append(df)
                        
                except Exception as e:
                    print(f"Error processing {fname}: {e}")
                    continue

            if data:
                combined = pd.concat(data, ignore_index=True)
                print(f"Loaded {len(combined)} milling records from filelist")
                
                # Create tool wear severity categories
                if 'tool_wear_label' in combined.columns:
                    combined['wear_severity'] = pd.cut(combined['tool_wear_label'], 
                                                     bins=[0, 50, 100, 160],
                                                     labels=['low', 'medium', 'high'])
                return combined
                
        except Exception as e:
            print(f"Error with filelist approach: {e}")
    
    # Strategy 2: Create more realistic synthetic data
    print("Creating realistic synthetic milling data...")
    synthetic_data = []
    np.random.seed(42)  # For reproducible results
    
    for i in range(50):  # Create 50 synthetic milling cycles
        # Realistic tool wear progression
        base_wear = np.random.randint(10, 100)
        time_points = 200  # More data points per cycle
        
        cycle_data = {
            'time': np.arange(time_points),
            'vibration_x': np.random.normal(0, 0.5 + i*0.01, time_points),  # Increasing vibration with cycles
            'vibration_y': np.random.normal(0, 0.5 + i*0.01, time_points),
            'force_z': np.random.normal(50, 5 + i*0.1, time_points),  # Increasing force variation
            'temperature': np.random.normal(25 + i*0.2, 2, time_points),  # Gradual temperature increase
            'tool_wear_label': base_wear + i * 2,  # Progressive wear
            'milling_cycle_id': i,
            'tool_wear_source_file': f'synthetic_cycle_{i}.csv'
        }
        df = pd.DataFrame(cycle_data)
        df['vibration_magnitude'] = df[['vibration_x', 'vibration_y']].std(axis=1)
        df['peak_vibration'] = df[['vibration_x', 'vibration_y']].max(axis=1)
        synthetic_data.append(df)
    
    combined = pd.concat(synthetic_data, ignore_index=True)
    combined['wear_severity'] = pd.cut(combined['tool_wear_label'], 
                                     bins=[0, 50, 100, 160],
                                     labels=['low', 'medium', 'high'])
    print(f"Created {len(combined)} realistic synthetic milling records")
    return combined

# -----------------------------
# Improved Merge Logic with Better Cycle Handling
# -----------------------------
def merge_datasets_enhanced(ff_df, cnc_df, milling_df):
    if ff_df.empty:
        print("No Future Factories data to merge")
        return pd.DataFrame()
    
    print(f"Merging {len(ff_df)} FF records with CNC and Milling data...")
    merged_data = []
    
    # Use Future Factories as base (temporal data)
    unique_cycles = ff_df['cycle_id'].unique()
    print(f"Processing {len(unique_cycles)} unique cycles")
    
    # If we have too few cycles, create synthetic variations for analysis
    if len(unique_cycles) < 10:
        print("Creating augmented cycles for better analysis...")
        augmented_cycles = []
        for cycle_id in unique_cycles:
            # Create 5 variations of each real cycle
            for variation in range(5):
                augmented_cycles.append(cycle_id * 100 + variation)
        unique_cycles = augmented_cycles[:50]  # Limit to 50 cycles max
        print(f"Augmented to {len(unique_cycles)} cycles for analysis")
    
    for cycle_id in unique_cycles:
        if cycle_id < 100:  # Real cycles
            ff_cycle = ff_df[ff_df['cycle_id'] == cycle_id]
        else:  # Synthetic augmented cycles
            base_cycle_id = cycle_id // 100
            ff_cycle = ff_df[ff_df['cycle_id'] == base_cycle_id].copy()
            # Add some noise to create variations
            for col in ff_cycle.select_dtypes(include=[np.number]).columns:
                if col != 'cycle_id':
                    noise = np.random.normal(0, ff_cycle[col].std() * 0.1)
                    ff_cycle[col] = ff_cycle[col] + noise
        
        if len(ff_cycle) == 0:
            continue
            
        cycle_data = {}
        
        # Base cycle information
        cycle_data['cycle_id'] = cycle_id
        cycle_data['is_synthetic'] = cycle_id >= 100  # Mark synthetic cycles
        
        if 'datetime' in ff_cycle.columns and not ff_cycle['datetime'].isna().all():
            cycle_data['timestamp'] = ff_cycle['datetime'].iloc[0]
        
        # Future Factories features - limit to most important columns
        numeric_cols = ff_cycle.select_dtypes(include=[np.number]).columns.tolist()
        if 'cycle_id' in numeric_cols:
            numeric_cols.remove('cycle_id')
        
        # Select key sensor columns for analysis
        key_columns = [col for col in numeric_cols if any(x in col for x in 
                      ['VFD', 'Gripper', 'Conv', 'CycleCount', 'temperature', 'load', 'speed'])]
        
        for col in key_columns[:15]:  # Limit to 15 key columns
            try:
                values = ff_cycle[col].dropna()
                if len(values) > 0:
                    cycle_data[f'ff_{col}_mean'] = values.mean()
                    cycle_data[f'ff_{col}_std'] = values.std()
                    cycle_data[f'ff_{col}_max'] = values.max()
            except Exception as e:
                cycle_data[f'ff_{col}_mean'] = np.nan
                cycle_data[f'ff_{col}_std'] = np.nan
                cycle_data[f'ff_{col}_max'] = np.nan
        
        # CNC Data Mapping
        if not cnc_df.empty:
            cnc_numeric = cnc_df.select_dtypes(include=[np.number])
            for col in cnc_numeric.columns[:3]:  # Limit to 3 CNC features
                try:
                    cycle_data[f'cnc_{col}_mean'] = cnc_numeric[col].mean()
                    cycle_data[f'cnc_{col}_std'] = cnc_numeric[col].std()
                except:
                    cycle_data[f'cnc_{col}_mean'] = np.nan
                    cycle_data[f'cnc_{col}_std'] = np.nan
        
        # Milling/Tool Wear Mapping
        if not milling_df.empty and 'tool_wear_label' in milling_df.columns:
            # Map based on cycle ID
            unique_milling_cycles = milling_df['milling_cycle_id'].unique()
            if len(unique_milling_cycles) > 0:
                mapped_cycle_id = cycle_id % len(unique_milling_cycles)
                wear_data = milling_df[milling_df['milling_cycle_id'] == mapped_cycle_id]
                
                if not wear_data.empty:
                    cycle_data['tool_wear'] = wear_data['tool_wear_label'].mean()
                    if 'vibration_magnitude' in wear_data.columns:
                        cycle_data['vibration_level'] = wear_data['vibration_magnitude'].mean()
                    if 'peak_vibration' in wear_data.columns:
                        cycle_data['peak_operation_force'] = wear_data['peak_vibration'].mean()
            
            # Global statistics
            cycle_data['global_tool_wear_mean'] = milling_df['tool_wear_label'].mean()
            cycle_data['global_tool_wear_std'] = milling_df['tool_wear_label'].std()
        
        # Calculate realistic waste indicators
        cycle_data = calculate_realistic_waste_indicators(cycle_data)
        merged_data.append(cycle_data)
        
        # Progress indicator
        if len(merged_data) % 20 == 0:
            print(f"Processed {len(merged_data)} cycles...")
    
    result_df = pd.DataFrame(merged_data)
    print(f"Final merged dataset: {result_df.shape}")
    return result_df

def calculate_realistic_waste_indicators(cycle_data):
    """Calculate realistic material waste and efficiency indicators"""
    
    # Realistic energy efficiency calculation
    vfd_temp = cycle_data.get('ff_avg_vfd_temperature_mean', 25)
    if not pd.isna(vfd_temp) and vfd_temp > 0:
        # Normalize temperature to 0-1 scale (assuming 20-80°C operating range)
        temp_norm = min(max((vfd_temp - 20) / 60, 0), 1)
        cycle_data['energy_efficiency'] = 1 - temp_norm * 0.3  # 30% efficiency loss at high temp
    else:
        cycle_data['energy_efficiency'] = 0.8  # Default efficiency
    
    # Realistic material waste calculation
    tool_wear = cycle_data.get('tool_wear', 50)
    vibration = cycle_data.get('vibration_level', 1.0)
    gripper_imbalance = cycle_data.get('ff_gripper_load_imbalance_mean', 0)
    
    # Normalize inputs
    wear_contribution = min(tool_wear / 160, 1.0)
    vibration_contribution = min(vibration / 2.0, 1.0)  # More realistic vibration scale
    gripper_contribution = min(gripper_imbalance / 30, 1.0)  # More realistic gripper scale
    
    # Calculate waste with realistic weights
    cycle_data['predicted_material_waste'] = (
        wear_contribution * 0.6 +  # Tool wear is most important
        vibration_contribution * 0.25 + 
        gripper_contribution * 0.15
    )
    
    cycle_data['process_stability'] = 1 - cycle_data['predicted_material_waste']
    cycle_data['overall_efficiency'] = cycle_data['energy_efficiency'] * cycle_data['process_stability']
    
    # Add quality metrics
    cycle_data['quality_score'] = 1 - (cycle_data['predicted_material_waste'] * 0.8)
    
    return cycle_data

# -----------------------------
# Main Execution
# -----------------------------
def main():
    base_path = "/Users/appdev/Downloads/Waste_Reduction"
    
    print("=" * 60)
    print("WASTE REDUCTION AI - IMPROVED DATA MERGING")
    print("=" * 60)
    
    # Load datasets
    ff_df = load_future_factories_enhanced(base_path, limit=1000)
    
    cnc_path = os.path.join(base_path, "CNC Machining Data Repository - Geometry, NC Code & High-Frequency Energy Consumption Data for Aluminum and Plastic Machining")
    cnc_df = load_cnc_repository_enhanced(cnc_path)
    
    milling_path = os.path.join(base_path, "Multivariate time series data of milling processes with varying tool wear and machine tools")
    milling_df = load_milling_enhanced(milling_path)
    
    # Merge datasets
    merged = merge_datasets_enhanced(ff_df, cnc_df, milling_df)
    
    if not merged.empty:
        # Save results
        output_path = os.path.join(base_path, "improved_merged_production_dataset.csv")
        merged.to_csv(output_path, index=False)
        print(f"\nSaved merged dataset to: {output_path}")
        
        # Generate comprehensive report
        generate_improved_report(merged)
    else:
        print("No data was merged - check your data sources")

def generate_improved_report(merged_df):
    """Generate a realistic waste analysis report"""
    print("\n" + "=" * 60)
    print("WASTE REDUCTION AI - REALISTIC ANALYSIS REPORT")
    print("=" * 60)
    
    # Filter out synthetic cycles for real analysis
    real_cycles = merged_df[~merged_df.get('is_synthetic', True)]
    synthetic_cycles = merged_df[merged_df.get('is_synthetic', False)]
    
    print(f"\nDataset Overview:")
    print(f"Total production cycles: {len(merged_df):,}")
    print(f"Real cycles: {len(real_cycles):,}")
    print(f"Synthetic cycles (for analysis): {len(synthetic_cycles):,}")
    print(f"Total features: {len(merged_df.columns):,}")
    
    # Waste Analysis using all data but noting synthetic nature
    waste_metrics = ['predicted_material_waste', 'process_stability', 'energy_efficiency', 'overall_efficiency', 'quality_score']
    available_metrics = [m for m in waste_metrics if m in merged_df.columns]
    
    print(f"\nKey Performance Metrics (using all data):")
    for metric in available_metrics:
        values = merged_df[metric].dropna()
        if len(values) > 0:
            print(f"  {metric:25} | Mean: {values.mean():.3f} | Std: {values.std():.3f}")
    
    # High waste analysis
    if 'predicted_material_waste' in merged_df.columns:
        high_waste_threshold = merged_df['predicted_material_waste'].quantile(0.75)
        high_waste_cycles = merged_df[merged_df['predicted_material_waste'] > high_waste_threshold]
        
        print(f"\nHigh Waste Analysis (top 25%):")
        print(f"  High waste cycles: {len(high_waste_cycles)}")
        if 'tool_wear' in high_waste_cycles.columns:
            print(f"  Avg tool wear in high waste: {high_waste_cycles['tool_wear'].mean():.1f}")
        if 'energy_efficiency' in high_waste_cycles.columns:
            print(f"  Avg energy efficiency: {high_waste_cycles['energy_efficiency'].mean():.3f}")
    
    # Root cause analysis
    print(f"\nRoot Cause Analysis:")
    if 'predicted_material_waste' in merged_df.columns:
        # Calculate correlations only if we have enough data
        if len(merged_df) >= 10:
            numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
            waste_correlations = merged_df[numeric_cols].corr()['predicted_material_waste'].abs().sort_values(ascending=False)
            
            # Get meaningful correlations (not perfect 1.0)
            meaningful_corrs = waste_correlations[(waste_correlations < 0.99) & (waste_correlations > 0.1)]
            if len(meaningful_corrs) > 0:
                print("  Top factors influencing material waste:")
                for factor, corr in meaningful_corrs.head(5).items():
                    print(f"    - {factor:40} | Impact: {corr:.3f}")
            else:
                print("  Not enough variation in data for correlation analysis")
        else:
            print("  Need more data points for reliable correlation analysis")
    
    print(f"\nRecommended Actions for Waste Reduction:")
    print("  1. Monitor and replace tools before wear exceeds 100 units")
    print("  2. Maintain vibration levels below 1.5 for optimal performance")
    print("  3. Balance gripper loads to minimize imbalance above 15")
    print("  4. Keep VFD temperatures below 60°C for better energy efficiency")
    print("  5. Implement real-time monitoring of these key parameters")

if __name__ == "__main__":
    main()