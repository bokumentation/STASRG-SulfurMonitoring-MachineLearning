import pandas as pd
import numpy as np

def convert_h2s_to_ppm(h2s_ugm3, temp_c):
    """
    Convert H2S from µg/m³ to ppm using temperature-dependent formula
    
    Args:
        h2s_ugm3: H2S concentration in µg/m³
        temp_c: Temperature in Celsius
    
    Returns:
        H2S concentration in ppm
    """
    temp_kelvin = temp_c + 273.15
    constant = temp_kelvin * 0.082057
    h2s_ppm = (h2s_ugm3 * constant) / 34080
    return h2s_ppm

def classify_so2(so2_ugm3):
    """
    Classify SO2 level based on 10-minute average thresholds
    
    Returns: (score, label)
    """
    if so2_ugm3 < 100:
        return 0, "Normal"
    elif so2_ugm3 < 200:
        return 1, "Caution"
    elif so2_ugm3 < 500:
        return 2, "Warning"
    else:
        return 3, "Danger"  # ≥500 is Danger, we can add Critical threshold later if needed

def classify_h2s(h2s_ppm):
    """
    Classify H2S level based on 10-30 minute average thresholds
    
    Returns: (score, label)
    """
    if h2s_ppm < 0.005:
        return 0, "Normal"
    elif h2s_ppm < 0.03:
        return 1, "Caution"
    elif h2s_ppm < 0.1:
        return 2, "Warning"
    elif h2s_ppm < 1.0:
        return 3, "Danger"
    else:
        return 4, "Critical"  # ≥1 ppm for public

def combine_classifications(so2_score, h2s_score):
    """
    Combine SO2 and H2S classifications using weighted average with safety override
    
    Logic:
    - If EITHER pollutant is Danger(3) or Critical(4), take the maximum
    - Otherwise, average the scores and round
    
    Returns: final_score (0-4)
    """
    # Safety override for high severity
    if so2_score >= 3 or h2s_score >= 3:
        return max(so2_score, h2s_score)
    else:
        # Average and round for lower severity levels
        return round((so2_score + h2s_score) / 2.0)

def score_to_label(score):
    """Convert numeric score to text label"""
    labels = {
        0: "Normal",
        1: "Caution", 
        2: "Warning",
        3: "Danger",
        4: "Critical"
    }
    return labels.get(score, "Unknown")

def label_dataset(input_csv, output_csv):
    """
    Main function to load data, apply labeling logic, and save labeled dataset
    
    Args:
        input_csv: Path to raw sensor data CSV
        output_csv: Path to save labeled dataset
    """
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"Loaded {len(df)} samples from {df['node'].nunique()} nodes")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Convert H2S to ppm
    print("\nConverting H2S to ppm...")
    df['h2s_ppm'] = df.apply(lambda row: convert_h2s_to_ppm(row['h2s_ugm3'], row['temp_c']), axis=1)
    
    # Classify each pollutant
    print("Classifying pollutant levels...")
    df['so2_score'], df['so2_label'] = zip(*df['so2_ugm3'].apply(classify_so2))
    df['h2s_score'], df['h2s_label'] = zip(*df['h2s_ppm'].apply(classify_h2s))
    
    # Combine classifications
    print("Applying combination logic...")
    df['final_score'] = df.apply(lambda row: combine_classifications(row['so2_score'], row['h2s_score']), axis=1)
    df['label'] = df['final_score'].apply(score_to_label)
    
    # Show class distribution
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION:")
    print("="*60)
    class_counts = df['label'].value_counts().sort_index()
    for label, count in class_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{label:10s}: {count:5d} samples ({percentage:5.1f}%)")
    
    # Show sample statistics
    print("\n" + "="*60)
    print("POLLUTANT RANGES:")
    print("="*60)
    print(f"SO2:  {df['so2_ugm3'].min():6.1f} - {df['so2_ugm3'].max():6.1f} µg/m³")
    print(f"H2S:  {df['h2s_ugm3'].min():6.1f} - {df['h2s_ugm3'].max():6.1f} µg/m³")
    print(f"      {df['h2s_ppm'].min():6.4f} - {df['h2s_ppm'].max():6.4f} ppm")
    
    # Prepare final dataset for ML training
    # Features: node, so2_ugm3, h2s_ugm3, temp_c, rh_pct, wind_speed_ms
    # Optional: wind_dir_deg, battery_v, rssi
    ml_features = ['node', 'so2_ugm3', 'h2s_ugm3', 'temp_c', 'rh_pct', 'wind_speed_ms']
    
    # Check if optional features exist
    optional_features = ['wind_dir_deg', 'battery_v', 'rssi']
    for feat in optional_features:
        if feat in df.columns:
            ml_features.append(feat)
    
    # Add label
    ml_features.append('label')
    
    # Save labeled dataset
    df_output = df[ml_features]
    df_output.to_csv(output_csv, index=False)
    
    print(f"\n✅ Labeled dataset saved to: {output_csv}")
    print(f"Features: {', '.join(ml_features[:-1])}")
    print(f"Label: {ml_features[-1]}")
    print(f"Total samples: {len(df_output)}")
    
    return df_output

# Example usage
if __name__ == "__main__":
    # Replace with your actual file paths
    input_file = "kawah_putih_synthetic.csv"
    output_file = "kawah_putih_labeled.csv"
    
    try:
        labeled_data = label_dataset(input_file, output_file)
        print("\n✅ Data labeling complete!")
        print("\nNext step: Train your neural network on this labeled dataset")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {input_file}")
        print("Please update the input_file path to match your CSV filename")
    except Exception as e:
        print(f"❌ Error during labeling: {e}")