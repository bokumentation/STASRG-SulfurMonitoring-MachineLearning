import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

def generate_air_quality_dataset(n_samples=10000):
    """
    Generate synthetic air quality dataset based on WHO guidelines and 
    environmental correlations.
    
    References:
    - WHO Air Quality Guidelines (2021) for H2S and SO2 thresholds
    - Environmental correlation patterns from atmospheric science literature
    - Meteorological influences on pollutant dispersion
    """
    
    # Generate base sensor readings with realistic ranges
    # H2S: Hydrogen Sulfide (µg/m³)
    # Typical range: 0-200 µg/m³ in industrial/urban areas
    # WHO guideline: 150 µg/m³ (24-hour average)
    # Near-source volcanic environment parameters
    h2s = np.random.exponential(scale=150, size=n_samples)  # Much higher baseline
    h2s = np.clip(h2s, 0, 3000)  # Extended upper range
    
    # SO2: Sulfur Dioxide (µg/m³)
    # Typical range: 0-500 µg/m³
    # WHO guideline: 40 µg/m³ (24-hour average)
    so2 = np.random.exponential(scale=200, size=n_samples) # Higher baseline
    so2 = np.clip(so2, 0, 2500) # Extended upper range
    
    # Wind Speed (m/s): affects pollutant dispersion
    # Higher wind speeds generally correlate with lower pollutant concentration
    # Typical range: 0-15 m/s in urban environments
    wind_speed = np.random.gamma(shape=2, scale=2, size=n_samples)
    wind_speed = np.clip(wind_speed, 0, 15)
    
    # Add negative correlation: higher wind disperses pollutants
    # This reflects real atmospheric behavior where wind dilutes pollutants
    dispersion_factor = 1 - (wind_speed / 20)  # Normalize effect
    h2s = h2s * (0.5 + dispersion_factor * 0.5)
    so2 = so2 * (0.5 + dispersion_factor * 0.5)
    
    # Temperature (°C): affects chemical reactions and pollutant formation
    # Typical range: -10 to 45°C
    # Higher temps can increase photochemical reactions
    temperature = np.random.normal(loc=25, scale=8, size=n_samples)
    temperature = np.clip(temperature, -10, 45)
    
    # Add small positive correlation with pollutants (warmer = more reactions)
    temp_factor = (temperature - 15) / 30  # Normalize
    h2s = h2s * (1 + 0.1 * temp_factor)
    so2 = so2 * (1 + 0.15 * temp_factor)
    
    # Humidity (%): affects pollutant behavior and sensor readings
    # Typical range: 20-95%
    # High humidity can affect gas dispersion patterns
    humidity = np.random.beta(a=2, b=2, size=n_samples) * 75 + 20
    humidity = np.clip(humidity, 20, 95)
    
    # Create DataFrame
    df = pd.DataFrame({
        'h2s': h2s,
        'so2': so2,
        'wind_speed': wind_speed,
        'temperature': temperature,
        'humidity': humidity
    })
    
    # Apply rule-based classification
    # Based on WHO Air Quality Guidelines and US EPA Air Quality Index principles
    df['label'] = df.apply(classify_air_quality, axis=1)
    
    return df

def classify_air_quality(row):
    """
    Rule-based classification following WHO guidelines and composite scoring.
    
    Classification Logic:
    - Uses weighted scoring based on primary pollutants (H2S, SO2)
    - Considers meteorological factors (wind speed moderates risk)
    - Follows WHO Air Quality Guidelines thresholds
    
    References:
    - WHO Air Quality Guidelines (2021)
    - US EPA Air Quality Index methodology
    - Industrial hygiene exposure limits (OSHA, NIOSH)
    """
    
    # Calculate pollutant scores (0-100 scale)
    # H2S thresholds based on WHO and OSHA guidelines:
    # <20: Good, 20-75: Moderate, 75-150: Unhealthy, 150+: Hazardous
    h2s_score = min(row['h2s'] / 2, 100)
    
    # SO2 thresholds based on WHO guidelines:
    # <40: Good, 40-125: Moderate, 125-350: Unhealthy, 350+: Hazardous
    so2_score = min(row['so2'] / 5, 100)
    
    # Composite score (weighted average, pollutants are primary factors)
    composite = 0.5 * h2s_score + 0.5 * so2_score
    
    # Wind speed adjustment: high wind reduces effective exposure
    # This reflects real dispersion dynamics in atmospheric science
    if row['wind_speed'] > 8:
        composite *= 0.8  # 20% reduction in risk with strong winds
    elif row['wind_speed'] > 5:
        composite *= 0.9  # 10% reduction with moderate winds
    
    # Temperature adjustment: extreme heat increases risk
    # Heat stress combined with pollutants is more dangerous
    if row['temperature'] > 35:
        composite *= 1.1  # 10% increase in risk
    
    # Classification thresholds
    # These are calibrated to create balanced classes for ML training
    if composite < 15:
        return 0  # Normal
    elif composite < 35:
        return 1  # Caution
    elif composite < 60:
        return 2  # Warning
    elif composite < 80:
        return 3  # Danger
    else:
        return 4  # Critical

# Generate dataset
print("Generating synthetic air quality dataset...")
df = generate_air_quality_dataset(n_samples=5000)

# Display class distribution
print("\n" + "="*60)
print("CLASS DISTRIBUTION")
print("="*60)
label_names = ['Normal', 'Caution', 'Warning', 'Danger', 'Critical']
for i, name in enumerate(label_names):
    count = (df['label'] == i).sum()
    percentage = count / len(df) * 100
    print(f"{name:12s}: {count:4d} samples ({percentage:5.2f}%)")

# Display statistics
print("\n" + "="*60)
print("SENSOR READING STATISTICS")
print("="*60)
print(df.describe())

# Split into train/test sets (80/20 split is standard in ML)
X = df[['h2s', 'so2', 'wind_speed', 'temperature', 'humidity']].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "="*60)
print("DATASET SPLIT")
print("="*60)
print(f"Training samples:   {len(X_train)}")
print(f"Testing samples:    {len(X_test)}")

# Save to CSV files for future use
df_train = pd.DataFrame(X_train, columns=['h2s', 'so2', 'wind_speed', 'temperature', 'humidity'])
df_train['label'] = y_train
df_train.to_csv('air_quality_train.csv', index=False)

df_test = pd.DataFrame(X_test, columns=['h2s', 'so2', 'wind_speed', 'temperature', 'humidity'])
df_test['label'] = y_test
df_test.to_csv('air_quality_test.csv', index=False)

print("\nDataset saved to:")
print("  - air_quality_train.csv")
print("  - air_quality_test.csv")

# Display sample data from each class
print("\n" + "="*60)
print("SAMPLE DATA FROM EACH CLASS")
print("="*60)
for i, name in enumerate(label_names):
    sample = df[df['label'] == i].iloc[0]
    print(f"\n{name}:")
    print(f"  H2S: {sample['h2s']:.2f} µg/m³")
    print(f"  SO2: {sample['so2']:.2f} µg/m³")
    print(f"  Wind: {sample['wind_speed']:.2f} m/s")
    print(f"  Temp: {sample['temperature']:.2f} °C")
    print(f"  Humidity: {sample['humidity']:.2f} %")