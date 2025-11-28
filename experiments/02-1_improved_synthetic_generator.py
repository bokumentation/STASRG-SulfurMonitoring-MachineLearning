import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_mountain_valley_wind(hour, base_speed=2.5):
    """
    Generate realistic mountain valley wind patterns
    
    Daytime (6am-6pm): Upslope winds, generally stronger
    Nighttime (6pm-6am): Downslope winds, generally weaker
    
    Returns: (wind_speed_ms, wind_direction_deg)
    """
    # Daytime upslope (stronger winds, direction ~180-270deg)
    if 6 <= hour < 18:
        speed = base_speed + np.random.uniform(0.5, 2.0)
        direction = np.random.uniform(180, 270)  # Southerly to westerly
    # Nighttime downslope (weaker winds, direction ~0-90deg)
    else:
        speed = base_speed + np.random.uniform(-0.5, 1.0)
        direction = np.random.uniform(0, 90)  # Northerly to easterly
    
    speed = max(0.5, speed)  # Minimum 0.5 m/s
    return speed, direction

def calculate_wind_dispersion_factor(wind_speed, wind_dir, node_position):
    """
    Calculate how wind affects pollutant concentration at each node
    
    Args:
        wind_speed: Wind speed in m/s
        wind_dir: Wind direction in degrees (0=North, 90=East, 180=South, 270=West)
        node_position: Node's position relative to fumarole
                       1 = closest to source
                       2 = middle distance
                       3 = farthest (tourist area)
    
    Returns:
        dispersion_factor: multiplier for base pollution (0.5 to 2.0)
    """
    # Base dispersion from wind speed (higher wind = more dispersion = lower concentration)
    speed_factor = 1.0 / (1.0 + wind_speed * 0.2)
    
    # Direction effect: simplified model
    # Assume fumarole is at center, nodes arranged roughly north-south
    # Node 1 (north, near source), Node 2 (center), Node 3 (south, tourist area)
    
    if node_position == 1:
        # Node 1: Near source, always has high pollution
        # Gets extra pollution if wind is from south (bringing plume toward it)
        if 135 <= wind_dir <= 225:  # Wind from south
            direction_factor = 1.5
        else:
            direction_factor = 1.2
            
    elif node_position == 2:
        # Node 2: Middle position
        # Moderate pollution, affected by wind direction
        if 90 <= wind_dir <= 180:  # Wind from east/south
            direction_factor = 1.3
        else:
            direction_factor = 1.0
            
    else:  # node_position == 3
        # Node 3: Tourist area, farthest from source
        # Lower pollution normally, but can spike if wind brings plume
        if 0 <= wind_dir <= 45 or 315 <= wind_dir <= 360:  # Wind from north
            direction_factor = 1.8  # Plume blown toward tourist area!
        else:
            direction_factor = 0.7  # Wind blows plume away
    
    return speed_factor * direction_factor

def generate_temporal_pattern(num_samples, base_value, volatility=0.15):
    """
    Generate time-series with gradual changes and occasional spikes
    Uses random walk with mean reversion
    
    Args:
        num_samples: Number of time steps
        base_value: Average value to oscillate around
        volatility: How much variation (0.1 = 10% typical change)
    
    Returns:
        Array of values with temporal continuity
    """
    values = np.zeros(num_samples)
    values[0] = base_value
    
    for i in range(1, num_samples):
        # Random walk with mean reversion
        change = np.random.normal(0, volatility * base_value)
        mean_reversion = (base_value - values[i-1]) * 0.05  # Pull back toward mean
        
        # Occasional spike (5% chance)
        if np.random.random() < 0.05:
            spike = np.random.uniform(0.5, 1.5) * base_value
            values[i] = values[i-1] + spike
        else:
            values[i] = values[i-1] + change + mean_reversion
        
        # Keep positive
        values[i] = max(0, values[i])
    
    return values

def generate_kawah_putih_dataset(num_samples=20000, start_date="2025-12-01 00:00:00"):
    """
    Generate realistic synthetic dataset for Kawah Putih volcanic monitoring
    
    Simulates 3 sensor nodes with:
    - Spatial variation (distance from fumarole)
    - Temporal patterns (smooth transitions with spikes)
    - Mountain valley wind cycles
    - Environmental correlations
    """
    
    samples_per_node = num_samples
    total_samples = samples_per_node * 3
    
    print(f"Generating {samples_per_node:,} samples per node...")
    print(f"Total dataset: {total_samples:,} samples")
    print(f"Simulated duration: ~{samples_per_node * 10 / 3600:.1f} hours per node")
    
    # Time parameters
    start = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    interval_seconds = 10
    
    data = []
    
    for node_id in [1, 2, 3]:
        print(f"\nGenerating Node {node_id} data...")
        
        # Base pollution levels depend on distance from source
        if node_id == 1:
            so2_base = 450  # High - near fumarole
            h2s_base = 200
        elif node_id == 2:
            so2_base = 300  # Medium - middle distance
            h2s_base = 120
        else:  # node_id == 3
            so2_base = 180  # Lower - tourist area
            h2s_base = 60
        
        # Generate base temporal patterns
        so2_temporal = generate_temporal_pattern(samples_per_node, so2_base, volatility=0.20)
        h2s_temporal = generate_temporal_pattern(samples_per_node, h2s_base, volatility=0.18)
        
        # Generate environmental variables
        temp_base = 22  # Base temperature at crater (cooler than lowlands)
        humidity_base = 75
        
        for i in range(samples_per_node):
            timestamp = start + timedelta(seconds=i * interval_seconds)
            hour = timestamp.hour
            
            # Temperature: cooler at night, warmer during day
            temp_variation = 3 * np.sin((hour - 6) * np.pi / 12)  # Peak at noon
            temp = temp_base + temp_variation + np.random.normal(0, 0.5)
            temp = max(15, min(30, temp))  # Clamp to reasonable range
            
            # Humidity: inverse relationship with temperature
            humidity = humidity_base - (temp - temp_base) * 2 + np.random.normal(0, 3)
            humidity = max(50, min(95, humidity))
            
            # Mountain valley wind
            wind_speed, wind_dir = generate_mountain_valley_wind(hour)
            
            # Apply wind dispersion to pollutants
            dispersion = calculate_wind_dispersion_factor(wind_speed, wind_dir, node_id)
            
            so2 = so2_temporal[i] * dispersion
            h2s = h2s_temporal[i] * dispersion
            
            # Add small random noise
            so2 += np.random.normal(0, 5)
            h2s += np.random.normal(0, 2)
            
            # Ensure non-negative and within realistic ranges
            so2 = max(60, min(1080, so2))  # Match your stated range
            h2s = max(0, min(360, h2s))
            
            # Simulate battery drain (slow decrease over time)
            battery = 11.5 - (i / samples_per_node) * 0.5 + np.random.normal(0, 0.02)
            
            # RSSI varies with environmental conditions
            rssi = -70 + np.random.normal(0, 5)
            
            # Convert to PPM for display (using formula)
            temp_kelvin = temp + 273.15
            constant = temp_kelvin * 0.082057
            so2_ppm = (so2 * constant) / 64060
            h2s_ppm = (h2s * constant) / 34080
            
            data.append({
                'node': node_id,
                'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'so2_ppm': round(so2_ppm, 3),
                'so2_ugm3': round(so2, 2),
                'h2s_ppm': round(h2s_ppm, 3),
                'h2s_ugm3': round(h2s, 2),
                'temp_c': round(temp, 0),
                'rh_pct': round(humidity, 0),
                'wind_speed_ms': round(wind_speed, 3),
                'wind_dir_deg': round(wind_dir, 0),
                'battery_v': round(battery, 2),
                'rssi': round(rssi, 0)
            })
    
    df = pd.DataFrame(data)
    
    print("\n" + "="*60)
    print("DATASET STATISTICS:")
    print("="*60)
    print(f"Total samples: {len(df):,}")
    print(f"Nodes: {df['node'].nunique()}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nPollutant Ranges:")
    print(f"  SO2:  {df['so2_ugm3'].min():.1f} - {df['so2_ugm3'].max():.1f} µg/m³")
    print(f"  H2S:  {df['h2s_ugm3'].min():.1f} - {df['h2s_ugm3'].max():.1f} µg/m³")
    print(f"\nBy Node:")
    for node in [1, 2, 3]:
        node_data = df[df['node'] == node]
        print(f"  Node {node} - SO2: {node_data['so2_ugm3'].mean():.1f} µg/m³ (avg), "
              f"H2S: {node_data['h2s_ugm3'].mean():.1f} µg/m³ (avg)")
    
    return df

if __name__ == "__main__":
    # Generate dataset
    df = generate_kawah_putih_dataset(num_samples=20000)
    
    # Save to CSV
    output_file = "kawah_putih_synthetic_v2.csv"
    df.to_csv(output_file, index=False)
    print(f"\nDataset saved to: {output_file}")
    print(f"\nNext steps:")
    print(f"1. Run the labeling script on this data")
    print(f"2. Train your model on the labeled data")
    print(f"3. Compare accuracy with previous 70% baseline")