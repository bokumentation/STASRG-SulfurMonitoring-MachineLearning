# ESP32 Air Quality ML Monitoring System

**Location:** Kawah Putih Volcanic Crater, Ciwidey, Indonesia  
**Purpose:** Distributed air quality monitoring using on-device ML inference  
**Hardware:** 5 × ESP32 microcontrollers  
**Status:** 🟡 In Development - ML Pipeline Complete, Hardware Integration In Progress

---

## Project Overview

This system deploys a trained neural network on ESP32 devices to monitor air quality near volcanic gas sources. Each ESP32 runs local inference to classify air quality conditions in real-time, providing immediate safety alerts without requiring cloud connectivity.

### Why This Approach?
- **Edge ML**: Inference runs locally on each ESP32 (no internet needed for predictions)
- **Safety-Critical**: Volcanic environments need immediate warnings, can't rely on network latency
- **Distributed**: 5 devices provide spatial coverage around the crater
- **Lightweight**: Entire model is only 7.22 KB, runs in <10ms per inference

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     KAWAH PUTIH CRATER                      │
│                                                             │
│   ESP32 #1 ◄──┐                                            │
│                │                                            │
│   ESP32 #2 ◄──┼──► Network Communication (ESP-NOW/LoRa)   │
│                │                                            │
│   ESP32 #3 ◄──┤                                            │
│                │                                            │
│   ESP32 #4 ◄──┤                                            │
│                │                                            │
│   ESP32 #5 ◄──┘                                            │
│                                                             │
│   Each ESP32:                                              │
│   • Reads 5 sensors (H2S, SO2, wind, temp, humidity)      │
│   • Runs ML inference locally                              │
│   • Classifies air quality (Normal → Critical)             │
│   • Broadcasts result to network                           │
│   • Triggers local alarms if dangerous                     │
└─────────────────────────────────────────────────────────────┘
```

---

## ML Model Specifications

### Model Architecture
- **Type:** Feedforward Artificial Neural Network (ANN)
- **Framework:** PyTorch (training) → ONNX → Custom C (deployment)
- **Size:** 7.22 KB (Float32), uses ~1.5% of ESP32 RAM
- **Inference Time:** <10ms per prediction
- **Parameters:** 2,629 total

**Architecture Details:**
```
Input Layer (5 neurons)
    ↓
Hidden Layer 1 (64 neurons, ReLU, Dropout 0.2)
    ↓
Hidden Layer 2 (32 neurons, ReLU, Dropout 0.2)
    ↓
Output Layer (5 neurons, Softmax)
```

### Input Requirements (Your Sensor Data Interface)

The model expects **5 input features** in this exact order:

| Index | Feature      | Unit    | Range (typical)  | Description                    |
|-------|--------------|---------|------------------|--------------------------------|
| 0     | `h2s`        | µg/m³   | 0 - 500          | Hydrogen Sulfide concentration |
| 1     | `so2`        | µg/m³   | 0 - 800          | Sulfur Dioxide concentration   |
| 2     | `wind_speed` | m/s     | 0 - 15           | Wind speed                     |
| 3     | `temperature`| °C      | 15 - 35          | Ambient temperature            |
| 4     | `humidity`   | %       | 30 - 100         | Relative humidity              |

**⚠️ Important for Sensor Team:**
- Values must be in these exact units
- Send raw sensor readings - no preprocessing needed
- Model handles normalization internally
- If a sensor fails, send `0` or last known value (we'll add fault detection later)

### Output Format

The model outputs **5 class probabilities** (summing to 1.0):

| Class ID | Name       | Description                | Typical Action                |
|----------|------------|----------------------------|-------------------------------|
| 0        | Normal     | Safe air quality           | Green LED, no alarm           |
| 1        | Caution    | Slight concern             | Yellow LED                    |
| 2        | Warning    | Moderate risk              | Orange LED, log data          |
| 3        | Danger     | High risk                  | Red LED, alarm beep           |
| 4        | Critical   | Extreme danger             | Red LED, continuous alarm     |

**Example Output:**
```c
float probabilities[5] = {0.05, 0.12, 0.23, 0.55, 0.05};
// Predicted class: 3 (Danger) - 55% confidence
```

### Model Behavior Characteristics

**Conservative Bias:** The model is intentionally trained to prefer false alarms over missed dangers.
- **Critical class recall:** 92% (catches dangerous conditions reliably)
- **Overall accuracy:** 70% on test data
- **Why 70%?** We're using synthetic data until real Kawah Putih data is available. Model will be retrained with actual field measurements.

---

## Project Structure

```
esp32-air-quality-ml/
│
├── data/                          # Training datasets
│   ├── air_quality_train.csv      # 4000 samples
│   └── air_quality_test.csv       # 1000 samples
│
├── models/                        # Trained models
│   ├── best_model.pth             # Use this one (PyTorch)
│   └── air_quality_model.onnx     # ONNX format
│
├── src/                           # Python ML pipeline
│   ├── 01-1_synthetic_dataset_generator.py
│   ├── 01-2_train_model.py
│   ├── 01-3_onnx_conversion.py
│   └── 01-4_weight_extractor.py
│
├── firmware/                      # ESP32 code (ESP-IDF)
│   ├── main/
│   │   ├── main.c                 # Main application
│   │   ├── model_inference.c      # ML inference engine
│   │   ├── model_inference.h
│   │   ├── model_weights.h        # Trained weights (33 KB)
│   │   ├── sensor_interface.c     # Your team works here
│   │   └── network_comm.c         # Network team works here
│   └── CMakeLists.txt
│
├── experiments/                   # Training logs & visualizations
│   ├── confusion_matrix.png
│   └── training_curves.png
│
└── README.md                      # This file
```

---

## Current Progress

### ✅ Completed
1. **Synthetic dataset generation** - Volcanic-profile data for development
2. **Model training** - PyTorch ANN with safety-focused configuration
3. **ONNX conversion** - Portable model format (Float32)
4. **Weight extraction** - Converted to C arrays for ESP32
5. **C inference engine** - Custom implementation (no TensorFlow Lite needed)
6. **PC testing** - Verifying inference accuracy before deployment

### 🔄 In Progress
7. **ESP-IDF project setup** - Porting C code to ESP32 firmware
8. **Hardware integration** - Connecting to actual sensors

### ⏳ Pending (Need Collaboration)
9. **Sensor interface** - Reading H2S, SO2, wind, temp, humidity sensors
10. **Network communication** - ESP-NOW or LoRa for device coordination
11. **Safety features** - Alarms, watchdog, data logging
12. **Field testing** - Deploy at Kawah Putih, collect real data
13. **Model retraining** - Use real field data to improve accuracy

---

## How to Interface with This Work

### For Sensor Hardware Team

**What you need to provide:**
```c
// Function signature you'll implement
typedef struct {
    float h2s;          // µg/m³
    float so2;          // µg/m³
    float wind_speed;   // m/s
    float temperature;  // °C
    float humidity;     // %
} SensorData;

SensorData read_sensors(void);
```

**What you'll receive back:**
```c
// Function we provide
typedef struct {
    int predicted_class;      // 0-4 (Normal to Critical)
    float confidence;         // 0.0-1.0
    float probabilities[5];   // Full probability distribution
} PredictionResult;

PredictionResult model_predict(float inputs[5]);
```

**Example integration:**
```c
SensorData data = read_sensors();
float inputs[5] = {data.h2s, data.so2, data.wind_speed, 
                   data.temperature, data.humidity};
PredictionResult result = model_predict(inputs);

if (result.predicted_class >= 3) {  // Danger or Critical
    trigger_alarm();
}
```

### For Network/Integration Team

**Each ESP32 will broadcast:**
```c
typedef struct {
    uint8_t device_id;          // 1-5
    float gps_lat;              // Device location
    float gps_lon;
    uint32_t timestamp;         // Unix timestamp
    int air_quality_class;      // 0-4
    float confidence;           // 0.0-1.0
    SensorData raw_sensors;     // Raw sensor values
} DeviceMessage;
```

**Communication requirements:**
- Update frequency: 1-10 Hz (configurable)
- Network protocol: TBD (ESP-NOW recommended for low latency)
- Range: ~100-200m between devices
- Fallback: Each device operates independently if network fails

---

## Running the ML Pipeline (For Development)

### Prerequisites
```bash
pip install torch torchvision onnx numpy pandas matplotlib scikit-learn
```

### Generate Synthetic Data
```bash
python 01-1_synthetic_dataset_generator.py
# Outputs: data/air_quality_train.csv, data/air_quality_test.csv
```

### Train Model
```bash
python 01-2_train_model.py
# Outputs: models/best_model.pth, experiments/confusion_matrix.png
```

### Convert to ONNX
```bash
python 01-3_onnx_conversion.py
# Outputs: models/air_quality_model.onnx
```

### Extract Weights for ESP32
```bash
python 01-4_weight_extractor.py
# Outputs: firmware/main/model_weights.h
```

---

## Technical Details

### Model Training Configuration
- **Loss function:** CrossEntropyLoss with class weights [1.0, 1.0, 1.2, 1.5, 2.0]
- **Optimizer:** Adam (lr=0.001)
- **Epochs:** 50
- **Batch size:** 32
- **Regularization:** Dropout (0.2), L2 weight decay
- **Class balancing:** WeightedRandomSampler for minority classes

### Why These Choices?
- **Class weights:** Penalize underestimating danger more heavily
- **Conservative bias:** Better false alarm than missed critical condition
- **Custom C inference:** Simpler than TensorFlow Lite, educational, sufficient for small model
- **Float32 (not Int8):** Quantization toolchain had issues; 7.22 KB still fits comfortably

### ESP32 Resource Usage
- **Model weights:** 7.22 KB
- **Inference scratch space:** ~0.5 KB
- **Total RAM:** ~7.64 KB (1.5% of 520 KB available)
- **Processing time:** <10ms per inference
- **Power consumption:** Negligible compared to sensors/WiFi

---

## Known Issues & Limitations

1. **Synthetic data:** Current model trained on rule-based synthetic data, not real measurements
   - **Impact:** 70% accuracy is acceptable for prototype, will improve with real data
   - **Plan:** Retrain after field deployment with actual Kawah Putih measurements

2. **Quantization failed:** Int8 quantization had toolchain issues
   - **Impact:** Using Float32 (4x larger, but still only 7.22 KB)
   - **Status:** Acceptable; may revisit if we need more complex models

3. **Sensor calibration needed:** Different sensor models may need adjustment factors
   - **Plan:** Calibrate against reference instruments during field testing

---

## Safety Considerations

⚠️ **This is a prototype system for research/educational purposes.**

- Model predictions should inform but not replace human judgment
- Conservative bias means expect false alarms (by design)
- Always have backup safety equipment (gas detectors, alarms)
- Validate sensor readings against reference instruments
- Test thoroughly in controlled environment before field deployment

---

## Future Enhancements

- [ ] Retrain with real Kawah Putih sensor data
- [ ] Add time-series analysis for trend prediction
- [ ] Implement OTA (over-the-air) model updates
- [ ] Add more inputs (atmospheric pressure, wind direction, GPS distance from fumarole)
- [ ] Web dashboard for monitoring all 5 devices
- [ ] Data logging to SD card or cloud storage
- [ ] Battery optimization for longer deployment

---

## Team Contacts & Roles

- **ML/Firmware:** [Your Name] - Model training, ESP32 inference implementation
- **Sensors:** [Team Member] - Hardware integration, sensor interface
- **Network:** [Team Member] - ESP-NOW/MQTT communication
- **Field Testing:** [Team Member] - Deployment, data collection, calibration

---

## Questions?

**For ML model questions:** Contact [Your Name]
**For sensor interface:** Check `firmware/main/sensor_interface.c` comments
**For network protocol:** See `firmware/main/network_comm.c` specification

---

## License & Attribution

[Add your institution/project license here]

**References:**
- WHO Air Quality Guidelines (pollutant thresholds)
- Kawah Putih volcanic gas research papers
- ESP-IDF Documentation: https://docs.espressif.com/projects/esp-idf/

---

**Last Updated:** 2025-11-21  
**Project Status:** Active Development  
**Next Milestone:** ESP32 hardware integration & sensor testing