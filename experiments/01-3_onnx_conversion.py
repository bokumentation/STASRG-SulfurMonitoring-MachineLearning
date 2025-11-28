import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# STEP 1: DEFINE MODEL ARCHITECTURES
# ============================================================================
# We need TWO versions of the model:
# 1. Training version (with Dropout) - to load trained weights
# 2. Inference version (without Dropout) - for ONNX export

class AirQualityNet(nn.Module):
    """Training version with Dropout"""
    
    def __init__(self, input_size=5, hidden1=64, hidden2=32, num_classes=5):
        super(AirQualityNet, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden2, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x


class AirQualityNetInference(nn.Module):
    """
    Inference version WITHOUT Dropout
    
    Why remove Dropout?
    - Dropout is only used during training to prevent overfitting
    - During inference, we want deterministic predictions
    - model.eval() disables Dropout, but the layers still exist in the graph
    - Removing them completely makes ONNX export cleaner and quantization easier
    - Also slightly reduces inference time on ESP32
    """
    
    def __init__(self, input_size=5, hidden1=64, hidden2=32, num_classes=5):
        super(AirQualityNetInference, self).__init__()
        
        # Same layers, but NO Dropout
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        # No dropout1 here!
        
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        # No dropout2 here!
        
        self.fc3 = nn.Linear(hidden2, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        # No dropout operation
        
        x = self.fc2(x)
        x = self.relu2(x)
        # No dropout operation
        
        x = self.fc3(x)
        return x


def copy_weights(source_model, target_model):
    """
    Copy weights from training model to inference model
    
    Since both models have the same Linear layers (fc1, fc2, fc3),
    we can directly copy the trained weights.
    We just skip the Dropout layers since they don't have weights.
    """
    # Get state dictionaries (weight storage)
    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    
    # Copy only the Linear layer weights (fc1, fc2, fc3)
    # These keys exist in both models
    for key in target_state.keys():
        if key in source_state:
            target_state[key] = source_state[key]
    
    # Load the copied weights
    target_model.load_state_dict(target_state)
    
    return target_model

# ============================================================================
# STEP 2: LOAD TRAINED MODEL AND PREPARE INFERENCE VERSION
# ============================================================================

print("="*70)
print("STEP 1: LOADING TRAINED MODEL & CREATING INFERENCE VERSION")
print("="*70)

# Load training model with Dropout
model_with_dropout = AirQualityNet()

try:
    model_with_dropout.load_state_dict(torch.load('best_model.pth'))
    print("✓ Loaded trained weights from 'best_model.pth'")
except FileNotFoundError:
    print("✗ Error: 'best_model.pth' not found!")
    print("  Make sure you've run the training script first.")
    exit(1)

# Create inference model without Dropout
model = AirQualityNetInference()
print("✓ Created inference model (without Dropout)")

# Copy weights from training model to inference model
model = copy_weights(model_with_dropout, model)
print("✓ Copied trained weights to inference model")

# Set to evaluation mode
model.eval()
print(f"✓ Model set to evaluation mode")
print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# ============================================================================
# STEP 3: VERIFY INFERENCE MODEL MATCHES TRAINING PERFORMANCE
# ============================================================================

print("="*70)
print("STEP 2: VERIFYING INFERENCE MODEL (PyTorch)")
print("="*70)

# Load test data
df_test = pd.read_csv('air_quality_test.csv')
X_test = df_test[['h2s', 'so2', 'wind_speed', 'temperature', 'humidity']].values
y_test = df_test['label'].values

print(f"Test samples: {len(X_test)}")

# Test PyTorch inference model
X_test_tensor = torch.FloatTensor(X_test)
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, y_pred_pytorch = torch.max(outputs, 1)
    y_pred_pytorch = y_pred_pytorch.numpy()

pytorch_accuracy = accuracy_score(y_test, y_pred_pytorch)
print(f"✓ PyTorch Inference Model Accuracy: {pytorch_accuracy:.4f} ({pytorch_accuracy*100:.2f}%)")
print("  (Should match training accuracy of ~70%)\n")

# ============================================================================
# STEP 4: EXPORT TO ONNX FORMAT
# ============================================================================

print("="*70)
print("STEP 3: EXPORTING TO ONNX FORMAT")
print("="*70)

# Create a dummy input tensor
dummy_input = torch.randn(1, 5)
print(f"Dummy input shape: {dummy_input.shape}")

# Export to ONNX
onnx_model_path = 'air_quality_model.onnx'

# Use opset_version=13 for better compatibility with quantization
# Opset 13 has better support for dynamic quantization
torch.onnx.export(
    model,
    dummy_input,
    onnx_model_path,
    export_params=True,
    opset_version=13,              # Changed from 11 to 13
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"✓ ONNX model exported to '{onnx_model_path}'")

# Verify the ONNX model
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
print(f"✓ ONNX model verification passed")

# Display ONNX model info
import os
onnx_size_kb = os.path.getsize(onnx_model_path) / 1024
print(f"✓ ONNX model size: {onnx_size_kb:.2f} KB\n")

# ============================================================================
# STEP 5: TEST ONNX MODEL (BEFORE QUANTIZATION)
# ============================================================================

print("="*70)
print("STEP 4: TESTING ONNX MODEL (Float32)")
print("="*70)

# Create ONNX Runtime session
ort_session = ort.InferenceSession(onnx_model_path)

# Get input name
input_name = ort_session.get_inputs()[0].name

# Run inference on test set
y_pred_onnx = []
for i in range(len(X_test)):
    input_data = X_test[i:i+1].astype(np.float32)
    outputs = ort_session.run(None, {input_name: input_data})
    pred = np.argmax(outputs[0], axis=1)[0]
    y_pred_onnx.append(pred)

# Calculate accuracy
onnx_accuracy = accuracy_score(y_test, y_pred_onnx)
print(f"✓ ONNX Float32 Accuracy: {onnx_accuracy:.4f} ({onnx_accuracy*100:.2f}%)")
print("  (Should match PyTorch accuracy)\n")

# ============================================================================
# STEP 6: QUANTIZE TO INT8
# ============================================================================

print("="*70)
print("STEP 5: QUANTIZING TO INT8")
print("="*70)

quantized_model_path = 'air_quality_model_int8.onnx'

print("Applying dynamic quantization...")
print("  - Converting Float32 weights → Int8")
print("  - Keeping activations in Float32")
print("  - This reduces model size by ~4x\n")

try:
    quantize_dynamic(
        model_input=onnx_model_path,
        model_output=quantized_model_path,
        weight_type=QuantType.QUInt8,
        # Optimize for size (important for ESP32)
        optimize_model=True
    )
    print(f"✓ Quantized model saved to '{quantized_model_path}'")
except Exception as e:
    print(f"✗ Quantization failed: {e}")
    print("\nTrying alternative quantization method...")
    
    # Alternative: Use QInt8 instead of QUInt8
    quantize_dynamic(
        model_input=onnx_model_path,
        model_output=quantized_model_path,
        weight_type=QuantType.QInt8
    )
    print(f"✓ Quantized model saved using QInt8")

# Compare file sizes
quantized_size_kb = os.path.getsize(quantized_model_path) / 1024
compression_ratio = onnx_size_kb / quantized_size_kb

print(f"\n{'Model':<20} {'Size (KB)':<15} {'Compression'}")
print("-" * 50)
print(f"{'Float32 (original)':<20} {onnx_size_kb:<15.2f} {'1.00x'}")
print(f"{'Int8 (quantized)':<20} {quantized_size_kb:<15.2f} {compression_ratio:.2f}x")
print(f"\n✓ Size reduction: {((1 - quantized_size_kb/onnx_size_kb) * 100):.1f}%\n")

# ============================================================================
# STEP 7: TEST QUANTIZED MODEL
# ============================================================================

print("="*70)
print("STEP 6: TESTING QUANTIZED MODEL (Int8)")
print("="*70)

# Create ONNX Runtime session for quantized model
ort_session_quantized = ort.InferenceSession(quantized_model_path)

# Get input name
input_name_q = ort_session_quantized.get_inputs()[0].name

# Run inference on test set
y_pred_quantized = []
for i in range(len(X_test)):
    input_data = X_test[i:i+1].astype(np.float32)
    outputs = ort_session_quantized.run(None, {input_name_q: input_data})
    pred = np.argmax(outputs[0], axis=1)[0]
    y_pred_quantized.append(pred)

# Calculate accuracy
quantized_accuracy = accuracy_score(y_test, y_pred_quantized)
accuracy_drop = (onnx_accuracy - quantized_accuracy) * 100

print(f"✓ Quantized Int8 Accuracy: {quantized_accuracy:.4f} ({quantized_accuracy*100:.2f}%)")
print(f"✓ Accuracy drop from quantization: {accuracy_drop:.2f} percentage points")

if abs(accuracy_drop) < 0.5:
    print("  → Excellent! Minimal accuracy loss from quantization\n")
elif abs(accuracy_drop) < 3:
    print("  → Good! Acceptable accuracy loss for 4x size reduction\n")
else:
    print("  → Note: Consider if accuracy trade-off is acceptable\n")

# ============================================================================
# STEP 8: DETAILED COMPARISON
# ============================================================================

print("="*70)
print("STEP 7: DETAILED PERFORMANCE COMPARISON")
print("="*70)

class_names = ['Normal', 'Caution', 'Warning', 'Danger', 'Critical']

print("\n--- Float32 Model ---")
print(classification_report(y_test, y_pred_onnx, target_names=class_names, digits=3))

print("\n--- Int8 Quantized Model ---")
print(classification_report(y_test, y_pred_quantized, target_names=class_names, digits=3))

# Confusion matrix for quantized model
cm = confusion_matrix(y_test, y_pred_quantized)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Quantized Int8 Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_quantized.png', dpi=300, bbox_inches='tight')
print("\n✓ Confusion matrix saved to 'confusion_matrix_quantized.png'")

# ============================================================================
# STEP 9: ESP32 DEPLOYMENT READINESS CHECK
# ============================================================================

print("\n" + "="*70)
print("ESP32 DEPLOYMENT READINESS")
print("="*70)

print(f"\n📊 Model Performance:")
print(f"  PyTorch Inference:      {pytorch_accuracy*100:.2f}%")
print(f"  ONNX Float32:           {onnx_accuracy*100:.2f}%")
print(f"  ONNX Int8 (quantized):  {quantized_accuracy*100:.2f}%")
print(f"  Accuracy drop:          {accuracy_drop:.2f} percentage points")

print(f"\n💾 Model Size:")
print(f"  Float32:  {onnx_size_kb:.2f} KB")
print(f"  Int8:     {quantized_size_kb:.2f} KB  ({compression_ratio:.2f}x smaller)")

# Estimate RAM usage during inference
# Rough estimate: model weights + intermediate activations
input_size = 5
hidden1_size = 64
hidden2_size = 32
output_size = 5

# Worst case: all activations in memory simultaneously
activation_memory = (input_size + hidden1_size + hidden2_size + output_size) * 4 / 1024  # KB (float32)
total_ram_estimate = quantized_size_kb + activation_memory

print(f"\n🔧 Estimated ESP32 RAM Usage:")
print(f"  Model weights (int8):     {quantized_size_kb:.2f} KB")
print(f"  Activations (float32):    ~{activation_memory:.2f} KB")
print(f"  Total estimate:           ~{total_ram_estimate:.2f} KB")
print(f"  ESP32 available RAM:      ~520 KB")
print(f"  RAM usage:                {(total_ram_estimate/520)*100:.1f}% of available")

if total_ram_estimate < 50:
    print("  ✓ Excellent! Model will fit comfortably on ESP32")
elif total_ram_estimate < 100:
    print("  ✓ Good! Sufficient RAM for model + application code")
else:
    print("  ⚠ May need optimization for production deployment")

print(f"\n✅ Files Generated:")
print(f"  1. {onnx_model_path} - ONNX Float32 model")
print(f"  2. {quantized_model_path} - ONNX Int8 quantized model (USE THIS FOR ESP32)")
print(f"  3. confusion_matrix_quantized.png - Performance visualization")

print(f"\n🚀 Next Steps:")
print(f"  1. The Int8 model is ready for ESP32 deployment")
print(f"  2. You'll need an ONNX Runtime library for ESP32")
print(f"  3. Set up ESP-IDF project with ONNX inference")
print(f"  4. Implement sensor data preprocessing")
print(f"  5. Add network communication between ESP32 devices")

print("\n" + "="*70)