import onnx
import numpy as np

# ============================================================================
# ONNX WEIGHT EXTRACTOR FOR ESP32 DEPLOYMENT
# ============================================================================
# This script extracts trained weights from your ONNX model and converts
# them to C arrays that can be compiled into your ESP-IDF project.
#
# Why we need this:
# - ESP32 can't load ONNX files directly
# - We need weights as static C arrays in flash memory
# - This makes deployment simple: just compile and flash
# ============================================================================

def extract_weights_from_onnx(onnx_path):
    """
    Extract all weights and biases from ONNX model
    
    Returns a dictionary with layer weights and biases
    """
    print("="*70)
    print("EXTRACTING WEIGHTS FROM ONNX MODEL")
    print("="*70)
    
    # Load ONNX model
    model = onnx.load(onnx_path)
    print(f"✓ Loaded ONNX model from '{onnx_path}'")
    
    # Dictionary to store extracted weights
    weights = {}
    
    # Get all initializers (these contain the trained weights)
    # In ONNX, trained parameters are stored as "initializers"
    for initializer in model.graph.initializer:
        name = initializer.name
        
        # Convert ONNX tensor to numpy array
        # ONNX stores data in a specific format, numpy makes it easier to work with
        data = onnx.numpy_helper.to_array(initializer)
        
        weights[name] = data
        print(f"  Found: {name:20s} shape={str(data.shape):15s} dtype={data.dtype}")
    
    print(f"\n✓ Extracted {len(weights)} weight tensors\n")
    
    return weights

def format_array_for_c(array, name, indent=4):
    """
    Format a numpy array as a C array declaration
    
    For example, converts:
        array([1.5, 2.3, -0.8])
    To C code:
        {1.5f, 2.3f, -0.8f}
    
    The 'f' suffix tells C compiler these are float (not double) values
    """
    spaces = " " * indent
    
    # Flatten multi-dimensional arrays to 1D
    # Neural network weights are matrices, but C stores them as 1D arrays
    flat = array.flatten()
    
    # Format each number with 6 decimal places and 'f' suffix
    formatted_values = [f"{val:.6f}f" for val in flat]
    
    # Build C array with proper formatting
    lines = []
    
    # Array values, 8 per line for readability
    for i in range(0, len(formatted_values), 8):
        chunk = formatted_values[i:i+8]
        lines.append(spaces + ", ".join(chunk) + ",")
    
    # Remove trailing comma from last line
    if lines:
        lines[-1] = lines[-1].rstrip(',')
    
    return "\n".join(lines)

def generate_c_header(weights, output_path='model_weights.h'):
    """
    Generate a C header file with all model weights
    
    This creates a .h file that you can #include in your ESP-IDF project
    """
    print("="*70)
    print("GENERATING C HEADER FILE")
    print("="*70)
    
    # Start building the header file content
    header_content = []
    
    # Header guard (prevents multiple inclusion)
    header_content.append("#ifndef MODEL_WEIGHTS_H")
    header_content.append("#define MODEL_WEIGHTS_H")
    header_content.append("")
    header_content.append("// Auto-generated model weights for ESP32 deployment")
    header_content.append("// DO NOT EDIT MANUALLY - Regenerate from ONNX model")
    header_content.append("")
    
    # Model architecture constants
    header_content.append("// Model architecture")
    header_content.append("#define INPUT_SIZE 5")
    header_content.append("#define HIDDEN1_SIZE 64")
    header_content.append("#define HIDDEN2_SIZE 32")
    header_content.append("#define OUTPUT_SIZE 5")
    header_content.append("")
    
    # Extract and format each weight tensor
    # Layer naming convention: fc1 = fully connected layer 1, etc.
    
    # Layer 1: fc1.weight and fc1.bias
    if 'fc1.weight' in weights:
        w = weights['fc1.weight']
        print(f"Processing fc1.weight: shape {w.shape}")
        header_content.append("// Layer 1: Input(5) -> Hidden1(64)")
        header_content.append(f"// Weight matrix: {w.shape[0]}x{w.shape[1]}")
        header_content.append("static const float fc1_weight[] = {")
        header_content.append(format_array_for_c(w, 'fc1_weight'))
        header_content.append("};")
        header_content.append("")
    
    if 'fc1.bias' in weights:
        b = weights['fc1.bias']
        print(f"Processing fc1.bias: shape {b.shape}")
        header_content.append(f"// Bias vector: {b.shape[0]}")
        header_content.append("static const float fc1_bias[] = {")
        header_content.append(format_array_for_c(b, 'fc1_bias'))
        header_content.append("};")
        header_content.append("")
    
    # Layer 2: fc2.weight and fc2.bias
    if 'fc2.weight' in weights:
        w = weights['fc2.weight']
        print(f"Processing fc2.weight: shape {w.shape}")
        header_content.append("// Layer 2: Hidden1(64) -> Hidden2(32)")
        header_content.append(f"// Weight matrix: {w.shape[0]}x{w.shape[1]}")
        header_content.append("static const float fc2_weight[] = {")
        header_content.append(format_array_for_c(w, 'fc2_weight'))
        header_content.append("};")
        header_content.append("")
    
    if 'fc2.bias' in weights:
        b = weights['fc2.bias']
        print(f"Processing fc2.bias: shape {b.shape}")
        header_content.append(f"// Bias vector: {b.shape[0]}")
        header_content.append("static const float fc2_bias[] = {")
        header_content.append(format_array_for_c(b, 'fc2_bias'))
        header_content.append("};")
        header_content.append("")
    
    # Layer 3: fc3.weight and fc3.bias
    if 'fc3.weight' in weights:
        w = weights['fc3.weight']
        print(f"Processing fc3.weight: shape {w.shape}")
        header_content.append("// Layer 3: Hidden2(32) -> Output(5)")
        header_content.append(f"// Weight matrix: {w.shape[0]}x{w.shape[1]}")
        header_content.append("static const float fc3_weight[] = {")
        header_content.append(format_array_for_c(w, 'fc3_weight'))
        header_content.append("};")
        header_content.append("")
    
    if 'fc3.bias' in weights:
        b = weights['fc3.bias']
        print(f"Processing fc3.bias: shape {b.shape}")
        header_content.append(f"// Bias vector: {b.shape[0]}")
        header_content.append("static const float fc3_bias[] = {")
        header_content.append(format_array_for_c(b, 'fc3_bias'))
        header_content.append("};")
        header_content.append("")
    
    # Class names for reference
    header_content.append("// Air quality class names")
    header_content.append("static const char* class_names[] = {")
    header_content.append('    "Normal",')
    header_content.append('    "Caution",')
    header_content.append('    "Warning",')
    header_content.append('    "Danger",')
    header_content.append('    "Critical"')
    header_content.append("};")
    header_content.append("")
    
    # Close header guard
    header_content.append("#endif // MODEL_WEIGHTS_H")
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(header_content))
    
    print(f"\n✓ C header file generated: '{output_path}'")
    
    # Calculate total size
    total_params = sum(w.size for w in weights.values())
    size_kb = total_params * 4 / 1024  # float32 = 4 bytes
    
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Estimated size in flash: {size_kb:.2f} KB")
    print("")

def verify_extraction(weights):
    """
    Verify that extracted weights match expected model architecture
    """
    print("="*70)
    print("VERIFICATION")
    print("="*70)
    
    expected_shapes = {
        'fc1.weight': (64, 5),   # 64 neurons, 5 inputs
        'fc1.bias': (64,),        # 64 biases
        'fc2.weight': (32, 64),   # 32 neurons, 64 inputs
        'fc2.bias': (32,),        # 32 biases
        'fc3.weight': (5, 32),    # 5 outputs, 32 inputs
        'fc3.bias': (5,)          # 5 biases
    }
    
    all_correct = True
    
    for name, expected_shape in expected_shapes.items():
        if name in weights:
            actual_shape = weights[name].shape
            if actual_shape == expected_shape:
                print(f"✓ {name:15s}: {str(actual_shape):15s} (correct)")
            else:
                print(f"✗ {name:15s}: {str(actual_shape):15s} (expected {expected_shape})")
                all_correct = False
        else:
            print(f"✗ {name:15s}: MISSING")
            all_correct = False
    
    print("")
    if all_correct:
        print("✓ All weights verified successfully!")
    else:
        print("✗ Warning: Some weights are missing or have wrong shape")
    
    print("")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Path to your ONNX model
    onnx_model_path = 'models/air_quality_model.onnx'
    
    # Output C header file
    output_header = 'model_weights.h'
    
    print("\n" + "="*70)
    print("ONNX TO C WEIGHT EXTRACTION FOR ESP32")
    print("="*70)
    print(f"Input:  {onnx_model_path}")
    print(f"Output: {output_header}")
    print("")
    
    try:
        # Step 1: Extract weights from ONNX
        weights = extract_weights_from_onnx(onnx_model_path)
        
        # Step 2: Verify extraction
        verify_extraction(weights)
        
        # Step 3: Generate C header file
        generate_c_header(weights, output_header)
        
        print("="*70)
        print("SUCCESS!")
        print("="*70)
        print(f"\nYour model weights are ready for ESP32 deployment!")
        print(f"Next steps:")
        print(f"  1. Copy '{output_header}' to your ESP-IDF project")
        print(f"  2. #include \"{output_header}\" in your inference code")
        print(f"  3. Use the weight arrays in your forward pass function")
        print("\n" + "="*70 + "\n")
        
    except FileNotFoundError:
        print(f"\n✗ Error: '{onnx_model_path}' not found!")
        print("  Make sure you've run the ONNX conversion script first.")
    except Exception as e:
        print(f"\n✗ Error: {e}")

if __name__ == '__main__':
    main()