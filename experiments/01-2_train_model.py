import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
# This ensures you get the same results every time you run the code
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# STEP 1: CUSTOM DATASET CLASS
# ============================================================================
# PyTorch needs data in a specific format. We create a custom Dataset class
# that loads our CSV data and converts it to PyTorch tensors.

class AirQualityDataset(Dataset):
    """
    Custom Dataset for air quality data.
    
    PyTorch's Dataset class requires three methods:
    - __init__: Initialize and load data
    - __len__: Return the number of samples
    - __getitem__: Return a single sample by index
    """
    def __init__(self, csv_file):
        # Load CSV file
        df = pd.read_csv(csv_file)
        
        # Separate features (X) and labels (y)
        # Features: the 5 sensor readings
        # Labels: the class (0-4)
        self.X = df[['h2s', 'so2', 'wind_speed', 'temperature', 'humidity']].values
        self.y = df['label'].values
        
        # Convert to PyTorch tensors
        # float32 is standard for neural network inputs
        # long (int64) is required for classification labels
        self.X = torch.FloatTensor(self.X)
        self.y = torch.LongTensor(self.y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Return a tuple of (features, label) for the given index
        return self.X[idx], self.y[idx]


# ============================================================================
# STEP 2: NEURAL NETWORK ARCHITECTURE
# ============================================================================
# We design a small, efficient network suitable for ESP32 deployment.
# Remember our earlier calculation: keeping it small is crucial for RAM constraints.

class AirQualityClassifier(nn.Module):
    """
    Compact neural network optimized for ESP32 deployment.
    
    Architecture:
    - Input: 5 features (H2S, SO2, wind_speed, temperature, humidity)
    - Hidden Layer 1: 64 neurons (reduced from 128 to save memory)
    - Hidden Layer 2: 32 neurons
    - Output: 5 classes (Normal, Caution, Warning, Danger, Critical)
    
    Design decisions for ESP32:
    - Smaller layers = less RAM during inference
    - ReLU activation = computationally cheap
    - Dropout = prevents overfitting, improves generalization
    """
    def __init__(self, input_size=5, hidden1=64, hidden2=32, num_classes=5):
        super(AirQualityClassifier, self).__init__()
        
        # Layer 1: Input -> Hidden1
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)  # Randomly drop 30% of neurons during training
        
        # Layer 2: Hidden1 -> Hidden2
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        # Layer 3: Hidden2 -> Output
        self.fc3 = nn.Linear(hidden2, num_classes)
        
    def forward(self, x):
        """
        Forward pass: how data flows through the network.
        This is called automatically during training and inference.
        """
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x
    
    def count_parameters(self):
        """Calculate total number of trainable parameters (weights + biases)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# STEP 3: CALCULATE CLASS WEIGHTS FOR IMBALANCED DATA
# ============================================================================
# This implements our "Option 1" - telling the model that some classes
# are more important than others.

def calculate_class_weights(dataset, safety_bias=2.0):
    """
    Calculate class weights with safety bias.
    
    Strategy:
    1. Inverse frequency weighting: rare classes get higher weights
    2. Safety multiplier: dangerous classes (3, 4) get extra weight
    
    Args:
        dataset: The training dataset
        safety_bias: Multiplier for dangerous classes (Danger=3, Critical=4)
    
    Returns:
        Tensor of weights for each class
    """
    # Count samples per class
    labels = dataset.y.numpy()
    class_counts = np.bincount(labels)
    
    # Calculate inverse frequency weights
    # Rare classes get higher weights to balance the dataset
    total_samples = len(labels)
    weights = total_samples / (len(class_counts) * class_counts)
    
    # Apply safety bias to dangerous classes
    # This is your "Option C" - heavily penalize underestimating danger
    weights[3] *= safety_bias  # Danger class
    weights[4] *= safety_bias  # Critical class
    
    print("\n" + "="*60)
    print("CLASS WEIGHTS (higher = more important)")
    print("="*60)
    class_names = ['Normal', 'Caution', 'Warning', 'Danger', 'Critical']
    for i, (name, count, weight) in enumerate(zip(class_names, class_counts, weights)):
        print(f"{name:12s}: {count:4d} samples, weight = {weight:.3f}")
    
    return torch.FloatTensor(weights)


# ============================================================================
# STEP 4: CREATE WEIGHTED SAMPLER FOR OVERSAMPLING
# ============================================================================
# This implements our "Option 2" - giving the model more examples of
# rare classes during training.

def create_weighted_sampler(dataset):
    """
    Create a sampler that oversamples minority classes.
    
    How it works:
    - Each sample gets a weight based on its class rarity
    - During training, rare classes are randomly selected more often
    - This balances what the model sees, even with imbalanced data
    """
    labels = dataset.y.numpy()
    class_counts = np.bincount(labels)
    
    # Each sample's weight is inversely proportional to its class frequency
    sample_weights = 1.0 / class_counts[labels]
    sample_weights = torch.DoubleTensor(sample_weights)
    
    # WeightedRandomSampler draws samples according to these weights
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # Allow sampling the same example multiple times
    )
    
    return sampler


# ============================================================================
# STEP 5: TRAINING FUNCTION
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch (one pass through the entire dataset).
    
    Training loop:
    1. Get a batch of data
    2. Forward pass: model makes predictions
    3. Calculate loss: how wrong are the predictions?
    4. Backward pass: calculate gradients
    5. Update weights: adjust model to reduce loss
    """
    model.train()  # Set model to training mode (enables dropout)
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the gradients from the previous batch
        optimizer.zero_grad()
        
        # Forward pass: get model predictions
        outputs = model(inputs)
        
        # Calculate loss with class weights
        loss = criterion(outputs, labels)
        
        # Backward pass: calculate gradients
        loss.backward()
        
        # Update model weights
        optimizer.step()
        
        # Track statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


# ============================================================================
# STEP 6: EVALUATION FUNCTION
# ============================================================================

def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on validation/test data.
    
    Key difference from training:
    - model.eval() disables dropout
    - torch.no_grad() saves memory by not computing gradients
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # Don't compute gradients during evaluation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy, all_preds, all_labels


# ============================================================================
# STEP 7: CONFUSION MATRIX VISUALIZATION
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Create a confusion matrix to see exactly where the model makes mistakes.
    
    Reading the matrix:
    - Rows = true labels
    - Columns = predicted labels
    - Diagonal = correct predictions
    - Off-diagonal = mistakes
    
    For safety analysis:
    - Below diagonal = overestimation (false alarms) - safer
    - Above diagonal = underestimation (missed danger) - DANGEROUS
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Air Quality Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved as 'confusion_matrix.png'")


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    # Set device (CPU for now, ESP32 will also use CPU inference)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    BATCH_SIZE = 32      # Number of samples processed together
    LEARNING_RATE = 0.001  # How big the weight update steps are
    NUM_EPOCHS = 50      # How many times to go through the entire dataset
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = AirQualityDataset('air_quality_train.csv')
    test_dataset = AirQualityDataset('air_quality_test.csv')
    
    # Calculate class weights for loss function
    class_weights = calculate_class_weights(train_dataset, safety_bias=2.0)
    class_weights = class_weights.to(device)
    
    # Create weighted sampler for oversampling
    train_sampler = create_weighted_sampler(train_dataset)
    
    # Create data loaders
    # Training loader uses weighted sampler for oversampling
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             sampler=train_sampler)
    # Test loader just goes through data sequentially
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = AirQualityClassifier(input_size=5, hidden1=64, hidden2=32, num_classes=5)
    model = model.to(device)
    
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")
    print(f"Estimated model size (float32): {model.count_parameters() * 4 / 1024:.2f} KB")
    print(f"Estimated model size (int8 quantized): {model.count_parameters() / 1024:.2f} KB")
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer (Adam is a good default choice)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    best_test_acc = 0
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(NUM_EPOCHS):
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate on test set
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
        
        # Store metrics for plotting
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    print(f"\nBest Test Accuracy: {best_test_acc:.2f}%")
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Final evaluation with confusion matrix
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)
    
    class_names = ['Normal', 'Caution', 'Warning', 'Danger', 'Critical']
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("\nTraining curves saved as 'training_curves.png'")
    
    # Save final model
    torch.save(model.state_dict(), 'air_quality_model.pth')
    print("\nFinal model saved as 'air_quality_model.pth'")
    
    # Save entire model for ONNX export (next step)
    torch.save(model, 'air_quality_model_complete.pth')
    print("Complete model saved as 'air_quality_model_complete.pth'")

if __name__ == "__main__":
    main()