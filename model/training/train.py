import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from model.models.cnn_model import CryingSenseCNN
import matplotlib.pyplot as plt

# Custom Dataset for loading .npy feature files
def get_label_from_path(path):
    # Assumes path like .../class_name/xxx.npy
    return os.path.basename(os.path.dirname(path))

class CryingSenseDataset(Dataset):
    def __init__(self, file_list, label_map, feature_base_dir, augment=False):
        self.file_list = file_list  # List of MFCC file paths (used as reference)
        self.label_map = label_map
        self.feature_base_dir = feature_base_dir
        self.augment = augment
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # Load all feature types and combine them
        mfcc_path = self.file_list[idx]
        
        # Construct paths for other features (same relative path, different feature dir)
        rel_path = os.path.relpath(mfcc_path, os.path.join(self.feature_base_dir, 'mfcc'))
        mel_path = os.path.join(self.feature_base_dir, 'mel_spectrogram', rel_path)
        chroma_path = os.path.join(self.feature_base_dir, 'chroma', rel_path)
        
        # Load features
        mfcc = np.load(mfcc_path)
        mel = np.load(mel_path)
        chroma = np.load(chroma_path)
        
        # Combine features into 4-channel input
        x = self._combine_features(mfcc, mel, chroma)
        x = torch.tensor(x, dtype=torch.float32)
        
        # Apply data augmentation during training
        if self.augment:
            x = self._augment_features(x)
        
        label_name = get_label_from_path(mfcc_path)
        y = self.label_map[label_name]
        return x, y
    
    def _combine_features(self, mfcc, mel, chroma):
        """Combine multiple features into a 4-channel array."""
        # Get target dimensions
        target_height = max(mfcc.shape[0], mel.shape[0], chroma.shape[0])
        target_width = mfcc.shape[1]  # Time steps should be the same
        
        # Pad features to target height
        mfcc_padded = self._pad_feature(mfcc, (target_height, target_width))
        mel_padded = self._pad_feature(mel, (target_height, target_width))
        chroma_padded = self._pad_feature(chroma, (target_height, target_width))
        
        # Calculate delta MFCC
        delta_mfcc = self._compute_delta(mfcc)
        delta_mfcc_padded = self._pad_feature(delta_mfcc, (target_height, target_width))
        
        # Stack into 4-channel array (channels, height, width)
        combined = np.stack([
            mfcc_padded,
            mel_padded,
            chroma_padded,
            delta_mfcc_padded
        ], axis=0)
        
        return combined
    
    def _pad_feature(self, feature, target_shape):
        """Pad feature to target shape."""
        padded = np.zeros(target_shape, dtype=feature.dtype)
        min_h = min(feature.shape[0], target_shape[0])
        min_w = min(feature.shape[1], target_shape[1])
        padded[:min_h, :min_w] = feature[:min_h, :min_w]
        return padded
    
    def _compute_delta(self, feature):
        """Compute delta (first derivative) of feature."""
        # Simple delta: difference between adjacent frames
        delta = np.zeros_like(feature)
        delta[:, 1:] = feature[:, 1:] - feature[:, :-1]
        return delta
    
    def _augment_features(self, features):
        """Apply data augmentation to features."""
        # Random noise addition (10% chance)
        if torch.rand(1) < 0.1:
            noise = torch.randn_like(features) * 0.01
            features = features + noise
        
        # Random time shift (20% chance)
        if torch.rand(1) < 0.2:
            shift = torch.randint(-10, 10, (1,)).item()
            features = torch.roll(features, shift, dims=-1)
        
        # Random amplitude scaling (20% chance)
        if torch.rand(1) < 0.2:
            scale = torch.rand(1) * 0.4 + 0.8  # 0.8 to 1.2
            features = features * scale
        
        return features

def get_file_list_and_labels(feature_base_dir):
    """Get file list from MFCC directory (used as reference for all features)."""
    mfcc_dir = os.path.join(feature_base_dir, 'mfcc')
    
    if not os.path.exists(mfcc_dir):
        return [], {}
    
    file_list = []
    for root, _, files in os.walk(mfcc_dir):
        for file in files:
            if file.endswith('.npy'):
                file_list.append(os.path.join(root, file))
    
    if not file_list:
        return [], {}
    
    labels = sorted(list(set(get_label_from_path(f) for f in file_list)))
    label_map = {label: i for i, label in enumerate(labels)}
    return file_list, label_map

def train_model(model, train_loader, val_loader, device, epochs=50, lr=1e-3, 
                patience=10, save_dir='../saved_models'):
    """
    Train the model with early stopping, learning rate scheduling, and comprehensive metrics.
    
    Args:
        model: CNN model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: torch device (cuda/cpu)
        epochs: Maximum number of epochs
        lr: Initial learning rate
        patience: Early stopping patience
        save_dir: Directory to save model checkpoints
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler: ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Early stopping variables
    best_val_acc = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*60)
    print("Starting Training")
    print("="*60)
    print(f"Device: {device}")
    print(f"Initial Learning Rate: {lr}")
    print(f"Max Epochs: {epochs}")
    print(f"Early Stopping Patience: {patience}")
    print("="*60)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x.size(0)
            _, pred = out.max(1)
            train_correct += (pred == y).sum().item()
            train_total += x.size(0)
        
        train_acc = train_correct / train_total
        train_loss = train_loss / train_total
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                
                val_loss += loss.item() * x.size(0)
                _, pred = out.max(1)
                val_correct += (pred == y).sum().item()
                val_total += x.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        val_acc = val_correct / val_total
        val_loss = val_loss / val_total
        
        # Calculate per-class metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Val Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Early stopping and model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            epochs_without_improvement = 0
            
            # Save best model
            model_path = os.path.join(save_dir, 'cryingsense_cnn_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss
            }, model_path)
            print(f"  ✓ Best model saved (Val Acc: {val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")
            
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation accuracy: {best_val_acc:.4f}")
                break
        
        print("-"*60)
    
    # Save training history
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plot_training_history(history, save_dir)
    
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_dir}/cryingsense_cnn_best.pth")
    print(f"Training history saved to: {history_path}")
    print("="*60)
    
    return history


def plot_training_history(history, save_dir):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy curves
    axes[0, 1].plot(history['train_acc'], label='Train Acc')
    axes[0, 1].plot(history['val_acc'], label='Val Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate
    axes[1, 0].plot(history['learning_rates'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True)
    
    # Val accuracy with best marker
    axes[1, 1].plot(history['val_acc'], marker='o', markersize=3)
    best_epoch = np.argmax(history['val_acc'])
    axes[1, 1].axvline(x=best_epoch, color='r', linestyle='--', 
                       label=f'Best (Epoch {best_epoch+1})')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Validation Accuracy')
    axes[1, 1].set_title('Validation Accuracy Progress')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Training curves saved to: {plot_path}")

if __name__ == "__main__":
    # Configuration
    feature_base_dir = "../../dataset/processed/feature_extraction/cleaned"
    save_dir = "../saved_models"
    
    print("="*60)
    print("CryingSense CNN Training")
    print("="*60)
    
    # Load data
    print("Loading dataset...")
    file_list, label_map = get_file_list_and_labels(feature_base_dir)
    
    if not file_list:
        print("Error: No feature files found!")
        print(f"Looking in: {os.path.abspath(feature_base_dir)}")
        print("\nPlease run feature extraction first:")
        print("  python scripts/feature_extraction.py")
        sys.exit(1)
    
    print(f"Total files: {len(file_list)}")
    print(f"Classes: {list(label_map.keys())}")
    
    # Split dataset (80% train, 20% val for now; proper split should use dataset_split.json)
    train_files, val_files = train_test_split(
        file_list, test_size=0.20, random_state=42, 
        stratify=[get_label_from_path(f) for f in file_list]
    )
    
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    
    # Create datasets with augmentation for training
    train_dataset = CryingSenseDataset(train_files, label_map, feature_base_dir, augment=True)
    val_dataset = CryingSenseDataset(val_files, label_map, feature_base_dir, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, 
                             num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, 
                           num_workers=2, pin_memory=True)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CryingSenseCNN(num_classes=len(label_map), dropout_rate=0.3).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Architecture: CryingSenseCNN")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (fp32)")
    print("="*60)
    
    # Train model
    history = train_model(model, train_loader, val_loader, device, 
                         epochs=50, lr=1e-3, patience=10, save_dir=save_dir)
