"""
CryingSense Model Training Script

Trains the CNN on the TRAIN split (from dataset_split.json).
The VAL split is used internally for early stopping and LR scheduling.
Outputs: saved model weights + training curves/history in performance_reports/training_report/

Pipeline order:
  1. python scripts/preprocess_audio.py
  2. python scripts/feature_extraction.py
  3. python scripts/dataset_split.py
  4. python model/training/train.py       <- this script
  5. python model/training/validate.py
  6. python model/training/evaluate.py
"""

import os
import sys
import json
import logging
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from model.models.cnn_model import CryingSenseCNN
from model.training.dataset import CryingSenseDataset, get_label_from_path
import matplotlib.pyplot as plt


def load_split_from_json(json_path, feature_base_dirs):
    """
    Load dataset split from JSON file.
    
    Args:
        json_path: Path to dataset_split.json
        feature_base_dirs: Dict mapping source names ('cleaned', 'raw') to base directories
        
    Returns:
        Dict with 'train', 'val', 'eval' keys containing lists of (mfcc_path, base_dir) tuples
    """
    with open(json_path, 'r') as f:
        split_data = json.load(f)
    
    result = {'train': [], 'val': [], 'eval': []}
    
    # Handle nested structure: {"splits": {"train": {"class_name": [files]}}}
    splits = split_data.get('splits', split_data)  # Support both formats
    
    for split_name in ['train', 'val', 'eval']:
        split_content = splits.get(split_name, {})
        
        # If split_content is a dict (organized by class), flatten it
        if isinstance(split_content, dict):
            for class_name, file_list in split_content.items():
                for entry in file_list:
                    # Entry format: "source:filename.npy"
                    if ':' in entry:
                        source, filename = entry.split(':', 1)
                    else:
                        source = 'cleaned'
                        filename = entry
                    
                    if source not in feature_base_dirs:
                        print(f"Warning: Unknown source '{source}' in split, skipping {entry}")
                        continue
                    
                    base_dir = feature_base_dirs[source]
                    mfcc_path = os.path.join(base_dir, 'mfcc', class_name, filename)
                    
                    if os.path.exists(mfcc_path):
                        result[split_name].append((mfcc_path, base_dir))
                    else:
                        print(f"Warning: Feature file not found: {mfcc_path}")
        else:
            # Legacy flat list format
            for entry in split_content:
                if ':' in entry:
                    source, filename = entry.split(':', 1)
                else:
                    source = 'cleaned'
                    filename = entry
                
                if source not in feature_base_dirs:
                    continue
                
                base_dir = feature_base_dirs[source]
                class_name = filename.rsplit('_', 1)[0] if '_' in filename else filename.replace('.npy', '')
                mfcc_path = os.path.join(base_dir, 'mfcc', class_name, filename)
                
                if os.path.exists(mfcc_path):
                    result[split_name].append((mfcc_path, base_dir))
    
    return result


def get_feature_file_list(feature_base_dir):
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
                patience=10, save_dir='../saved_models', training_report_dir=None):
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
        save_dir: Directory to save model checkpoints (.pth files)
        training_report_dir: Directory to save training reports (curves, history)
    """
    # Use training_report_dir if provided, otherwise fall back to save_dir
    report_dir = training_report_dir if training_report_dir else save_dir
    logger = logging.getLogger(__name__)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler: ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
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
    
    logger.info("Starting Training")
    logger.info(f"Device: {device}")
    logger.info(f"Initial Learning Rate: {lr}")
    logger.info(f"Max Epochs: {epochs}")
    logger.info(f"Early Stopping Patience: {patience}")
    
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
        
        # Print and log epoch results
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Val Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
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
            logger.info(f"Best model saved at epoch {epoch+1} with Val Acc: {val_acc:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")
            
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation accuracy: {best_val_acc:.4f}")
                logger.info(f"Early stopping triggered after {epoch+1} epochs. Best Val Acc: {best_val_acc:.4f}")
                break
        
        print("-"*60)
    
    # Save training history to training_report directory
    history_path = os.path.join(report_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves to training_report directory
    plot_training_history(history, report_dir)
    
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_dir}/cryingsense_cnn_best.pth")
    print(f"Training history saved to: {history_path}")
    print("="*60)
    
    logger.info("Training Complete")
    logger.info(f"Best Validation Accuracy: {best_val_acc:.4f}")
    logger.info(f"Best Validation Loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to: {save_dir}/cryingsense_cnn_best.pth")
    logger.info(f"Training history saved to: {history_path}")
    
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
    # Get the project root directory (2 levels up from this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))
    
    # Feature directories for both cleaned and raw data
    feature_base_dirs = {
        'cleaned': os.path.join(project_root, 'dataset', 'processed', 'features', 'cleaned'),
        'raw': os.path.join(project_root, 'dataset', 'processed', 'features', 'raw')
    }
    save_dir = os.path.join(project_root, 'model', 'saved_models')
    training_report_dir = os.path.join(project_root, 'performance_reports', 'training_report')
    logs_dir = os.path.join(project_root, 'performance_reports', 'logs')
    split_json_path = os.path.join(project_root, 'dataset', 'dataset_split.json')
    
    # Create directories
    os.makedirs(training_report_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(logs_dir, f'train_{timestamp}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("CryingSense CNN Training")
    logger.info("="*60)
    
    print("="*60)
    print("CryingSense CNN Training")
    print("="*60)
    
    # Try to load from dataset_split.json first
    if os.path.exists(split_json_path):
        print(f"Loading splits from: {split_json_path}")
        splits = load_split_from_json(split_json_path, feature_base_dirs)
        train_files = splits['train']
        val_files = splits['val']
        
        if not train_files:
            print("Error: No training files found in dataset_split.json!")
            print("Please run: python scripts/dataset_split.py")
            sys.exit(1)
        
        # Build label map from training files
        labels = sorted(list(set(get_label_from_path(f[0]) for f in train_files)))
        label_map = {label: i for i, label in enumerate(labels)}
        
        print(f"Loaded from JSON - Train: {len(train_files)}, Val: {len(val_files)}")
        print(f"Classes: {list(label_map.keys())}")
    else:
        print("Error: dataset_split.json not found!")
        print("Dataset splitting is handled by a separate script.")
        print("Please run: python scripts/dataset_split.py")
        sys.exit(1)
    
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    
    # Create datasets with augmentation for training
    train_dataset = CryingSenseDataset(train_files, label_map, feature_base_dirs, augment=True)
    val_dataset = CryingSenseDataset(val_files, label_map, feature_base_dirs, augment=False)
    
    # Create data loaders
    # Note: num_workers=0 for Windows compatibility; increase on Linux for better performance
    use_pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, 
                             num_workers=0, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, 
                           num_workers=0, pin_memory=use_pin_memory)
    
    # Initialize model
    print(f"\n PyTorch version: {torch.__version__}")
    print(f" CUDA available: {torch.cuda.is_available()}")
    print(f" CUDA version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f" GPU: {torch.cuda.get_device_name(0)}")
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
    
    # Train model (save_dir is for model weights, training_report_dir is for training outputs)
    history = train_model(model, train_loader, val_loader, device, 
                         epochs=50, lr=1e-3, patience=10, save_dir=save_dir, 
                         training_report_dir=training_report_dir)
