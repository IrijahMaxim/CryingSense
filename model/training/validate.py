import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from model.models.cnn_model import CryingSenseCNN
from sklearn.metrics import classification_report, confusion_matrix


def get_label_from_path(path):
    """Extract class label from file path."""
    return os.path.basename(os.path.dirname(path))


class CryingSenseDataset(Dataset):
    """Dataset for loading feature files."""
    def __init__(self, file_list, label_map, feature_base_dir):
        self.file_list = file_list
        self.label_map = label_map
        self.feature_base_dir = feature_base_dir
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # Load all feature types and combine them
        mfcc_path = self.file_list[idx]
        
        # Construct paths for other features
        rel_path = os.path.relpath(mfcc_path, os.path.join(self.feature_base_dir, 'mfcc'))
        mel_path = os.path.join(self.feature_base_dir, 'mel_spectrogram', rel_path)
        chroma_path = os.path.join(self.feature_base_dir, 'chroma', rel_path)
        
        # Load and combine features
        mfcc = np.load(mfcc_path)
        mel = np.load(mel_path)
        chroma = np.load(chroma_path)
        
        x = self._combine_features(mfcc, mel, chroma)
        x = torch.tensor(x, dtype=torch.float32)
        
        label_name = get_label_from_path(mfcc_path)
        y = self.label_map[label_name]
        return x, y
    
    def _combine_features(self, mfcc, mel, chroma):
        """Combine features into 4-channel array."""
        target_height = max(mfcc.shape[0], mel.shape[0], chroma.shape[0])
        target_width = mfcc.shape[1]
        
        mfcc_padded = self._pad_feature(mfcc, (target_height, target_width))
        mel_padded = self._pad_feature(mel, (target_height, target_width))
        chroma_padded = self._pad_feature(chroma, (target_height, target_width))
        
        delta_mfcc = np.zeros_like(mfcc)
        delta_mfcc[:, 1:] = mfcc[:, 1:] - mfcc[:, :-1]
        delta_mfcc_padded = self._pad_feature(delta_mfcc, (target_height, target_width))
        
        return np.stack([mfcc_padded, mel_padded, chroma_padded, delta_mfcc_padded], axis=0)
    
    def _pad_feature(self, feature, target_shape):
        """Pad feature to target shape."""
        padded = np.zeros(target_shape, dtype=feature.dtype)
        min_h = min(feature.shape[0], target_shape[0])
        min_w = min(feature.shape[1], target_shape[1])
        padded[:min_h, :min_w] = feature[:min_h, :min_w]
        return padded


def get_file_list_and_labels(feature_base_dir):
    """Get all feature files and create label mapping."""
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

if __name__ == "__main__":
    # Configuration - use absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))
    
    feature_base_dir = os.path.join(project_root, 'dataset', 'processed', 'feature_extraction', 'cleaned')
    model_path = os.path.join(project_root, 'model', 'saved_models', 'cryingsense_cnn_best.pth')
    
    print("="*60)
    print("CryingSense CNN Validation")
    print("="*60)
    
    # Load dataset
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
    print("="*60)
    
    # Load model
    print("Loading model...")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("\nPlease train the model first:")
        print("  python model/training/train.py")
        sys.exit(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CryingSenseCNN(num_classes=len(label_map)).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        # Initialize the model with a dummy forward pass to create _fc1 layer
        dummy_input = torch.randn(1, 4, 128, 216).to(device)
        _ = model(dummy_input)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Training accuracy: {checkpoint.get('train_acc', 0):.4f}")
        print(f"Validation accuracy: {checkpoint.get('val_acc', 0):.4f}")
    else:
        # Initialize the model with a dummy forward pass to create _fc1 layer
        dummy_input = torch.randn(1, 4, 128, 216).to(device)
        _ = model(dummy_input)
        
        model.load_state_dict(checkpoint)
    
    print(f"Device: {device}")
    print("="*60)
    
    # Create validation dataset and loader
    val_files = file_list  # Use all files for evaluation or split as needed
    val_dataset = CryingSenseDataset(val_files, label_map, feature_base_dir)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    
    # Evaluate
    print("Evaluating model...")
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validating"):
            x = x.to(device)
            out = model(x)
            preds = out.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    
    print("\n" + "="*60)
    print("Validation Results")
    print("="*60)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=list(label_map.keys())))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("="*60)
