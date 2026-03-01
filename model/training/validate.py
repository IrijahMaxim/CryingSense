import os
import sys
import json
import logging
from datetime import datetime
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
    def __init__(self, file_list, label_map, feature_base_dirs=None):
        """
        Args:
            file_list: List of (mfcc_path, base_dir) tuples or just mfcc paths
            label_map: Dict mapping class names to indices
            feature_base_dirs: Dict mapping source names to base directories
        """
        self.file_list = file_list
        self.label_map = label_map
        self.feature_base_dirs = feature_base_dirs or {}
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # Get file info - can be (path, base_dir) tuple or just path
        item = self.file_list[idx]
        if isinstance(item, tuple):
            mfcc_path, base_dir = item
        else:
            mfcc_path = item
            base_dir = self._infer_base_dir(mfcc_path)
        
        # Construct paths for other features
        rel_path = os.path.relpath(mfcc_path, os.path.join(base_dir, 'mfcc'))
        mel_path = os.path.join(base_dir, 'mel_spectrogram', rel_path)
        chroma_path = os.path.join(base_dir, 'chroma', rel_path)
        
        # Load and combine features
        mfcc = np.load(mfcc_path)
        mel = np.load(mel_path)
        chroma = np.load(chroma_path)
        
        x = self._combine_features(mfcc, mel, chroma)
        x = torch.tensor(x, dtype=torch.float32)
        
        label_name = get_label_from_path(mfcc_path)
        y = self.label_map[label_name]
        return x, y
    
    def _infer_base_dir(self, mfcc_path):
        """Infer base directory from MFCC path (legacy support)."""
        path_parts = mfcc_path.replace('\\', '/').split('/')
        for i, part in enumerate(path_parts):
            if part == 'mfcc':
                return '/'.join(path_parts[:i])
        if self.feature_base_dirs:
            return list(self.feature_base_dirs.values())[0]
        return os.path.dirname(os.path.dirname(mfcc_path))
        
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
    
    # Feature directories for both cleaned and raw data
    feature_base_dirs = {
        'cleaned': os.path.join(project_root, 'dataset', 'processed', 'feature_extraction', 'cleaned'),
        'raw': os.path.join(project_root, 'dataset', 'processed', 'feature_extraction', 'raw')
    }
    model_path = os.path.join(project_root, 'model', 'saved_models', 'cryingsense_cnn_best.pth')
    validation_report_dir = os.path.join(project_root, 'performance_reports', 'validation_report')
    logs_dir = os.path.join(project_root, 'performance_reports', 'logs')
    split_json_path = os.path.join(project_root, 'dataset', 'dataset_split.json')
    
    # Create directories
    os.makedirs(validation_report_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(logs_dir, f'validate_{timestamp}.log')
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
    logger.info("CryingSense CNN Validation")
    logger.info("="*60)
    
    print("="*60)
    print("CryingSense CNN Validation")
    print("="*60)
    
    # Try to load from dataset_split.json first
    if os.path.exists(split_json_path):
        print(f"Loading val set from: {split_json_path}")
        splits = load_split_from_json(split_json_path, feature_base_dirs)
        val_files = splits['val']
        
        if not val_files:
            print("Error: No validation files found in dataset_split.json!")
            print("Please run: python scripts/dataset_split.py")
            sys.exit(1)
        
        # Build label map from validation files
        labels = sorted(list(set(get_label_from_path(f[0]) for f in val_files)))
        label_map = {label: i for i, label in enumerate(labels)}
        
        print(f"Loaded from JSON - Val samples: {len(val_files)}")
        print(f"Classes: {list(label_map.keys())}")
    else:
        # Fallback: Load from single directory
        print("Warning: dataset_split.json not found, using all files")
        print("For reproducible splits, run: python scripts/dataset_split.py")
        
        feature_base_dir = feature_base_dirs['cleaned']
        file_list, label_map = get_file_list_and_labels(feature_base_dir)
        
        if not file_list:
            print("Error: No feature files found!")
            print(f"Looking in: {os.path.abspath(feature_base_dir)}")
            print("\nPlease run feature extraction first:")
            print("  python scripts/feature_extraction.py")
            sys.exit(1)
        
        print(f"Total files: {len(file_list)}")
        print(f"Classes: {list(label_map.keys())}")
        
        # Convert to tuple format
        val_files = [(f, feature_base_dir) for f in file_list]
    
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
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
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
    # Note: num_workers=0 for Windows compatibility
    use_pin_memory = torch.cuda.is_available()
    val_dataset = CryingSenseDataset(val_files, label_map, feature_base_dirs)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=use_pin_memory)
    
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
    report = classification_report(all_labels, all_preds, target_names=list(label_map.keys()))
    print(report)
    
    # Save validation classification report
    report_path = os.path.join(validation_report_dir, 'validation_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("CryingSense Model - Validation Classification Report\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    print(f"\nValidation report saved to: {report_path}")
    
    # Calculate and log accuracy
    from sklearn.metrics import accuracy_score
    val_accuracy = accuracy_score(all_labels, all_preds)
    logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
    logger.info(f"Report saved to: {report_path}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    print("="*60)
    
    logger.info("Validation Complete")
