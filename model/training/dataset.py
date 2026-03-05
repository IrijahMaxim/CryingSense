"""
Shared Dataset Module for CryingSense Model Training

This module contains the shared CryingSenseDataset class and utility functions
used by train.py, evaluate.py, and validate.py.

Having this in a single place avoids code duplication and ensures consistency.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


def get_label_from_path(path):
    """
    Extract class label from file path.
    
    Assumes path like: .../class_name/filename.npy
    
    Args:
        path: File path to feature file
    
    Returns:
        str: Class label (directory name containing the file)
    """
    return os.path.basename(os.path.dirname(path))


class CryingSenseDataset(Dataset):
    """
    Dataset for loading CryingSense feature files.
    
    Loads MFCC, Mel spectrogram, and Chroma features,
    combines them into a 4-channel input tensor for the CNN model.
    """
    
    def __init__(self, file_list, label_map, feature_base_dirs=None, augment=False):
        """
        Initialize the dataset.
        
        Args:
            file_list: List of (mfcc_path, base_dir) tuples or just mfcc paths
            label_map: Dict mapping class names to integer indices
            feature_base_dirs: Dict mapping source names to base directories
            augment: Whether to apply data augmentation during training
        """
        self.file_list = file_list
        self.label_map = label_map
        self.feature_base_dirs = feature_base_dirs or {}
        self.augment = augment
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """Load and return a single sample."""
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
        
        # Get label
        label_name = get_label_from_path(mfcc_path)
        y = self.label_map[label_name]
        
        return x, y
    
    def _combine_features(self, mfcc, mel, chroma):
        """
        Combine multiple features into a 4-channel array.
        
        Channels:
            0: MFCC (padded to max height)
            1: Mel Spectrogram (padded to max height)
            2: Chroma features (padded to max height)
            3: Delta MFCC (first derivative)
        
        Args:
            mfcc: MFCC features (n_mfcc, time_steps)
            mel: Mel spectrogram (n_mels, time_steps)
            chroma: Chroma features (n_chroma, time_steps)
        
        Returns:
            np.ndarray: Combined features (4, height, width)
        """
        # Get target dimensions
        target_height = max(mfcc.shape[0], mel.shape[0], chroma.shape[0])
        target_width = mfcc.shape[1]  # Time steps
        
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
        """
        Pad feature to target shape with zeros.
        
        Args:
            feature: Input feature array
            target_shape: Desired output shape (height, width)
        
        Returns:
            np.ndarray: Padded feature array
        """
        padded = np.zeros(target_shape, dtype=feature.dtype)
        min_h = min(feature.shape[0], target_shape[0])
        min_w = min(feature.shape[1], target_shape[1])
        padded[:min_h, :min_w] = feature[:min_h, :min_w]
        return padded
    
    def _compute_delta(self, feature):
        """
        Compute delta (first derivative) of feature.
        
        Args:
            feature: Input feature array
        
        Returns:
            np.ndarray: Delta feature (same shape as input)
        """
        delta = np.zeros_like(feature)
        delta[:, 1:] = feature[:, 1:] - feature[:, :-1]
        return delta
    
    def _augment_features(self, features):
        """
        Apply data augmentation to features.
        
        Augmentations applied randomly:
        - Noise addition (10% chance)
        - Time shift (20% chance)
        - Amplitude scaling (20% chance)
        
        Args:
            features: Input feature tensor
        
        Returns:
            torch.Tensor: Augmented features
        """
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
    
    def _infer_base_dir(self, mfcc_path):
        """
        Infer base directory from MFCC path (legacy support).
        
        Args:
            mfcc_path: Path to MFCC feature file
        
        Returns:
            str: Base directory path
        """
        path_parts = mfcc_path.replace('\\', '/').split('/')
        for i, part in enumerate(path_parts):
            if part == 'mfcc':
                return '/'.join(path_parts[:i])
        # Fallback to first registered base dir
        if self.feature_base_dirs:
            return list(self.feature_base_dirs.values())[0]
        return os.path.dirname(os.path.dirname(mfcc_path))


def get_file_list_and_labels(feature_base_dir):
    """
    Get list of all feature files and create label mapping.
    
    Args:
        feature_base_dir: Base directory containing mfcc/, mel_spectrogram/, chroma/
    
    Returns:
        tuple: (file_list, label_map)
            - file_list: List of (mfcc_path, base_dir) tuples
            - label_map: Dict mapping class names to indices
    """
    mfcc_dir = os.path.join(feature_base_dir, 'mfcc')
    
    if not os.path.exists(mfcc_dir):
        raise FileNotFoundError(f"MFCC directory not found: {mfcc_dir}")
    
    file_list = []
    
    # Walk through class directories
    for class_name in sorted(os.listdir(mfcc_dir)):
        class_dir = os.path.join(mfcc_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        for filename in os.listdir(class_dir):
            if filename.endswith('.npy'):
                mfcc_path = os.path.join(class_dir, filename)
                file_list.append((mfcc_path, feature_base_dir))
    
    # Create label mapping from sorted unique labels
    labels = sorted(list(set(get_label_from_path(f[0]) for f in file_list)))
    label_map = {label: i for i, label in enumerate(labels)}
    
    return file_list, label_map


def load_split_from_json(json_path, feature_base_dirs):
    """
    Load dataset split from JSON file.
    
    Args:
        json_path: Path to dataset_split.json
        feature_base_dirs: Dict mapping source names to feature directories
    
    Returns:
        tuple: (train_files, val_files, test_files) as lists of (path, base_dir) tuples
    """
    import json
    
    with open(json_path, 'r') as f:
        split_data = json.load(f)
    
    train_files = []
    val_files = []
    test_files = []
    
    # Handle nested structure: {"splits": {"train": {"class_name": [files]}}}
    splits = split_data.get('splits', split_data)
    
    for split_name, split_content in splits.items():
        target_list = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }.get(split_name)
        
        if target_list is None:
            continue
        
        # If split_content is a dict (organized by class), flatten it
        if isinstance(split_content, dict):
            for class_name, file_list in split_content.items():
                for filename in file_list:
                    # Determine which base_dir to use
                    for source_name, base_dir in feature_base_dirs.items():
                        mfcc_path = os.path.join(base_dir, 'mfcc', class_name, filename)
                        if os.path.exists(mfcc_path):
                            target_list.append((mfcc_path, base_dir))
                            break
        else:
            # List of filenames
            for filename in split_content:
                for source_name, base_dir in feature_base_dirs.items():
                    # Try to find the file
                    class_name = filename.rsplit('_', 1)[0] if '_' in filename else filename.replace('.npy', '')
                    mfcc_path = os.path.join(base_dir, 'mfcc', class_name, filename)
                    if os.path.exists(mfcc_path):
                        target_list.append((mfcc_path, base_dir))
                        break
    
    return train_files, val_files, test_files
