"""
Dataset Split Module for CryingSense

This module splits the preprocessed dataset into training, validation, and test sets
while ensuring that samples from the same recording session don't appear in multiple splits.

Split ratios:
- Training: 70%
- Validation: 15%
- Test: 15%
"""

import os
import json
import shutil
import numpy as np
from collections import defaultdict
from pathlib import Path


def extract_session_id(filename):
    """
    Extract session/infant ID from filename to group related recordings.
    
    Examples:
        '357c_part1.wav' -> '357c'
        'burping_aug_701.wav' -> 'burping'
        'cry_001.wav' -> 'cry'
    
    Args:
        filename: Name of the audio file
    
    Returns:
        Session identifier string
    """
    # Remove extension
    base = os.path.splitext(filename)[0]
    
    # Extract base session ID (before _part, _aug, etc.)
    if '_part' in base:
        return base.split('_part')[0]
    elif '_aug_' in base:
        return base.split('_aug_')[0]
    elif '_' in base:
        # Generic case: use prefix before last underscore
        parts = base.rsplit('_', 1)
        return parts[0]
    else:
        return base


def get_file_groups(data_dir, classes):
    """
    Group files by class and session ID to ensure proper splitting.
    
    Args:
        data_dir: Directory containing class subdirectories with audio files
        classes: List of class names (subdirectories)
    
    Returns:
        dict: Nested dictionary {class: {session_id: [files]}}
    """
    groups = defaultdict(lambda: defaultdict(list))
    
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Class directory not found: {class_dir}")
            continue
        
        files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
        
        for file in files:
            session_id = extract_session_id(file)
            groups[class_name][session_id].append(file)
    
    return groups


def split_dataset(data_dir, output_dir, train_ratio=0.70, val_ratio=0.15, 
                 test_ratio=0.15, random_seed=42):
    """
    Split dataset into train/validation/test sets by session.
    
    Args:
        data_dir: Directory with cleaned audio files
        output_dir: Directory to save split information
        train_ratio: Proportion for training set (default: 0.70)
        val_ratio: Proportion for validation set (default: 0.15)
        test_ratio: Proportion for test set (default: 0.15)
        random_seed: Random seed for reproducibility (default: 42)
    
    Returns:
        dict: Split statistics and file mappings
    """
    np.random.seed(random_seed)
    
    # Define classes (excluding 'noise' from training splits if needed)
    classes = ['belly_pain', 'burp', 'discomfort', 'hunger', 'tired', 'noise']
    
    # Get file groups by session
    groups = get_file_groups(data_dir, classes)
    
    # Initialize split data structure
    splits = {
        'train': defaultdict(list),
        'val': defaultdict(list),
        'test': defaultdict(list)
    }
    
    statistics = {
        'train': {},
        'val': {},
        'test': {},
        'total': {}
    }
    
    # Split each class independently
    for class_name in classes:
        session_ids = list(groups[class_name].keys())
        np.random.shuffle(session_ids)
        
        n_sessions = len(session_ids)
        n_train = int(n_sessions * train_ratio)
        n_val = int(n_sessions * val_ratio)
        
        train_sessions = session_ids[:n_train]
        val_sessions = session_ids[n_train:n_train + n_val]
        test_sessions = session_ids[n_train + n_val:]
        
        # Assign files to splits
        for session_id in train_sessions:
            splits['train'][class_name].extend(groups[class_name][session_id])
        
        for session_id in val_sessions:
            splits['val'][class_name].extend(groups[class_name][session_id])
        
        for session_id in test_sessions:
            splits['test'][class_name].extend(groups[class_name][session_id])
        
        # Calculate statistics
        statistics['train'][class_name] = len(splits['train'][class_name])
        statistics['val'][class_name] = len(splits['val'][class_name])
        statistics['test'][class_name] = len(splits['test'][class_name])
        statistics['total'][class_name] = (
            statistics['train'][class_name] + 
            statistics['val'][class_name] + 
            statistics['test'][class_name]
        )
    
    # Save split information to JSON
    os.makedirs(output_dir, exist_ok=True)
    split_file = os.path.join(output_dir, 'dataset_split.json')
    
    split_data = {
        'splits': {
            'train': dict(splits['train']),
            'val': dict(splits['val']),
            'test': dict(splits['test'])
        },
        'statistics': statistics,
        'config': {
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'random_seed': random_seed,
            'classes': classes
        }
    }
    
    with open(split_file, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    return split_data


def main():
    """Main function to run dataset splitting."""
    import sys
    
    # Get paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    data_dir = os.path.join(project_root, "dataset", "processed", "cleaned")
    output_dir = os.path.join(project_root, "dataset")
    
    print("="*60)
    print("CryingSense Dataset Splitting")
    print("="*60)
    print(f"Input directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Train ratio: 70%")
    print(f"Validation ratio: 15%")
    print(f"Test ratio: 15%")
    print("="*60)
    print()
    
    # Check if cleaned data exists
    if not os.path.exists(data_dir):
        print(f"Error: Cleaned data directory not found: {data_dir}")
        print("Please run preprocess_audio.py first.")
        sys.exit(1)
    
    # Perform split
    split_data = split_dataset(data_dir, output_dir)
    
    print()
    print("="*60)
    print("Dataset Split Complete")
    print("="*60)
    
    # Print statistics
    stats = split_data['statistics']
    for split_name in ['train', 'val', 'test']:
        print(f"\n{split_name.upper()} SET:")
        total = 0
        for class_name in stats['train'].keys():
            count = stats[split_name][class_name]
            total += count
            print(f"  {class_name:15s}: {count:4d} files")
        print(f"  {'Total':15s}: {total:4d} files")
    
    print(f"\nSplit information saved to: {output_dir}/dataset_split.json")
    print("="*60)


if __name__ == "__main__":
    main()
