"""
Dataset Split Module for CryingSense

This module splits the dataset into training, validation, and evaluate sets
while ensuring that samples from the same recording session don't appear in multiple splits.

Supports both cleaned (preprocessed) and raw audio datasets.

Split ratios:
- Training: 80%
- Validation: 10%
- Evaluate: 10%

Usage:
  python dataset_split.py              # Split cleaned data only (default)
  python dataset_split.py --raw-only   # Split raw data only
  python dataset_split.py --all        # Split both cleaned and raw data
  python dataset_split.py --noise-raw  # All cleaned + only noise from raw

Output: dataset_split.json (always the same filename)
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
        '357c_part1.npy' -> '357c'
        'burping_aug_701.npy' -> 'burping'
        'cry_001.npy' -> 'cry'
    
    Args:
        filename: Name of the feature file (.npy)
    
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


def get_file_groups(feature_base_dir, classes):
    """
    Group feature files by class and session ID to ensure proper splitting.
    
    Uses MFCC directory as reference (all feature types have same files).
    
    Args:
        feature_base_dir: Base directory for features (e.g., .../feature_extraction/cleaned)
        classes: List of class names (subdirectories)
    
    Returns:
        dict: Nested dictionary {class: {session_id: [files]}}
    """
    groups = defaultdict(lambda: defaultdict(list))
    
    # Use MFCC directory as reference for file discovery
    mfcc_dir = os.path.join(feature_base_dir, 'mfcc')
    
    for class_name in classes:
        class_dir = os.path.join(mfcc_dir, class_name)
        if not os.path.exists(class_dir):
            # Not all datasets have all classes
            continue
        
        files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        
        for file in files:
            session_id = extract_session_id(file)
            groups[class_name][session_id].append(file)
    
    return groups


def split_dataset(feature_base_dir, output_dir, train_ratio=0.65, val_ratio=0.15, 
                 eval_ratio=0.20, random_seed=42, class_filter=None):
    """
    Split feature files into train/validation/evaluate sets by session.
    
    Splits by FILE COUNTS while keeping sessions together to achieve proper 80/10/10 ratios.
    
    Args:
        feature_base_dir: Base directory with extracted features (e.g., .../feature_extraction/cleaned)
        output_dir: Directory to save split information
        train_ratio: Proportion for training set (default: 0.80)
        val_ratio: Proportion for validation set (default: 0.10)
        eval_ratio: Proportion for evaluate set (default: 0.10)
        random_seed: Random seed for reproducibility (default: 42)
        class_filter: List of classes to include, or None for all classes
    
    Returns:
        dict: Split statistics and file mappings
    """
    np.random.seed(random_seed)
    
    # Define classes
    all_classes = ['belly_pain', 'burp', 'discomfort', 'hunger', 'tired', 'noise']
    classes = class_filter if class_filter else all_classes
    
    # Get file groups by session (from feature files)
    groups = get_file_groups(feature_base_dir, classes)
    
    # Initialize split data structure
    splits = {
        'train': defaultdict(list),
        'val': defaultdict(list),
        'eval': defaultdict(list)
    }
    
    statistics = {
        'train': {},
        'val': {},
        'eval': {},
        'total': {}
    }
    
    # Split each class independently - by FILE COUNT not session count
    for class_name in classes:
        # Get all sessions with their file counts
        sessions_with_counts = [
            (session_id, len(files)) 
            for session_id, files in groups[class_name].items()
        ]
        
        # Shuffle sessions for randomization
        np.random.shuffle(sessions_with_counts)
        
        # Calculate total files and target counts
        total_files = sum(count for _, count in sessions_with_counts)
        target_train = int(total_files * train_ratio)
        target_val = int(total_files * val_ratio)
        target_eval = total_files - target_train - target_val
        
        # Assign sessions to splits using balanced bin-packing
        # Track current counts for each split
        train_sessions = []
        val_sessions = []
        eval_sessions = []
        
        train_count = 0
        val_count = 0
        eval_count = 0
        
        # Sort sessions by size (largest first) for better packing
        sessions_sorted = sorted(sessions_with_counts, key=lambda x: x[1], reverse=True)
        
        for session_id, file_count in sessions_sorted:
            # Calculate how far each split is from its target (as a ratio of target)
            train_need = (target_train - train_count) / max(target_train, 1)
            val_need = (target_val - val_count) / max(target_val, 1)
            eval_need = (target_eval - eval_count) / max(target_eval, 1)
            
            # Assign to the split that needs files most
            max_need = max(train_need, val_need, eval_need)
            
            if train_need == max_need:
                train_sessions.append(session_id)
                train_count += file_count
            elif val_need == max_need:
                val_sessions.append(session_id)
                val_count += file_count
            else:
                eval_sessions.append(session_id)
                eval_count += file_count
        
        # Assign files to splits
        for session_id in train_sessions:
            splits['train'][class_name].extend(groups[class_name][session_id])
        
        for session_id in val_sessions:
            splits['val'][class_name].extend(groups[class_name][session_id])
        
        for session_id in eval_sessions:
            splits['eval'][class_name].extend(groups[class_name][session_id])
        
        # Calculate statistics
        statistics['train'][class_name] = len(splits['train'][class_name])
        statistics['val'][class_name] = len(splits['val'][class_name])
        statistics['eval'][class_name] = len(splits['eval'][class_name])
        statistics['total'][class_name] = (
            statistics['train'][class_name] + 
            statistics['val'][class_name] + 
            statistics['eval'][class_name]
        )
    
    # Save split information to JSON
    os.makedirs(output_dir, exist_ok=True)
    split_file = os.path.join(output_dir, 'dataset_split.json')
    
    split_data = {
        'splits': {
            'train': dict(splits['train']),
            'val': dict(splits['val']),
            'eval': dict(splits['eval'])
        },
        'statistics': statistics,
        'config': {
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'eval_ratio': eval_ratio,
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
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Split CryingSense dataset into train/val/eval sets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split cleaned data only (default)
  python dataset_split.py
  
  # Split raw data only
  python dataset_split.py --raw-only
  
  # Split both cleaned and raw data
  python dataset_split.py --all
  
  # All cleaned data + only noise from raw
  python dataset_split.py --noise-raw

Output: dataset_split.json
        """
    )
    
    parser.add_argument('--raw-only', action='store_true',
                       help='Only split raw dataset')
    parser.add_argument('--all', action='store_true',
                       help='Split both cleaned and raw datasets')
    parser.add_argument('--noise-raw', action='store_true',
                       help='All cleaned data + only noise class from raw')
    
    args = parser.parse_args()
    
    # Get paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Define feature extraction directories
    cleaned_dir = os.path.join(project_root, "dataset", "processed", "feature_extraction", "cleaned")
    raw_dir = os.path.join(project_root, "dataset", "processed", "feature_extraction", "raw")
    output_dir = os.path.join(project_root, "dataset")
    
    # Determine which datasets to process
    # Format: (name, dir, class_filter or None)
    if args.all:
        # Both cleaned and raw
        datasets_to_process = [
            ('cleaned', cleaned_dir, None),
            ('raw', raw_dir, None)
        ]
        mode_name = "BOTH (CLEANED + RAW)"
    elif args.noise_raw:
        # All cleaned + only noise from raw
        datasets_to_process = [
            ('cleaned', cleaned_dir, None),
            ('raw', raw_dir, ['noise'])  # Only noise class from raw
        ]
        mode_name = "CLEANED + RAW NOISE ONLY"
    elif args.raw_only:
        datasets_to_process = [('raw', raw_dir, None)]
        mode_name = "RAW ONLY"
    else:
        # Default: cleaned only
        datasets_to_process = [('cleaned', cleaned_dir, None)]
        mode_name = "CLEANED ONLY"
    
    output_filename = 'dataset_split.json'
    
    # Collect all splits from all datasets
    combined_splits = {
        'train': defaultdict(list),
        'val': defaultdict(list),
        'eval': defaultdict(list)
    }
    combined_stats = {
        'train': defaultdict(int),
        'val': defaultdict(int),
        'eval': defaultdict(int),
        'total': defaultdict(int)
    }
    sources_processed = []
    
    print("="*60)
    print(f"CryingSense Dataset Splitting - {mode_name}")
    print("="*60)
    print(f"Output file: {output_filename}")
    print(f"Train ratio: 65%")
    print(f"Validation ratio: 15%")
    print(f"Evaluate ratio: 20%")
    print("="*60)
    print()
    
    for dataset_name, data_dir, class_filter in datasets_to_process:
        filter_info = f" (classes: {class_filter})" if class_filter else ""
        print(f"Processing {dataset_name.upper()} dataset{filter_info}...")
        print(f"  Input: {data_dir}")
        
        # Check if data exists
        if not os.path.exists(data_dir):
            print(f"  Warning: Directory not found, skipping.")
            print()
            continue
        
        # Perform split
        split_data = split_dataset(data_dir, output_dir, class_filter=class_filter)
        sources_processed.append(dataset_name if not class_filter else f"{dataset_name}:{','.join(class_filter)}")
        
        # Merge splits (prefix filenames with source to avoid collisions)
        for split_name in ['train', 'val', 'eval']:
            for class_name, files in split_data['splits'][split_name].items():
                # Add source prefix to distinguish cleaned vs raw files
                prefixed_files = [f"{dataset_name}:{f}" for f in files]
                combined_splits[split_name][class_name].extend(prefixed_files)
        
        # Accumulate statistics
        for split_name in ['train', 'val', 'eval', 'total']:
            for class_name, count in split_data['statistics'].get(split_name, {}).items():
                combined_stats[split_name][class_name] += count
        
        print(f"  Processed: {sum(split_data['statistics']['total'].values())} files")
        print()
    
    if not sources_processed:
        print("Error: No datasets found to process.")
        sys.exit(1)
    
    # Save combined split
    split_file = os.path.join(output_dir, output_filename)
    with open(split_file, 'w') as f:
        json.dump({
            'sources': sources_processed,
            'splits': {
                'train': dict(combined_splits['train']),
                'val': dict(combined_splits['val']),
                'eval': dict(combined_splits['eval'])
            },
            'statistics': {
                'train': dict(combined_stats['train']),
                'val': dict(combined_stats['val']),
                'eval': dict(combined_stats['eval']),
                'total': dict(combined_stats['total'])
            },
            'config': {
                'train_ratio': 0.65,
                'val_ratio': 0.15,
                'eval_ratio': 0.20,
                'random_seed': 42,
                'classes': ['belly_pain', 'burp', 'discomfort', 'hunger', 'tired', 'noise']
            }
        }, f, indent=2)
    
    print("="*60)
    print("Dataset Split Complete")
    print("="*60)
    
    # Print statistics
    for split_name in ['train', 'val', 'eval']:
        print(f"\n{split_name.upper()} SET:")
        total = 0
        for class_name in combined_stats['total'].keys():
            count = combined_stats[split_name].get(class_name, 0)
            total += count
            print(f"  {class_name:15s}: {count:4d} files")
        print(f"  {'Total':15s}: {total:4d} files")
    
    print(f"\nSources: {', '.join(sources_processed)}")
    print(f"Split information saved to: {split_file}")
    print("="*60)


if __name__ == "__main__":
    main()
