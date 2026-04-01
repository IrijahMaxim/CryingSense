"""
Dataset Split Module for CryingSense

This module splits the dataset into training, validation, and evaluation sets
while ensuring that samples from the same recording session don't appear in multiple splits.

Supports both cleaned (preprocessed) and raw audio datasets.

Default split ratios:
- Training: 60%
- Validation: 20%
- Evaluation: 20%

Custom split configuration:
- Use --custom-split flag for interactive prompt
- Specify your own train/val/eval percentages
- Optionally limit the number of files per class:
  * Uniform limit (e.g., 50 files per class)
  * Individual limits per class
  * Use all files (no limit)

Usage:
  python dataset_split.py              # Split cleaned data only (default 60/20/20, all files)
  python dataset_split.py --custom-split  # Interactive custom ratio and file limit prompt
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
        feature_base_dir: Base directory for features (e.g., .../features/cleaned)
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


def split_dataset(feature_base_dir, output_dir, train_ratio=0.60, val_ratio=0.20, 
                 eval_ratio=0.20, random_seed=42, class_filter=None, max_files_per_class=None):
    """
    Split feature files into train/validation/evaluate sets by session.
    
    Splits by FILE COUNTS while keeping sessions together to achieve proper 60/20/20 ratios.
    
    Args:
        feature_base_dir: Base directory with extracted features (e.g., .../features/cleaned)
        output_dir: Directory to save split information
        train_ratio: Proportion for training set (default: 0.60)
        val_ratio: Proportion for validation set (default: 0.20)
        eval_ratio: Proportion for evaluate set (default: 0.20)
        random_seed: Random seed for reproducibility (default: 42)
        class_filter: List of classes to include, or None for all classes
        max_files_per_class: Maximum files per class, or None/dict for no limit
    
    Returns:
        dict: Split statistics and file mappings
    """
    np.random.seed(random_seed)
    
    # Define classes
    all_classes = ['belly_pain', 'burp', 'discomfort', 'hunger', 'tired', 'noise', 'speech']
    classes = class_filter if class_filter else all_classes
    
    # Get file groups by session (from feature files)
    groups = get_file_groups(feature_base_dir, classes)
    
    # Limit files per class if specified
    if max_files_per_class:
        for class_name in classes:
            if class_name in groups:
                # Get limit for this class
                if isinstance(max_files_per_class, dict):
                    limit = max_files_per_class.get(class_name, None)
                else:
                    limit = max_files_per_class
                
                if limit:
                    # Count current files and limit sessions if needed
                    session_ids = list(groups[class_name].keys())
                    np.random.shuffle(session_ids)  # Randomize session selection
                    
                    total_files = 0
                    selected_sessions = {}
                    
                    for session_id in session_ids:
                        session_files = groups[class_name][session_id]
                        if total_files + len(session_files) <= limit:
                            selected_sessions[session_id] = session_files
                            total_files += len(session_files)
                        elif total_files < limit:
                            # Partially include this session
                            remaining = limit - total_files
                            selected_sessions[session_id] = session_files[:remaining]
                            total_files = limit
                            break
                        else:
                            break
                    
                    groups[class_name] = selected_sessions
    
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


def get_available_file_limits_by_class(datasets_to_process):
    """
    Calculate available file counts per class from selected dataset feature subfolders.

    Uses each dataset's MFCC subfolder as the source of truth.

    Args:
        datasets_to_process: List of tuples (dataset_name, feature_dir, class_filter)

    Returns:
        tuple: (aggregate_counts, per_dataset_counts)
            - aggregate_counts: dict {class_name: total_available_files}
            - per_dataset_counts: dict {dataset_name: {class_name: available_files}}
    """
    all_classes = ['belly_pain', 'burp', 'discomfort', 'hunger', 'tired', 'noise', 'speech']
    aggregate_counts = defaultdict(int)
    per_dataset_counts = {}

    for dataset_name, data_dir, class_filter in datasets_to_process:
        classes = class_filter if class_filter else all_classes
        dataset_counts = {}
        mfcc_dir = os.path.join(data_dir, 'mfcc')

        for class_name in classes:
            class_dir = os.path.join(mfcc_dir, class_name)
            if not os.path.exists(class_dir):
                dataset_counts[class_name] = 0
                continue

            count = len([f for f in os.listdir(class_dir) if f.endswith('.npy')])
            dataset_counts[class_name] = count
            aggregate_counts[class_name] += count

        per_dataset_counts[dataset_name] = dataset_counts

    return dict(aggregate_counts), per_dataset_counts


def get_custom_split_ratios(datasets_to_process):
    """
    Prompt user for custom split ratios and file count limits.
    
    Args:
        datasets_to_process: List of tuples (dataset_name, feature_dir, class_filter)

    Returns:
        tuple: (train_ratio, val_ratio, eval_ratio, max_files_per_class)
    """
    print("\n" + "="*60)
    print("Custom Split Configuration")
    print("="*60)
    print("\nDefault split: 60% train, 20% val, 20% eval")
    print("\nOptions:")
    print("  1. Use default split (60/20/20) with all files")
    print("  2. Enter custom configuration")
    print("="*60)
    
    while True:
        choice = input("\nChoose option (1 or 2): ").strip()
        
        if choice == '1':
            print("\n✓ Using default split: 60% train, 20% val, 20% eval")
            print("✓ Using all available files")
            return 0.60, 0.20, 0.20, None
        
        elif choice == '2':
            print("\n" + "-"*60)
            print("Enter custom split ratios (percentages)")
            print("-"*60)
            
            while True:
                try:
                    train_pct = float(input("Training set percentage (e.g., 70): ").strip())
                    val_pct = float(input("Validation set percentage (e.g., 15): ").strip())
                    eval_pct = float(input("Evaluation set percentage (e.g., 15): ").strip())
                    
                    # Validate
                    total = train_pct + val_pct + eval_pct
                    if abs(total - 100.0) > 0.01:
                        print(f"\n❌ Error: Percentages must sum to 100 (got {total:.1f})")
                        print("Please try again.\n")
                        continue
                    
                    if train_pct <= 0 or val_pct <= 0 or eval_pct <= 0:
                        print("\n❌ Error: All percentages must be greater than 0")
                        print("Please try again.\n")
                        continue
                    
                    # Convert to ratios
                    train_ratio = train_pct / 100.0
                    val_ratio = val_pct / 100.0
                    eval_ratio = eval_pct / 100.0
                    
                    print(f"\n✓ Custom split: {train_pct:.1f}% train, {val_pct:.1f}% val, {eval_pct:.1f}% eval")
                    break
                    
                except ValueError:
                    print("\n❌ Error: Please enter valid numbers")
                    print("Please try again.\n")
                    continue
            
            # Ask about file count limits
            print("\n" + "-"*60)
            print("File Count Limits (Optional)")
            print("-"*60)
            print("\nLimit the number of files used per class?")
            print("  1. Use all available files")
            print("  2. Set a uniform limit for all classes")
            print("  3. Set individual limits per class")
            
            while True:
                limit_choice = input("\nChoose option (1, 2, or 3): ").strip()
                
                if limit_choice == '1':
                    print("\n✓ Using all available files")
                    max_files = None
                    break
                
                elif limit_choice == '2':
                    try:
                        available_by_class, available_by_dataset = get_available_file_limits_by_class(datasets_to_process)
                        print("\nMaximum available files per class (from selected dataset subfolders):")
                        for dataset_name, class_counts in available_by_dataset.items():
                            print(f"  [{dataset_name}]")
                            for class_name in sorted(class_counts.keys()):
                                print(f"    {class_name:12s}: {class_counts[class_name]}")
                        print("  [combined]")
                        for class_name in sorted(available_by_class.keys()):
                            print(f"    {class_name:12s}: {available_by_class[class_name]}")

                        limit = int(input("\nEnter file limit per class (e.g., 100): ").strip())
                        if limit <= 0:
                            print("\n❌ Error: Limit must be greater than 0")
                            continue

                        if available_by_class:
                            max_uniform = min(v for v in available_by_class.values() if v > 0) if any(v > 0 for v in available_by_class.values()) else 0
                            if max_uniform > 0 and limit > max_uniform:
                                print(f"\n⚠️  Note: limit {limit} exceeds the smallest non-zero class maximum ({max_uniform}).")

                        print(f"\n✓ Using {limit} files per class")
                        max_files = limit
                        break
                    except ValueError:
                        print("\n❌ Error: Please enter a valid number")
                        continue
                
                elif limit_choice == '3':
                    print("\nEnter file limits for each class (or 0 to use all):")
                    available_by_class, available_by_dataset = get_available_file_limits_by_class(datasets_to_process)
                    classes = sorted(available_by_class.keys())

                    print("\nMaximum available files per class (from selected dataset subfolders):")
                    for dataset_name, class_counts in available_by_dataset.items():
                        print(f"  [{dataset_name}]")
                        for class_name in sorted(class_counts.keys()):
                            print(f"    {class_name:12s}: {class_counts[class_name]}")
                    print("\n  [combined]")
                    for class_name in classes:
                        print(f"    {class_name:12s}: {available_by_class[class_name]}")

                    max_files = {}
                    
                    try:
                        print("\nEnter file limits for each class")
                        for class_name in classes:
                            limit = int(input(f"  {class_name}: ").strip())
                            if limit > 0:
                                class_max = available_by_class.get(class_name, 0)
                                if class_max > 0 and limit > class_max:
                                    print(f"    ⚠️  Note: {limit} exceeds available max ({class_max}) for {class_name}")
                                max_files[class_name] = limit
                        
                        # Show summary
                        print("\n✓ Custom limits set:")
                        for class_name in classes:
                            if class_name in max_files:
                                print(f"  {class_name}: {max_files[class_name]} files")
                            else:
                                print(f"  {class_name}: all files")
                        
                        if not max_files:
                            max_files = None
                        break
                    except ValueError:
                        print("\n❌ Error: Please enter valid numbers")
                        continue
                
                else:
                    print("\n❌ Invalid choice. Please enter 1, 2, or 3.")
            
            # Final confirmation
            print("\n" + "="*60)
            print("Configuration Summary")
            print("="*60)
            print(f"Split ratios: {train_pct:.1f}% train, {val_pct:.1f}% val, {eval_pct:.1f}% evaluation")
            if max_files is None:
                print("File limits: Using all available files")
            elif isinstance(max_files, dict):
                print("File limits: Custom per class")
            else:
                print(f"File limits: {max_files} files per class")
            print("="*60)
            
            confirm = input("\nConfirm configuration? (yes/no): ").strip().lower()
            if confirm in ['yes', 'y']:
                return train_ratio, val_ratio, eval_ratio, max_files
            else:
                print("\nLet's start over...\n")
                continue
                        
        else:
            print("\n❌ Invalid choice. Please enter 1 or 2.")


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
    parser.add_argument('--custom-split', action='store_true',
                       help='Use custom split ratios and file count limits (interactive prompt)')
    
    args = parser.parse_args()
    
    # Get paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Define feature extraction directories
    cleaned_dir = os.path.join(project_root, "dataset", "processed", "features", "cleaned")
    raw_dir = os.path.join(project_root, "dataset", "processed", "features", "raw")
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
    
    # Get split ratios and file limits (default or custom)
    if args.custom_split:
        train_ratio, val_ratio, eval_ratio, max_files_per_class = get_custom_split_ratios(datasets_to_process)
    else:
        train_ratio, val_ratio, eval_ratio, max_files_per_class = 0.60, 0.20, 0.20, None
    
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
    print(f"Train ratio: {train_ratio*100:.1f}%")
    print(f"Validation ratio: {val_ratio*100:.1f}%")
    print(f"Evaluation ratio: {eval_ratio*100:.1f}%")
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
        split_data = split_dataset(data_dir, output_dir, 
                                   train_ratio=train_ratio,
                                   val_ratio=val_ratio,
                                   eval_ratio=eval_ratio,
                                   class_filter=class_filter,
                                   max_files_per_class=max_files_per_class)
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
                'train_ratio': train_ratio,
                'val_ratio': val_ratio,
                'eval_ratio': eval_ratio,
                'random_seed': 42,
                'max_files_per_class': max_files_per_class if max_files_per_class else 'all',
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
