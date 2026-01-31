import os
import shutil
import random
from sklearn.model_selection import train_test_split
from pathlib import Path


def split_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        input_dir: Directory containing class subdirectories with audio files
        output_dir: Directory to save split datasets
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    random.seed(seed)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(input_dir) 
                 if os.path.isdir(os.path.join(input_dir, d))]
    
    print(f"Found {len(class_dirs)} classes: {class_dirs}")
    
    for class_name in class_dirs:
        class_path = os.path.join(input_dir, class_name)
        
        # Get all files in this class
        files = [f for f in os.listdir(class_path) 
                if f.endswith(('.wav', '.npy'))]
        
        print(f"\nClass '{class_name}': {len(files)} files")
        
        if len(files) == 0:
            print(f"  Warning: No files found in {class_name}, skipping...")
            continue
        
        # Split files
        train_files, temp_files = train_test_split(
            files, test_size=(val_ratio + test_ratio), random_state=seed, shuffle=True
        )
        
        val_files, test_files = train_test_split(
            temp_files, test_size=(test_ratio / (val_ratio + test_ratio)), 
            random_state=seed, shuffle=True
        )
        
        print(f"  Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
        
        # Copy files to respective directories
        for split, file_list in [('train', train_files), ('val', val_files), ('test', test_files)]:
            split_class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            
            for file in file_list:
                src = os.path.join(class_path, file)
                dst = os.path.join(split_class_dir, file)
                shutil.copy2(src, dst)
    
    print(f"\nDataset split complete! Files saved to {output_dir}")
    print_split_summary(output_dir)


def print_split_summary(output_dir):
    """Print summary of the split dataset."""
    print("\n" + "="*60)
    print("DATASET SPLIT SUMMARY")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split)
        print(f"\n{split.upper()}:")
        
        if not os.path.exists(split_dir):
            print("  Directory not found")
            continue
        
        class_dirs = [d for d in os.listdir(split_dir) 
                     if os.path.isdir(os.path.join(split_dir, d))]
        
        total_files = 0
        for class_name in sorted(class_dirs):
            class_path = os.path.join(split_dir, class_name)
            num_files = len([f for f in os.listdir(class_path) 
                           if f.endswith(('.wav', '.npy'))])
            print(f"  {class_name}: {num_files} files")
            total_files += num_files
        
        print(f"  TOTAL: {total_files} files")
    
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test sets')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory containing class subdirectories')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for split dataset')
    parser.add_argument('--train', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test', type=float, default=0.15,
                       help='Test set ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    split_dataset(
        input_dir=args.input,
        output_dir=args.output,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed
    )
