"""
Dataset Cleanup Script for CryingSense

This script removes all files from:
- dataset/visualizations/ (all visualization images)
- dataset/processed/cleaned/ (all cleaned audio files)
- dataset/processed/features/ (all extracted .npy feature files)
- dataset/processed/feature_extraction/ (legacy feature files)

Use this before regenerating processed data and visualizations.

Usage:
    python cleanup_dataset.py [--confirm] [--targets TARGET1 TARGET2 ...]
    
Options:
    --confirm    Skip confirmation prompt and delete immediately
    --targets    Specific targets to clean: viz, cleaned, features, legacy, or all (default: all)
    
Examples:
    # Clean only the three folders
       python cleanup_dataset.py --targets viz cleaned features

    # Clean only visualizations
        python cleanup_dataset.py --targets viz

    # Clean only one specific folder
        python cleanup_dataset.py --targets cleaned/viz/features/legacy

    # Clean everything (default behavior)
        python cleanup_dataset.py
"""

import os
import shutil
import sys
import argparse
from pathlib import Path


def get_folder_size(folder_path):
    """Calculate total size of folder in bytes."""
    total_size = 0
    if os.path.exists(folder_path):
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
    return total_size


def format_size(bytes_size):
    """Format bytes into human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def count_files(folder_path):
    """Count total number of files in folder."""
    if not os.path.exists(folder_path):
        return 0
    count = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        count += len(filenames)
    return count


def cleanup_folder(folder_path, folder_description):
    """Remove all contents of a folder while keeping the folder itself."""
    if not os.path.exists(folder_path):
        print(f"⚠️  {folder_description} does not exist: {folder_path}")
        return 0, 0
    
    # Count before deletion
    file_count = count_files(folder_path)
    folder_size = get_folder_size(folder_path)
    
    if file_count == 0:
        print(f"✓ {folder_description} is already empty")
        return 0, 0
    
    # Delete all contents
    try:
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        
        print(f"✓ Cleaned {folder_description}: {file_count} files ({format_size(folder_size)}) removed")
        return file_count, folder_size
    
    except Exception as e:
        print(f"✗ Error cleaning {folder_description}: {str(e)}")
        return 0, 0


def main():
    """Main cleanup function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Clean CryingSense dataset processed files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--confirm', action='store_true',
                       help='Skip confirmation prompt and delete immediately')
    parser.add_argument('--targets', nargs='+', 
                       choices=['viz', 'cleaned', 'features', 'legacy', 'all'],
                       default=['all'],
                       help='Specific targets to clean (default: all)')
    
    args = parser.parse_args()
    
    # Determine which folders to clean
    targets = set(args.targets)
    if 'all' in targets:
        clean_viz = clean_cleaned = clean_features = clean_legacy = True
    else:
        clean_viz = 'viz' in targets
        clean_cleaned = 'cleaned' in targets
        clean_features = 'features' in targets
        clean_legacy = 'legacy' in targets
    
    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Define folders to clean
    folders_to_clean = []
    
    if clean_viz:
        folders_to_clean.append({
            'path': os.path.join(project_root, "dataset", "visualizations"),
            'description': 'Visualizations',
            'label': '1'
        })
    
    if clean_cleaned:
        folders_to_clean.append({
            'path': os.path.join(project_root, "dataset", "processed", "cleaned"),
            'description': 'Cleaned Audio',
            'label': '2'
        })
    
    if clean_features:
        folders_to_clean.append({
            'path': os.path.join(project_root, "dataset", "processed", "features"),
            'description': 'Feature Files (.npy)',
            'label': '3'
        })
    
    if clean_legacy:
        folders_to_clean.append({
            'path': os.path.join(project_root, "dataset", "processed", "feature_extraction"),
            'description': 'Legacy Feature Files',
            'label': '4'
        })
    
    # Auto-confirm flag (for backward compatibility)
    auto_confirm = args.confirm or "--confirm" in sys.argv
    # Auto-confirm flag (for backward compatibility)
    auto_confirm = args.confirm or "--confirm" in sys.argv
    
    print("=" * 70)
    print("CryingSense Dataset Cleanup")
    print("=" * 70)
    print()
    
    # Show what will be deleted
    print("The following folders will be cleaned:")
    print()
    
    total_count = 0
    total_size = 0
    
    for i, folder_info in enumerate(folders_to_clean, 1):
        folder_path = folder_info['path']
        file_count = count_files(folder_path)
        folder_size = get_folder_size(folder_path)
        
        print(f"{i}. {folder_info['description']}: {folder_path}")
        print(f"   Files: {file_count} ({format_size(folder_size)})")
        print()
        
        total_count += file_count
        total_size += folder_size
    
    print(f"Total: {total_count} files ({format_size(total_size)})")
    print("=" * 70)
    
    if total_count == 0:
        print("\n✓ All selected folders are already empty. Nothing to clean!")
        return
    
    # Confirmation
    if not auto_confirm:
        print()
        response = input("Are you sure you want to delete all these files? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("\n❌ Cleanup cancelled")
            return
    
    print()
    print("🗑️  Starting cleanup...")
    print()
    
    # Clean folders
    total_deleted_files = 0
    total_deleted_size = 0
    
    for folder_info in folders_to_clean:
        deleted_files, deleted_size = cleanup_folder(
            folder_info['path'], 
            f"{folder_info['description']} folder"
        )
        total_deleted_files += deleted_files
        total_deleted_size += deleted_size
    
    print()
    print("=" * 70)
    print("Cleanup Complete")
    print("=" * 70)
    print(f"Total files removed: {total_deleted_files}")
    print(f"Total space freed: {format_size(total_deleted_size)}")
    print()
    print("✓ Selected folders are now clean and ready for regeneration!")
    print()
    
    # Show relevant next steps based on what was cleaned
    if clean_cleaned or clean_features or 'all' in targets:
        print("Next steps:")
        if clean_cleaned:
            print("  1. Run: python preprocess_audio.py")
        if clean_features:
            print("  2. Run: python feature_extraction.py")
        if clean_viz:
            print("  3. Run: python visualize_dataset.py  (optional)")
        print("  4. Run: python dataset_split.py")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
