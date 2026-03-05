"""
Dataset Cleanup Script for CryingSense

This script removes all files from:
- dataset/visualizations/ (all visualization images)
- dataset/processed/cleaned/ (all cleaned audio files)
- dataset/processed/features/ (all extracted .npy feature files)
- dataset/processed/feature_extraction/ (legacy feature files)

Use this before regenerating processed data and visualizations.

Usage:
    python cleanup_dataset.py [--confirm]
    
Options:
    --confirm    Skip confirmation prompt and delete immediately
"""

import os
import shutil
import sys
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
    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Define folders to clean
    visualizations_dir = os.path.join(project_root, "dataset", "visualizations")
    cleaned_dir = os.path.join(project_root, "dataset", "processed", "cleaned")
    features_dir = os.path.join(project_root, "dataset", "processed", "features")
    legacy_features_dir = os.path.join(project_root, "dataset", "processed", "feature_extraction")
    
    # Check for --confirm flag
    auto_confirm = "--confirm" in sys.argv
    
    print("=" * 70)
    print("CryingSense Dataset Cleanup")
    print("=" * 70)
    print()
    
    # Show what will be deleted
    print("The following folders will be cleaned:")
    print()
    
    viz_count = count_files(visualizations_dir)
    viz_size = get_folder_size(visualizations_dir)
    print(f"1. Visualizations: {visualizations_dir}")
    print(f"   Files: {viz_count} ({format_size(viz_size)})")
    print()
    
    cleaned_count = count_files(cleaned_dir)
    cleaned_size = get_folder_size(cleaned_dir)
    print(f"2. Cleaned Audio: {cleaned_dir}")
    print(f"   Files: {cleaned_count} ({format_size(cleaned_size)})")
    print()
    
    features_count = count_files(features_dir)
    features_size = get_folder_size(features_dir)
    print(f"3. Feature Files (.npy): {features_dir}")
    print(f"   Files: {features_count} ({format_size(features_size)})")
    print()
    
    legacy_features_count = count_files(legacy_features_dir)
    legacy_features_size = get_folder_size(legacy_features_dir)
    if legacy_features_count > 0:
        print(f"4. Legacy Feature Files: {legacy_features_dir}")
        print(f"   Files: {legacy_features_count} ({format_size(legacy_features_size)})")
        print()
    
    total_count = viz_count + cleaned_count + features_count + legacy_features_count
    total_size = viz_size + cleaned_size + features_size + legacy_features_size
    
    print(f"Total: {total_count} files ({format_size(total_size)})")
    print("=" * 70)
    
    if total_count == 0:
        print("\n✓ All folders are already empty. Nothing to clean!")
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
    
    deleted_files, deleted_size = cleanup_folder(visualizations_dir, "Visualizations folder")
    total_deleted_files += deleted_files
    total_deleted_size += deleted_size
    
    deleted_files, deleted_size = cleanup_folder(cleaned_dir, "Cleaned audio folder")
    total_deleted_files += deleted_files
    total_deleted_size += deleted_size
    
    deleted_files, deleted_size = cleanup_folder(features_dir, "Feature files folder")
    total_deleted_files += deleted_files
    total_deleted_size += deleted_size
    
    deleted_files, deleted_size = cleanup_folder(legacy_features_dir, "Legacy feature files folder")
    total_deleted_files += deleted_files
    total_deleted_size += deleted_size
    
    print()
    print("=" * 70)
    print("Cleanup Complete")
    print("=" * 70)
    print(f"Total files removed: {total_deleted_files}")
    print(f"Total space freed: {format_size(total_deleted_size)}")
    print()
    print("✓ Folders are now clean and ready for regeneration!")
    print()
    print("Next steps:")
    print("  1. Run: python preprocess_audio.py")
    print("  2. Run: python feature_extraction.py")
    print("  3. Run: python visualize_dataset.py  (optional)")
    print("  4. Run: python dataset_split.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
