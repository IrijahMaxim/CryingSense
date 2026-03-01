"""
Feature Extraction Module for CryingSense

This module extracts acoustic features from audio files:
- MFCC (Mel-Frequency Cepstral Coefficients) - 40 coefficients
- Mel Spectrograms - 128 Mel bands, converted to dB scale
- Chroma features - 12 chroma bins for pitch/harmonic content

Supports both cleaned (preprocessed) and raw audio datasets.
Features are stored in:
- dataset/processed/feature_extraction/cleaned/ (for cleaned audio)
- dataset/processed/feature_extraction/raw/ (for raw audio)

Visualizations are automatically saved as PNG files for a random subset (~100 samples) in:
- dataset/visualizations/waveforms/
- dataset/visualizations/mel_spectrograms/
- dataset/visualizations/mfcc/

Note: Visualizations exclude the 'noise' category and are distributed evenly across other classes.

Each feature type is saved in subdirectories (mfcc/, mel_spectrogram/, chroma/).
Each feature is saved as a .npy file with 1:1 mapping to source audio.

Usage:
  python feature_extraction.py              # Extract from cleaned data (default)
  python feature_extraction.py --raw-only   # Extract from raw data only
  python feature_extraction.py --include-raw # Extract from both datasets
"""

import os
import random
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for saving plots
import matplotlib.pyplot as plt
from tqdm import tqdm


def pad_or_crop(feature, target_shape):
    """
    Resize feature array to target shape by padding with zeros or cropping.
    
    Args:
        feature: Input feature array (n_features, time_steps)
        target_shape: Desired output shape (n_features, time_steps)
    
    Returns:
        Resized feature array
    """
    padded = np.zeros(target_shape, dtype=feature.dtype)
    min_shape = (min(feature.shape[0], target_shape[0]), 
                 min(feature.shape[1], target_shape[1]))
    padded[:min_shape[0], :min_shape[1]] = feature[:min_shape[0], :min_shape[1]]
    return padded


def save_visualizations(y, sr, mfcc, mel_db, chroma, file_name, category, viz_base_dir, hop_length=512):
    """
    Save feature visualizations as PNG images.
    
    Args:
        y: Audio time series
        sr: Sample rate
        mfcc: MFCC features (before padding/cropping)
        mel_db: Mel spectrogram in dB (before padding/cropping)
        chroma: Chroma features
        file_name: Base filename (without extension)
        category: Audio category (e.g., 'hunger', 'tired')
        viz_base_dir: Base directory for visualizations (dataset/visualizations)
        hop_length: Hop length used for feature extraction
    """
    try:
        # Create output paths
        waveform_path = os.path.join(viz_base_dir, "waveforms", f"{category}_{file_name}_waveform.png")
        mel_path = os.path.join(viz_base_dir, "mel_spectrograms", f"{category}_{file_name}_mel_spectrogram.png")
        mfcc_path = os.path.join(viz_base_dir, "mfcc", f"{category}_{file_name}_mfcc.png")
        
        os.makedirs(os.path.dirname(waveform_path), exist_ok=True)
        os.makedirs(os.path.dirname(mel_path), exist_ok=True)
        os.makedirs(os.path.dirname(mfcc_path), exist_ok=True)
        
        # 1. Waveform visualization
        plt.figure(figsize=(12, 4))
        time_axis = np.linspace(0, len(y)/sr, len(y))
        plt.plot(time_axis, y, linewidth=0.5, color='#1f77b4')
        plt.xlabel('Time (s)', fontsize=10)
        plt.ylabel('Amplitude', fontsize=10)
        plt.title(f'Waveform - {category.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, time_axis[-1]])
        plt.tight_layout()
        plt.savefig(waveform_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # 2. Mel Spectrogram visualization
        plt.figure(figsize=(12, 6))
        img = librosa.display.specshow(mel_db, sr=sr, hop_length=hop_length, 
                                       x_axis='time', y_axis='mel', cmap='viridis')
        plt.colorbar(img, format='%+2.0f dB')
        plt.xlabel('Time (s)', fontsize=10)
        plt.ylabel('Mel Frequency (Hz)', fontsize=10)
        plt.title(f'Mel Spectrogram - {category.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(mel_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # 3. MFCC visualization
        plt.figure(figsize=(12, 6))
        img = librosa.display.specshow(mfcc, sr=sr, hop_length=hop_length,
                                       x_axis='time', cmap='coolwarm')
        plt.colorbar(img, format='%+2.0f')
        plt.xlabel('Time (s)', fontsize=10)
        plt.ylabel('MFCC Coefficients', fontsize=10)
        plt.title(f'MFCC Features - {category.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(mfcc_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        print(f"Warning: Could not save visualization for {file_name}: {str(e)}")
        return False


def extract_features(input_dir, output_base_dir, viz_base_dir=None, sample_rate=16000, 
                    n_mfcc=40, n_mels=128, n_chroma=12, 
                    n_fft=1024, hop_length=512, duration=5.0, max_viz_samples=100):
    """
    Extract MFCC, Mel spectrogram, and Chroma features from audio files.
    
    Features are saved separately in:
    - output_base_dir/mfcc/
    - output_base_dir/mel_spectrogram/
    - output_base_dir/chroma/
    
    If viz_base_dir is provided, also saves PNG visualizations to:
    - viz_base_dir/waveforms/
    - viz_base_dir/mel_spectrograms/
    - viz_base_dir/mfcc/
    
    Args:
        input_dir: Directory containing cleaned .wav files
        output_base_dir: Base directory for feature outputs
        viz_base_dir: Base directory for visualization outputs (optional)
        sample_rate: Audio sample rate (default: 16000 Hz)
        n_mfcc: Number of MFCC coefficients (default: 40)
        n_mels: Number of Mel bands (default: 128)
        n_chroma: Number of chroma bins (default: 12)
        n_fft: FFT window size (default: 1024)
        hop_length: Number of samples between frames (default: 512)
        duration: Audio duration in seconds (default: 5.0)
        max_viz_samples: Maximum number of samples to visualize (default: 100)
    
    Returns:
        dict: Statistics about feature extraction
    """
    # Calculate expected time steps for consistency
    target_time_steps = int(np.ceil((sample_rate * duration) / hop_length))
    
    # Create output directories
    mfcc_dir = os.path.join(output_base_dir, "mfcc")
    mel_dir = os.path.join(output_base_dir, "mel_spectrogram")
    chroma_dir = os.path.join(output_base_dir, "chroma")
    
    os.makedirs(mfcc_dir, exist_ok=True)
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(chroma_dir, exist_ok=True)
    
    stats = {
        'total_files': 0,
        'processed_files': 0,
        'visualizations_saved': 0,
        'errors': []
    }
    
    # First pass: Collect all audio files by category
    files_by_category = {}
    all_files = []
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.endswith('.wav'):
                continue
            
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, input_dir)
            category = os.path.dirname(rel_path).split(os.sep)[0] if os.sep in rel_path else 'unknown'
            
            if category not in files_by_category:
                files_by_category[category] = []
            files_by_category[category].append(file_path)
            all_files.append(file_path)
    
    # Select random files for visualization (excluding noise category)
    viz_files = set()
    if viz_base_dir and max_viz_samples > 0:
        # Filter out noise category
        categories_to_visualize = {k: v for k, v in files_by_category.items() 
                                   if k.lower() != 'noise'}
        
        if categories_to_visualize:
            # Calculate samples per category (distribute evenly)
            num_categories = len(categories_to_visualize)
            samples_per_category = max(1, max_viz_samples // num_categories)
            
            print(f"\nSelecting {max_viz_samples} random samples for visualization:")
            for category, category_files in categories_to_visualize.items():
                # Randomly select files from this category
                num_to_select = min(samples_per_category, len(category_files))
                selected = random.sample(category_files, num_to_select)
                viz_files.update(selected)
                print(f"  {category}: {num_to_select}/{len(category_files)} files")
            
            print(f"Total files selected for visualization: {len(viz_files)}")
            print()
    
    # Walk through all audio files
    for root, _, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Extracting {os.path.basename(root)}"):
            if not file.endswith('.wav'):
                continue
            
            stats['total_files'] += 1
            file_path = os.path.join(root, file)
            
            try:
                # Load audio
                y, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
                
                # Get relative path to maintain directory structure
                rel_path = os.path.relpath(file_path, input_dir)
                base_name = os.path.splitext(rel_path)[0] + '.npy'
                
                # Extract MFCC features
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, 
                                           n_fft=n_fft, hop_length=hop_length)
                mfcc_original = mfcc.copy()  # Save original for visualization
                mfcc = pad_or_crop(mfcc, (n_mfcc, target_time_steps))
                
                mfcc_path = os.path.join(mfcc_dir, base_name)
                os.makedirs(os.path.dirname(mfcc_path), exist_ok=True)
                np.save(mfcc_path, mfcc)
                
                # Extract Mel Spectrogram features
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                                    n_fft=n_fft, hop_length=hop_length)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                mel_db_original = mel_db.copy()  # Save original for visualization
                mel_db = pad_or_crop(mel_db, (n_mels, target_time_steps))
                
                mel_path = os.path.join(mel_dir, base_name)
                os.makedirs(os.path.dirname(mel_path), exist_ok=True)
                np.save(mel_path, mel_db)
                
                # Extract Chroma features
                chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma,
                                                    n_fft=n_fft, hop_length=hop_length)
                chroma = pad_or_crop(chroma, (n_chroma, target_time_steps))
                
                chroma_path = os.path.join(chroma_dir, base_name)
                os.makedirs(os.path.dirname(chroma_path), exist_ok=True)
                np.save(chroma_path, chroma)
                
                stats['processed_files'] += 1
                
                # Save visualizations only for selected files
                if viz_base_dir and file_path in viz_files:
                    # Extract category from relative path (e.g., 'hunger', 'tired')
                    category = os.path.dirname(rel_path).split(os.sep)[0] if os.sep in rel_path else 'unknown'
                    file_base = os.path.splitext(file)[0]
                    
                    # Visualize original (non-padded) features
                    if save_visualizations(y, sr, mfcc_original, mel_db_original, chroma, 
                                         file_base, category, viz_base_dir, hop_length):
                        stats['visualizations_saved'] += 1
                
            except Exception as e:
                error_msg = f"Error processing {file_path}: {str(e)}"
                stats['errors'].append(error_msg)
                print(f"\n{error_msg}")
    
    return stats


def main():
    """Main function to run feature extraction pipeline."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract audio features for CryingSense',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract features from cleaned data only (default)
  python feature_extraction.py
  
  # Extract features from raw data only
  python feature_extraction.py --raw-only
  
  # Extract features from both cleaned and raw data
  python feature_extraction.py --include-raw
        """
    )
    
    parser.add_argument('--include-raw', action='store_true',
                       help='Also extract features from raw dataset')
    parser.add_argument('--raw-only', action='store_true',
                       help='Only extract features from raw dataset')
    
    args = parser.parse_args()
    
    # Get paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Define directories
    cleaned_input = os.path.join(project_root, "dataset", "processed", "cleaned")
    cleaned_output = os.path.join(project_root, "dataset", "processed", 
                                  "feature_extraction", "cleaned")
    
    raw_input = os.path.join(project_root, "dataset", "raw")
    raw_output = os.path.join(project_root, "dataset", "processed", 
                              "feature_extraction", "raw")
    
    # Visualization directory (shared for both)
    viz_dir = os.path.join(project_root, "dataset", "visualizations")
    
    # Determine which datasets to process
    datasets_to_process = []
    if args.raw_only:
        datasets_to_process = [('raw', raw_input, raw_output)]
    elif args.include_raw:
        datasets_to_process = [
            ('cleaned', cleaned_input, cleaned_output),
            ('raw', raw_input, raw_output)
        ]
    else:
        datasets_to_process = [('cleaned', cleaned_input, cleaned_output)]
    
    all_stats = []
    
    for dataset_name, input_dir, output_dir in datasets_to_process:
        print("="*60)
        print(f"CryingSense Feature Extraction - {dataset_name.upper()}")
        print("="*60)
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Visualization directory: {viz_dir}")
        print(f"Sample rate: 16000 Hz")
        print(f"MFCC coefficients: 40")
        print(f"Mel bands: 128")
        print(f"Chroma bins: 12")
        print(f"FFT size: 1024")
        print(f"Hop length: 512")
        print("="*60)
        print()
        
        # Check if input exists
        if not os.path.exists(input_dir):
            print(f"Warning: Input directory not found: {input_dir}")
            print(f"Skipping {dataset_name} dataset.")
            print()
            continue
        
        # Extract features
        stats = extract_features(input_dir, output_dir, viz_base_dir=viz_dir)
        stats['dataset'] = dataset_name
        all_stats.append(stats)
        
        print()
        print("="*60)
        print(f"Feature Extraction Complete - {dataset_name.upper()}")
        print("="*60)
        print(f"Total files found: {stats['total_files']}")
        print(f"Successfully processed: {stats['processed_files']}")
        print(f"Visualizations saved: {stats['visualizations_saved']}")
        print(f"Errors: {len(stats['errors'])}")
        
        if stats['errors']:
            print("\nErrors encountered:")
            for error in stats['errors'][:10]:  # Limit error output
                print(f"  - {error}")
            if len(stats['errors']) > 10:
                print(f"  ... and {len(stats['errors']) - 10} more errors")
        
        print("="*60)
        print()
    
    # Summary
    if len(all_stats) > 1:
        print("\n" + "="*60)
        print("OVERALL SUMMARY")
        print("="*60)
        total_files = sum(s['total_files'] for s in all_stats)
        total_processed = sum(s['processed_files'] for s in all_stats)
        total_visualizations = sum(s['visualizations_saved'] for s in all_stats)
        total_errors = sum(len(s['errors']) for s in all_stats)
        print(f"Total files processed: {total_processed}/{total_files}")
        print(f"Total visualizations saved: {total_visualizations}")
        print(f"Total errors: {total_errors}")
        print("="*60)
    
    # Exit with error if any failures
    if any(s['errors'] for s in all_stats):
        sys.exit(1)


if __name__ == "__main__":
    main()
