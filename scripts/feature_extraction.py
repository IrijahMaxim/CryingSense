"""
Feature Extraction Module for CryingSense

This module extracts acoustic features from audio files for model training:
- MFCC (Mel-Frequency Cepstral Coefficients) - 40 coefficients
- Mel Spectrograms - 128 Mel bands, converted to dB scale
- Chroma features - 12 chroma bins for pitch/harmonic content

Supports both cleaned (preprocessed) and raw audio datasets.
Features are stored as .npy files in:
- dataset/processed/features/mfcc/{category}/
- dataset/processed/features/mel_spectrogram/{category}/
- dataset/processed/features/chroma/{category}/

Each feature is saved with 1:1 mapping to source audio files.

For visualizations, use visualize_dataset.py instead.

Usage:
  python feature_extraction.py              # Extract from cleaned data (default)
  python feature_extraction.py --raw        # Extract from raw data
"""

import os
import numpy as np
import librosa
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


def extract_features(input_dir, output_base_dir, sample_rate=16000, 
                    n_mfcc=40, n_mels=128, n_chroma=12, 
                    n_fft=1024, hop_length=512, duration=5.0):
    """
    Extract MFCC, Mel spectrogram, and Chroma features from audio files.
    
    Features are saved separately in:
    - output_base_dir/mfcc/{category}/
    - output_base_dir/mel_spectrogram/{category}/
    - output_base_dir/chroma/{category}/
    
    Args:
        input_dir: Directory containing audio files
        output_base_dir: Base directory for feature outputs
        sample_rate: Audio sample rate (default: 16000 Hz)
        n_mfcc: Number of MFCC coefficients (default: 40)
        n_mels: Number of Mel bands (default: 128)
        n_chroma: Number of chroma bins (default: 12)
        n_fft: FFT window size (default: 1024)
        hop_length: Number of samples between frames (default: 512)
        duration: Audio duration in seconds (default: 5.0)
    
    Returns:
        dict: Statistics about feature extraction
    """
    # Calculate expected time steps for consistency
    target_time_steps = int(np.ceil((sample_rate * duration) / hop_length))
    
    # Create output directories
    mfcc_dir = os.path.join(output_base_dir, "mfcc")
    mel_dir = os.path.join(output_base_dir, "mel_spectrogram")
    chroma_dir = os.path.join(output_base_dir, "chroma")
    
    stats = {
        'total_files': 0,
        'processed_files': 0,
        'errors': []
    }
    
    # Walk through all audio files
    for root, _, files in os.walk(input_dir):
        category = os.path.basename(root)
        wav_files = [f for f in files if f.endswith('.wav')]
        
        if not wav_files:
            continue
            
        for file in tqdm(wav_files, desc=f"Extracting {category}"):
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
                mfcc = pad_or_crop(mfcc, (n_mfcc, target_time_steps))
                
                mfcc_path = os.path.join(mfcc_dir, base_name)
                os.makedirs(os.path.dirname(mfcc_path), exist_ok=True)
                np.save(mfcc_path, mfcc)
                
                # Extract Mel Spectrogram features
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                                    n_fft=n_fft, hop_length=hop_length)
                mel_db = librosa.power_to_db(mel, ref=np.max)
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
  # Extract features from cleaned data (default)
  python feature_extraction.py
  
  # Extract features from raw data
  python feature_extraction.py --raw
        """
    )
    
    parser.add_argument('--raw', action='store_true',
                       help='Extract features from raw dataset instead of cleaned')
    
    args = parser.parse_args()
    
    # Get paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Define directories
    if args.raw:
        input_dir = os.path.join(project_root, "dataset", "raw")
        output_dir = os.path.join(project_root, "dataset", "processed", "features", "raw")
        dataset_name = "RAW"
    else:
        input_dir = os.path.join(project_root, "dataset", "processed", "cleaned")
        output_dir = os.path.join(project_root, "dataset", "processed", "features", "cleaned")
        dataset_name = "CLEANED"
    
    print("="*70)
    print(f"CryingSense Feature Extraction - {dataset_name}")
    print("="*70)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    print("Feature extraction parameters:")
    print("  • Sample rate: 16000 Hz")
    print("  • MFCC coefficients: 40")
    print("  • Mel bands: 128")
    print("  • Chroma bins: 12")
    print("  • FFT size: 1024")
    print("  • Hop length: 512")
    print("  • Duration: 5.0 seconds")
    print("="*70)
    print()
    
    # Check if input exists
    if not os.path.exists(input_dir):
        print(f"❌ Error: Input directory not found: {input_dir}")
        if not args.raw:
            print("\nPlease run preprocessing first:")
            print("  python preprocess_audio.py")
        sys.exit(1)
    
    # Extract features
    stats = extract_features(input_dir, output_dir)
    
    print()
    print("="*70)
    print("Feature Extraction Complete")
    print("="*70)
    print(f"Total files found: {stats['total_files']}")
    print(f"Successfully processed: {stats['processed_files']}")
    print(f"Errors: {len(stats['errors'])}")
    
    if stats['errors']:
        print("\n❌ Errors encountered:")
        for error in stats['errors'][:10]:
            print(f"  - {error}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more errors")
    
    if stats['processed_files'] > 0:
        print()
        print(f"✓ Features saved to: {output_dir}")
        print()
        print("Features saved in:")
        print(f"  • {os.path.join(output_dir, 'mfcc/')}")
        print(f"  • {os.path.join(output_dir, 'mel_spectrogram/')}")
        print(f"  • {os.path.join(output_dir, 'chroma/')}")
        print()
        print("For visualizations, run:")
        print("  python visualize_dataset.py")
    
    print("="*70)
    
    sys.exit(0 if len(stats['errors']) == 0 else 1)


if __name__ == "__main__":
    main()
