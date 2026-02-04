"""
Feature Extraction Module for CryingSense

This module extracts acoustic features from preprocessed audio:
- MFCC (Mel-Frequency Cepstral Coefficients) - 40 coefficients
- Mel Spectrograms - 128 Mel bands, converted to dB scale
- Chroma features - 12 chroma bins for pitch/harmonic content

Features are extracted from both cleaned and raw audio, stored separately
in dataset/processed/feature_extraction/ with subdirectories for each feature type.
Each feature is saved as a .npy file with 1:1 mapping to source audio.
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
    - output_base_dir/mfcc/
    - output_base_dir/mel_spectrogram/
    - output_base_dir/chroma/
    
    Args:
        input_dir: Directory containing cleaned .wav files
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
    
    os.makedirs(mfcc_dir, exist_ok=True)
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(chroma_dir, exist_ok=True)
    
    stats = {
        'total_files': 0,
        'processed_files': 0,
        'errors': []
    }
    
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
    
    # Get paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    input_dir = os.path.join(project_root, "dataset", "processed", "cleaned")
    output_dir = os.path.join(project_root, "dataset", "processed", 
                             "feature_extraction", "cleaned")
    
    print("="*60)
    print("CryingSense Feature Extraction")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Sample rate: 16000 Hz")
    print(f"MFCC coefficients: 40")
    print(f"Mel bands: 128")
    print(f"Chroma bins: 12")
    print(f"FFT size: 1024")
    print(f"Hop length: 512")
    print("="*60)
    print()
    
    # Extract features
    stats = extract_features(input_dir, output_dir)
    
    print()
    print("="*60)
    print("Feature Extraction Complete")
    print("="*60)
    print(f"Total files found: {stats['total_files']}")
    print(f"Successfully processed: {stats['processed_files']}")
    print(f"Errors: {len(stats['errors'])}")
    
    if stats['errors']:
        print("\nErrors encountered:")
        for error in stats['errors']:
            print(f"  - {error}")
        sys.exit(1)
    
    print("="*60)


if __name__ == "__main__":
    main()
