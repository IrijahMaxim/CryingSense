"""
Audio Preprocessing Module for CryingSense

This module handles the preprocessing of raw infant cry audio recordings:
- Loads raw .wav files from dataset/raw/
- Applies noise reduction and silence trimming
- Normalizes amplitude to consistent levels
- Trims or pads all audio to exactly 5 seconds
- Saves cleaned audio to dataset/processed/cleaned/ with 1:1 mapping

Note: Data augmentation is NOT applied during preprocessing to avoid dataset inflation.
Augmentation is applied during training time only.
"""

import os
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import noisereduce as nr


def preprocess_audio(input_dir, output_dir, sample_rate=16000, duration=5.0, top_db=20):
    """
    Preprocess raw audio files for CryingSense model training.
    
    Args:
        input_dir: Path to raw audio directory (e.g., dataset/raw/)
        output_dir: Path to output cleaned audio (e.g., dataset/processed/cleaned/)
        sample_rate: Target sample rate in Hz (default: 16000)
        duration: Target duration in seconds (default: 5.0)
        top_db: Threshold for silence trimming in dB (default: 20)
    
    Returns:
        dict: Statistics about the preprocessing (total files, errors, etc.)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    stats = {
        'total_files': 0,
        'processed_files': 0,
        'errors': []
    }
    
    # Walk through all subdirectories
    for root, _, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Processing {os.path.basename(root)}"):
            if not file.endswith('.wav'):
                continue
            
            stats['total_files'] += 1
            file_path = os.path.join(root, file)
            
            try:
                # Load audio at target sample rate
                y, sr = librosa.load(file_path, sr=sample_rate)
                
                # Trim silence from beginning and end
                y, _ = librosa.effects.trim(y, top_db=top_db)
                
                # Apply noise reduction
                y = nr.reduce_noise(y=y, sr=sr)
                
                # Normalize amplitude to [-1, 1] range
                y = librosa.util.normalize(y)
                
                # Trim or pad to fixed duration (5 seconds)
                target_length = int(sample_rate * duration)
                if len(y) > target_length:
                    # Trim from center to preserve most relevant audio
                    start = (len(y) - target_length) // 2
                    y = y[start:start + target_length]
                else:
                    # Pad with zeros to reach target length
                    y = np.pad(y, (0, target_length - len(y)), mode='constant')
                
                # Maintain directory structure and filename
                rel_path = os.path.relpath(file_path, input_dir)
                out_path = os.path.join(output_dir, rel_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                
                # Save as 16-bit PCM WAV file
                wavfile.write(out_path, sample_rate, (y * 32767).astype(np.int16))
                
                stats['processed_files'] += 1
                
            except Exception as e:
                error_msg = f"Error processing {file_path}: {str(e)}"
                stats['errors'].append(error_msg)
                print(f"\n{error_msg}")
    
    return stats


def main():
    """Main function to run preprocessing pipeline."""
    import sys
    
    # Get paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    input_dir = os.path.join(project_root, "dataset", "raw")
    output_dir = os.path.join(project_root, "dataset", "processed", "cleaned")
    
    print("="*60)
    print("CryingSense Audio Preprocessing")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Sample rate: 16000 Hz")
    print(f"Duration: 5.0 seconds")
    print("="*60)
    print()
    
    # Run preprocessing without augmentation (augmentation done at training time)
    stats = preprocess_audio(input_dir, output_dir)
    
    print()
    print("="*60)
    print("Preprocessing Complete")
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
