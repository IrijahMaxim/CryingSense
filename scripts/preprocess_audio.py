"""
Audio Preprocessing Module for CryingSense

This module handles the preprocessing of raw audio recordings:
- Loads raw .wav files from dataset/raw/
- Processes both infant cry recordings AND speech samples
- Speech samples (in raw/speech/) are treated as noise for model training
- Applies noise reduction and silence trimming
- Generates 4 versions per audio file:
  1. Normalized only
  2. Normalized and time shifted
  3. Normalized and Background noise added
  4. Normalized, background noise, and time shifted
- Saves cleaned audio to dataset/processed/cleaned/
- Trims or pads all audio to exactly 5 seconds

Categories processed:
- Cry types: belly_pain, burp, discomfort, hunger, tired
- Non-cry: noise (environmental sounds), speech (human speech/babbling)

Time shifting: Random shifts audio by up to 20% of its length
"""

import os
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import noisereduce as nr
import random
import glob


def time_shift_audio(y, shift_max=0.2):
    """
    Shift audio samples along the time axis.
    
    Args:
        y: Audio time series
        shift_max: Maximum shift as a fraction of audio length (default: 0.2 = 20%)
    
    Returns:
        np.ndarray: Time-shifted audio
    """
    shift_amount = int(len(y) * shift_max * random.uniform(-1, 1))
    return np.roll(y, shift_amount)


def add_background_noise(y, noise_dir, sr=16000, noise_factor=0.02):
    """
    Add background noise to audio signal.
    
    Args:
        y: Audio time series
        noise_dir: Path to directory containing noise audio files
        sr: Sample rate
        noise_factor: Factor to control noise level (default: 0.02 = 2%)
    
    Returns:
        np.ndarray: Audio with added background noise
    """
    if not os.path.exists(noise_dir):
        # If noise directory doesn't exist, return original audio
        return y
    
    # Get all noise files
    noise_files = glob.glob(os.path.join(noise_dir, '*.wav'))
    if not noise_files:
        return y
    
    # Randomly select a noise file
    noise_file = random.choice(noise_files)
    
    try:
        # Load noise
        noise, _ = librosa.load(noise_file, sr=sr)
        
        # Make noise same length as audio
        if len(noise) > len(y):
            # Random crop if noise is longer
            start = random.randint(0, len(noise) - len(y))
            noise = noise[start:start + len(y)]
        else:
            # Tile/repeat if noise is shorter
            repeats = int(np.ceil(len(y) / len(noise)))
            noise = np.tile(noise, repeats)[:len(y)]
        
        # Mix audio with noise at specified factor
        noisy_audio = y + noise_factor * noise
        
        # Normalize to prevent clipping
        if np.max(np.abs(noisy_audio)) > 0:
            noisy_audio = noisy_audio / np.max(np.abs(noisy_audio))
        
        return noisy_audio
    except Exception as e:
        # If error loading noise, return original audio
        print(f"\nWarning: Could not load noise from {noise_file}: {e}")
        return y


def preprocess_audio(input_dir, output_dir, noise_dir=None,
                     sample_rate=16000, duration=5.0, top_db=20, shift_max=0.2, noise_factor=0.02):
    """
    Preprocess raw audio files and generate 4 versions per file.
    
    Args:
        input_dir: Path to raw audio directory (e.g., dataset/raw/)
        output_dir: Path for output audio (e.g., dataset/processed/cleaned/)
        noise_dir: Path to background noise directory (e.g., dataset/raw/noise/)
        sample_rate: Target sample rate in Hz (default: 16000)
        duration: Target duration in seconds (default: 5.0)
        top_db: Threshold for silence trimming in dB (default: 20)
        shift_max: Maximum time shift as fraction of length (default: 0.2 = 20%)
        noise_factor: Background noise mixing factor (default: 0.02 = 2%)
    
    Returns:
        dict: Statistics about the preprocessing (total files, errors, etc.)
    """
    # Create output directory
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
                y_base = nr.reduce_noise(y=y, sr=sr)
                
                # Trim or pad to fixed duration (5 seconds)
                target_length = int(sample_rate * duration)
                if len(y_base) > target_length:
                    # Trim from center to preserve most relevant audio
                    start = (len(y_base) - target_length) // 2
                    y_base = y_base[start:start + target_length]
                else:
                    # Pad with zeros to reach target length
                    y_base = np.pad(y_base, (0, target_length - len(y_base)), mode='constant')
                
                # Maintain directory structure and create output paths with tags
                rel_path = os.path.relpath(file_path, input_dir)
                base_name = os.path.splitext(rel_path)[0]
                extension = os.path.splitext(rel_path)[1]
                
                # Version 1: Normalized only
                y_normalized = librosa.util.normalize(y_base)
                out_path_normalized = os.path.join(output_dir, base_name + '_normalized' + extension)
                os.makedirs(os.path.dirname(out_path_normalized), exist_ok=True)
                wavfile.write(out_path_normalized, sample_rate, (y_normalized * 32767).astype(np.int16))
                
                # Version 2: Time shifted only (no normalization)
                y_shifted = time_shift_audio(y_base, shift_max=shift_max)
                out_path_shifted = os.path.join(output_dir, base_name + '_shifted' + extension)
                os.makedirs(os.path.dirname(out_path_shifted), exist_ok=True)
                wavfile.write(out_path_shifted, sample_rate, (y_shifted * 32767).astype(np.int16))
                
                # Version 3: Both normalized and shifted
                y_both = time_shift_audio(y_base, shift_max=shift_max)
                y_both = librosa.util.normalize(y_both)
                out_path_both = os.path.join(output_dir, base_name + '_both' + extension)
                os.makedirs(os.path.dirname(out_path_both), exist_ok=True)
                wavfile.write(out_path_both, sample_rate, (y_both * 32767).astype(np.int16))
                
                # Version 4: Normalized with background noise
                if noise_dir:
                    y_noise = add_background_noise(y_base, noise_dir, sr=sample_rate, noise_factor=noise_factor)
                    y_noise = librosa.util.normalize(y_noise)
                    out_path_noise = os.path.join(output_dir, base_name + '_noise' + extension)
                    os.makedirs(os.path.dirname(out_path_noise), exist_ok=True)
                    wavfile.write(out_path_noise, sample_rate, (y_noise * 32767).astype(np.int16))
                    
                    # Version 5: Normalized, background noise, and time shifted
                    y_noise_shifted = time_shift_audio(y_noise, shift_max=shift_max)
                    y_noise_shifted = librosa.util.normalize(y_noise_shifted)
                    out_path_noise_shifted = os.path.join(output_dir, base_name + '_noise_shifted' + extension)
                    os.makedirs(os.path.dirname(out_path_noise_shifted), exist_ok=True)
                    wavfile.write(out_path_noise_shifted, sample_rate, (y_noise_shifted * 32767).astype(np.int16))
                
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
    noise_dir = os.path.join(project_root, "dataset", "raw", "noise")
    
    # Check if noise directory exists
    use_noise = os.path.exists(noise_dir)
    if not use_noise:
        print(f"\n⚠️  Warning: Noise directory not found: {noise_dir}")
        print("Background noise augmentation will be skipped.\n")
        noise_dir = None
    
    print("="*60)
    print("CryingSense Audio Preprocessing")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    if use_noise:
        print(f"Noise directory: {noise_dir}")
        print(f"Output tags: _normalized, _shifted, _both, _noise, _noise_shifted")
    else:
        print(f"Output tags: _normalized, _shifted, _both")
    print(f"Sample rate: 16000 Hz")
    print(f"Duration: 5.0 seconds")
    print(f"Time shift: up to 20% of audio length")
    if use_noise:
        print(f"Background noise: 2% mixing factor")
    print()
    print("Processing categories:")
    print("  • Cry types: belly_pain, burp, discomfort, hunger, tired")
    print("  • Non-cry: noise, speech (treated as noise by model)")
    print("="*60)
    print()
    
    # Run preprocessing to generate augmented versions
    stats = preprocess_audio(input_dir, output_dir, noise_dir=noise_dir)
    
    versions_per_file = 5 if use_noise else 3
    
    print()
    print("="*60)
    print("Preprocessing Complete")
    print("="*60)
    print(f"Total files found: {stats['total_files']}")
    print(f"Successfully processed: {stats['processed_files']}")
    print(f"Generated versions per input: {versions_per_file}")
    if use_noise:
        print(f"  - normalized, shifted, both, noise, noise_shifted")
    else:
        print(f"  - normalized, shifted, both")
    print(f"Total output files: {stats['processed_files'] * versions_per_file}")
    print(f"Errors: {len(stats['errors'])}")
    
    if stats['errors']:
        print("\nErrors encountered:")
        for error in stats['errors']:
            print(f"  - {error}")
        sys.exit(1)
    
    print("="*60)


if __name__ == "__main__":
    main()
