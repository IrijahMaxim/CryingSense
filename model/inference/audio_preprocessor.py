"""
Real-time Audio Preprocessing for CryingSense Inference

Handles preprocessing of audio input for inference:
- Loads and resamples audio
- Applies noise reduction
- Trims silence
- Normalizes amplitude
- Pads or crops to target duration
"""

import os
import numpy as np
import librosa
import soundfile as sf


class AudioPreprocessor:
    """Real-time audio preprocessing for inference."""
    
    def __init__(self, sample_rate=16000, duration=5.0, top_db=20, 
                 apply_noise_reduction=True):
        """
        Initialize audio preprocessor.
        
        Args:
            sample_rate: Target sample rate in Hz (default: 16000)
            duration: Target duration in seconds (default: 5.0)
            top_db: Threshold for silence trimming in dB (default: 20)
            apply_noise_reduction: Whether to apply noise reduction (default: True)
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.top_db = top_db
        self.apply_noise_reduction = apply_noise_reduction
        self.target_length = int(sample_rate * duration)
    
    def load_audio(self, audio_path):
        """
        Load audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Loaded audio signal, sample rate
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load audio with librosa
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        return y, sr
    
    def reduce_noise(self, audio, sr):
        """
        Apply noise reduction to audio signal.
        
        Args:
            audio: Audio signal
            sr: Sample rate
        
        Returns:
            Noise-reduced audio
        """
        try:
            import noisereduce as nr
            # Use first 0.5 seconds as noise profile
            noise_sample_length = min(int(0.5 * sr), len(audio) // 4)
            reduced = nr.reduce_noise(
                y=audio, 
                sr=sr,
                stationary=True,
                prop_decrease=0.8
            )
            return reduced
        except ImportError:
            print("Warning: noisereduce not installed. Skipping noise reduction.")
            print("Install with: pip install noisereduce")
            return audio
        except Exception as e:
            print(f"Warning: Noise reduction failed: {e}")
            return audio
    
    def trim_silence(self, audio, sr):
        """
        Trim silence from audio.
        
        Args:
            audio: Audio signal
            sr: Sample rate
        
        Returns:
            Trimmed audio
        """
        trimmed, _ = librosa.effects.trim(audio, top_db=self.top_db)
        
        # Ensure minimum length
        if len(trimmed) < sr * 0.5:  # Less than 0.5 seconds
            return audio  # Return original if too short after trimming
        
        return trimmed
    
    def normalize_audio(self, audio):
        """
        Normalize audio amplitude.
        
        Args:
            audio: Audio signal
        
        Returns:
            Normalized audio
        """
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.9  # Normalize to 90% to avoid clipping
        return audio
    
    def pad_or_crop(self, audio):
        """
        Pad or crop audio to target length.
        
        Args:
            audio: Audio signal
        
        Returns:
            Audio with target length
        """
        if len(audio) > self.target_length:
            # Crop from center
            start = (len(audio) - self.target_length) // 2
            audio = audio[start:start + self.target_length]
        elif len(audio) < self.target_length:
            # Pad with zeros
            padding = self.target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        
        return audio
    
    def preprocess(self, audio_path_or_array):
        """
        Complete preprocessing pipeline.
        
        Args:
            audio_path_or_array: Path to audio file or numpy array
        
        Returns:
            Preprocessed audio array
        """
        # Load audio if path provided
        if isinstance(audio_path_or_array, str):
            audio, sr = self.load_audio(audio_path_or_array)
        else:
            audio = audio_path_or_array
            sr = self.sample_rate
        
        # Apply noise reduction if enabled
        if self.apply_noise_reduction:
            audio = self.reduce_noise(audio, sr)
        
        # Trim silence
        audio = self.trim_silence(audio, sr)
        
        # Normalize amplitude
        audio = self.normalize_audio(audio)
        
        # Pad or crop to target duration
        audio = self.pad_or_crop(audio)
        
        return audio
    
    def preprocess_streaming(self, audio_chunk, sr=None):
        """
        Preprocess audio chunk for streaming inference.
        
        Args:
            audio_chunk: Audio chunk array
            sr: Sample rate (uses instance default if not provided)
        
        Returns:
            Preprocessed audio chunk
        """
        if sr is None:
            sr = self.sample_rate
        
        # Basic preprocessing without trimming
        if self.apply_noise_reduction:
            audio_chunk = self.reduce_noise(audio_chunk, sr)
        
        audio_chunk = self.normalize_audio(audio_chunk)
        audio_chunk = self.pad_or_crop(audio_chunk)
        
        return audio_chunk
    
    def save_preprocessed(self, audio, output_path):
        """
        Save preprocessed audio to file.
        
        Args:
            audio: Preprocessed audio array
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, audio, self.sample_rate)
        print(f"Preprocessed audio saved to: {output_path}")


if __name__ == "__main__":
    # Test audio preprocessing
    import argparse
    
    parser = argparse.ArgumentParser(description='Test audio preprocessing')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--output', type=str, default=None, help='Output path (optional)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("CryingSense Audio Preprocessor Test")
    print("="*70)
    print(f"Input: {args.audio}")
    print(f"Sample rate: 16000 Hz")
    print(f"Target duration: 5.0 seconds")
    print("="*70)
    
    preprocessor = AudioPreprocessor()
    
    # Preprocess audio
    audio = preprocessor.preprocess(args.audio)
    
    print(f"\nPreprocessed audio shape: {audio.shape}")
    print(f"Audio duration: {len(audio) / 16000:.2f} seconds")
    print(f"Audio range: [{audio.min():.4f}, {audio.max():.4f}]")
    
    if args.output:
        preprocessor.save_preprocessed(audio, args.output)
    
    print("\n" + "="*70)
    print("Preprocessing complete!")
    print("="*70)
