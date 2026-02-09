"""
Real-time Feature Extraction for CryingSense Inference

Extracts acoustic features from preprocessed audio for model input:
- MFCC (Mel-Frequency Cepstral Coefficients)
- Mel Spectrograms
- Chroma features
- Combines features into 4-channel input
"""

import numpy as np
import librosa


class FeatureExtractor:
    """Real-time feature extraction for inference."""
    
    def __init__(self, sample_rate=16000, n_mfcc=40, n_mels=128, n_chroma=12,
                 n_fft=1024, hop_length=512, duration=5.0):
        """
        Initialize feature extractor.
        
        Args:
            sample_rate: Audio sample rate (default: 16000 Hz)
            n_mfcc: Number of MFCC coefficients (default: 40)
            n_mels: Number of Mel bands (default: 128)
            n_chroma: Number of chroma bins (default: 12)
            n_fft: FFT window size (default: 1024)
            hop_length: Number of samples between frames (default: 512)
            duration: Audio duration in seconds (default: 5.0)
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_chroma = n_chroma
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        
        # Calculate target time steps for consistency
        self.target_time_steps = int(np.ceil((sample_rate * duration) / hop_length))
        
        # Target height for combined features (max of all feature dims)
        self.target_height = max(n_mfcc, n_mels, n_chroma)
    
    def pad_or_crop(self, feature, target_shape):
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
    
    def extract_mfcc(self, audio):
        """
        Extract MFCC features.
        
        Args:
            audio: Audio signal array
        
        Returns:
            MFCC feature array (n_mfcc, time_steps)
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        mfcc = self.pad_or_crop(mfcc, (self.n_mfcc, self.target_time_steps))
        return mfcc
    
    def extract_mel_spectrogram(self, audio):
        """
        Extract Mel Spectrogram features.
        
        Args:
            audio: Audio signal array
        
        Returns:
            Mel Spectrogram feature array in dB (n_mels, time_steps)
        """
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = self.pad_or_crop(mel_db, (self.n_mels, self.target_time_steps))
        return mel_db
    
    def extract_chroma(self, audio):
        """
        Extract Chroma features.
        
        Args:
            audio: Audio signal array
        
        Returns:
            Chroma feature array (n_chroma, time_steps)
        """
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_chroma=self.n_chroma,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        chroma = self.pad_or_crop(chroma, (self.n_chroma, self.target_time_steps))
        return chroma
    
    def extract_all_features(self, audio):
        """
        Extract all features (MFCC, Mel, Chroma).
        
        Args:
            audio: Audio signal array
        
        Returns:
            Dictionary containing all features
        """
        mfcc = self.extract_mfcc(audio)
        mel = self.extract_mel_spectrogram(audio)
        chroma = self.extract_chroma(audio)
        
        return {
            'mfcc': mfcc,
            'mel_spectrogram': mel,
            'chroma': chroma
        }
    
    def combine_features(self, features_dict):
        """
        Combine multiple features into a single multi-channel array.
        
        Creates a 4-channel input by:
        - Channel 0: MFCC (padded to target_height)
        - Channel 1: Mel Spectrogram (padded to target_height)
        - Channel 2: Chroma (padded to target_height)
        - Channel 3: Delta MFCC (padded to target_height)
        
        Args:
            features_dict: Dictionary with 'mfcc', 'mel_spectrogram', 'chroma' keys
        
        Returns:
            Combined feature array (4, height, width)
        """
        mfcc = features_dict['mfcc']
        mel = features_dict['mel_spectrogram']
        chroma = features_dict['chroma']
        
        # Pad features to target height
        mfcc_padded = self.pad_or_crop(mfcc, (self.target_height, self.target_time_steps))
        mel_padded = self.pad_or_crop(mel, (self.target_height, self.target_time_steps))
        chroma_padded = self.pad_or_crop(chroma, (self.target_height, self.target_time_steps))
        
        # Calculate delta MFCC
        delta_mfcc = librosa.feature.delta(mfcc)
        delta_mfcc_padded = self.pad_or_crop(delta_mfcc, (self.target_height, self.target_time_steps))
        
        # Stack into 4-channel array
        combined = np.stack([
            mfcc_padded,
            mel_padded,
            chroma_padded,
            delta_mfcc_padded
        ], axis=0)
        
        return combined
    
    def extract_features_for_inference(self, audio):
        """
        Complete feature extraction pipeline for model inference.
        
        Args:
            audio: Preprocessed audio signal array
        
        Returns:
            Combined feature array ready for model input (4, height, width)
        """
        # Extract all features
        features = self.extract_all_features(audio)
        
        # Combine into multi-channel array
        combined = self.combine_features(features)
        
        return combined
    
    def get_feature_shape(self):
        """
        Get the output shape of extracted features.
        
        Returns:
            Tuple (channels, height, width)
        """
        return (4, self.target_height, self.target_time_steps)


if __name__ == "__main__":
    # Test feature extraction
    import argparse
    
    parser = argparse.ArgumentParser(description='Test feature extraction')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--save-features', action='store_true', 
                       help='Save extracted features to .npy file')
    
    args = parser.parse_args()
    
    print("="*70)
    print("CryingSense Feature Extractor Test")
    print("="*70)
    print(f"Input: {args.audio}")
    print(f"Sample rate: 16000 Hz")
    print(f"MFCC coefficients: 40")
    print(f"Mel bands: 128")
    print(f"Chroma bins: 12")
    print("="*70)
    
    # Load audio
    audio, sr = librosa.load(args.audio, sr=16000)
    print(f"\nLoaded audio: {len(audio)} samples ({len(audio)/16000:.2f} seconds)")
    
    # Extract features
    extractor = FeatureExtractor()
    features = extractor.extract_features_for_inference(audio)
    
    print(f"\nExtracted features shape: {features.shape}")
    print(f"Expected shape: {extractor.get_feature_shape()}")
    print(f"Feature range: [{features.min():.4f}, {features.max():.4f}]")
    
    if args.save_features:
        output_path = args.audio.replace('.wav', '_features.npy')
        np.save(output_path, features)
        print(f"\nFeatures saved to: {output_path}")
    
    print("\n" + "="*70)
    print("Feature extraction complete!")
    print("="*70)
