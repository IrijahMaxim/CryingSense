import torch
import numpy as np
import librosa
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_model import CryingSenseCNN
from models.model_utils import load_model


class CryPredictor:
    """
    Real-time cry classification predictor.
    """
    def __init__(self, model_path, device='cpu', class_names=None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model file
            device: Device to run inference on ('cpu' or 'cuda')
            class_names: List of class names in order (default: sorted alphabetically)
        """
        self.device = torch.device(device)
        
        # Default class names based on dataset
        if class_names is None:
            self.class_names = ['belly_pain', 'burp', 'discomfort', 'hunger', 'tired']
        else:
            self.class_names = class_names
        
        # Load model
        self.model = CryingSenseCNN(num_classes=len(self.class_names)).to(self.device)
        load_model(self.model, model_path, device=self.device)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Device: {self.device}")
        print(f"Classes: {self.class_names}")
    
    def extract_features(self, audio_path=None, audio_data=None, sr=16000, 
                        n_mfcc=40, n_mels=128, n_chroma=12, duration=5.0, 
                        target_time_steps=216):
        """
        Extract features from audio file or audio data.
        
        Args:
            audio_path: Path to audio file (WAV)
            audio_data: Raw audio data as numpy array
            sr: Sample rate
            n_mfcc: Number of MFCC coefficients
            n_mels: Number of Mel bands
            n_chroma: Number of chroma bins
            duration: Duration to process (seconds)
            target_time_steps: Target number of time steps
        
        Returns:
            Feature tensor of shape (1, 4, features, time)
        """
        # Load audio
        if audio_path is not None:
            y, sr = librosa.load(audio_path, sr=sr, duration=duration)
        elif audio_data is not None:
            y = audio_data
        else:
            raise ValueError("Either audio_path or audio_data must be provided")
        
        # Normalize audio
        y = librosa.util.normalize(y)
        
        # Pad or trim to fixed duration
        target_length = int(sr * duration)
        if len(y) > target_length:
            y = y[:target_length]
        else:
            y = np.pad(y, (0, target_length - len(y)))
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc = self._pad_or_crop(mfcc, (n_mfcc, target_time_steps))
        
        # Extract Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = self._pad_or_crop(mel_db, (n_mels, target_time_steps))
        
        # SpecAugment version (for robustness, use same mel_db for now)
        mel_db_aug = mel_db.copy()
        
        # Extract Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
        chroma = self._pad_or_crop(chroma, (n_chroma, target_time_steps))
        
        # Stack features: (4, features, time)
        features = np.stack([mfcc, mel_db, mel_db_aug, chroma], axis=0)
        
        # Convert to tensor and add batch dimension
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        return features_tensor
    
    def _pad_or_crop(self, feature, target_shape):
        """Pad or crop feature to target shape."""
        padded = np.zeros(target_shape, dtype=feature.dtype)
        min_shape = (min(feature.shape[0], target_shape[0]), 
                    min(feature.shape[1], target_shape[1]))
        padded[:min_shape[0], :min_shape[1]] = feature[:min_shape[0], :min_shape[1]]
        return padded
    
    def predict(self, audio_path=None, audio_data=None, return_probabilities=False):
        """
        Predict the cry class from audio.
        
        Args:
            audio_path: Path to audio file
            audio_data: Raw audio data as numpy array
            return_probabilities: If True, return probabilities for all classes
        
        Returns:
            If return_probabilities is False: (predicted_class, confidence)
            If return_probabilities is True: (predicted_class, confidence, probabilities_dict)
        """
        # Extract features
        features = self.extract_features(audio_path=audio_path, audio_data=audio_data)
        features = features.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(features)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = self.class_names[predicted.item()]
        confidence_value = confidence.item()
        
        if return_probabilities:
            probs_dict = {
                class_name: probabilities[0][i].item() 
                for i, class_name in enumerate(self.class_names)
            }
            return predicted_class, confidence_value, probs_dict
        else:
            return predicted_class, confidence_value
    
    def predict_batch(self, audio_paths):
        """
        Predict cry classes for a batch of audio files.
        
        Args:
            audio_paths: List of paths to audio files
        
        Returns:
            List of tuples (predicted_class, confidence) for each audio file
        """
        results = []
        for audio_path in audio_paths:
            predicted_class, confidence = self.predict(audio_path=audio_path)
            results.append((predicted_class, confidence))
        return results


def main():
    """Example usage of CryPredictor."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict infant cry classification')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to trained model (.pth file)')
    parser.add_argument('--audio', type=str, required=True,
                       help='Path to audio file (.wav)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--probabilities', action='store_true',
                       help='Show probabilities for all classes')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = CryPredictor(model_path=args.model, device=args.device)
    
    # Make prediction
    if args.probabilities:
        predicted_class, confidence, probs = predictor.predict(
            audio_path=args.audio, 
            return_probabilities=True
        )
        print(f"\nPredicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        print("\nProbabilities for all classes:")
        for class_name, prob in probs.items():
            print(f"  {class_name}: {prob:.4f}")
    else:
        predicted_class, confidence = predictor.predict(audio_path=args.audio)
        print(f"\nPredicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()

# SECURITY NOTE: When loading models, always use weights_only=True to prevent
# remote code execution vulnerabilities (CVE-2024-XXXXX)
# Example: torch.load(filepath, map_location=device, weights_only=True)
