"""
Live Audio Testing Module for CryingSense

Tests the trained CNN model on live audio recordings from the microphone.
Performs feature extraction and inference in real-time.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import librosa
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.models.cnn_model import CryingSenseCNN


class CryingSensePredictor:
    """Handles inference on audio files using trained CNN model."""
    
    def __init__(self, model_path, num_classes=5, confidence_threshold=0.6):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            num_classes: Number of classes in the model
            confidence_threshold: Minimum confidence for predictions
        """
        self.model_path = model_path
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model()
        
        # Class names (should match training order)
        self.class_names = [
            'belly_pain', 'burp', 'discomfort', 'hunger', 'noise', 'tired'
        ][:num_classes]
    
    def _load_model(self):
        """Load trained model from checkpoint."""
        print(f"Loading model from: {self.model_path}")
        print(f"Device: {self.device}")
        
        model = CryingSenseCNN(num_classes=self.num_classes).to(self.device)
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"Validation accuracy: {checkpoint.get('val_acc', 'N/A')}")
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        print(f"Estimated size: ~{total_params * 4 / 1024 / 1024:.2f} MB\n")
        
        return model
    
    def extract_features(self, audio_path, sample_rate=16000, duration=5.0):
        """
        Extract features from audio file.
        
        Args:
            audio_path: Path to audio file
            sample_rate: Sample rate for loading
            duration: Duration to process
        
        Returns:
            torch.Tensor: Combined features (MFCC, Mel, Chroma)
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=sample_rate, duration=duration)
        
        # Feature extraction parameters
        n_fft = 1024
        hop_length = 512
        n_mfcc = 40
        n_mels = 128
        n_chroma = 12
        
        # Calculate expected time steps
        target_time_steps = int(np.ceil((sample_rate * duration) / hop_length))
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, 
                                    n_fft=n_fft, hop_length=hop_length)
        mfcc = self._pad_or_crop(mfcc, (n_mfcc, target_time_steps))
        
        # Extract Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                            n_fft=n_fft, hop_length=hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = self._pad_or_crop(mel_db, (n_mels, target_time_steps))
        
        # Extract Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma,
                                            n_fft=n_fft, hop_length=hop_length)
        chroma = self._pad_or_crop(chroma, (n_chroma, target_time_steps))
        
        # Combine features (stack along channel dimension)
        # For simplicity, we'll use a subset to match the model's expected 4 channels
        # Take first 40 MFCC, first 128 Mel, first 12 Chroma = total 180 features
        # We need to reshape/select to get 4 channels
        # Let's use: MFCC (channel 0-1), Mel (channel 2), Chroma (channel 3)
        
        # Simplified: take averages to create 4 channels
        feature_ch0 = mfcc[:20, :].mean(axis=0, keepdims=True)  # Shape: (1, time)
        feature_ch1 = mfcc[20:40, :].mean(axis=0, keepdims=True)
        feature_ch2 = mel_db[:64, :].mean(axis=0, keepdims=True)
        feature_ch3 = mel_db[64:128, :].mean(axis=0, keepdims=True)
        
        # Stack to create (4, 1, time) then tile to (4, 128, time)
        combined = np.vstack([feature_ch0, feature_ch1, feature_ch2, feature_ch3])
        combined = np.tile(combined, (32, 1))  # Tile to create height dimension
        
        # Alternative: Use actual features properly
        # For demo, let's properly construct 4-channel input (4, 128, time)
        combined = np.zeros((4, 128, target_time_steps))
        combined[0, :40, :] = mfcc
        combined[1, :128, :] = mel_db
        combined[2, :12, :] = chroma
        # Channel 3 could be delta MFCC or other features
        
        # Convert to torch tensor
        features = torch.from_numpy(combined).float()
        features = features.unsqueeze(0)  # Add batch dimension
        
        return features
    
    def _pad_or_crop(self, feature, target_shape):
        """Pad or crop feature to target shape."""
        padded = np.zeros(target_shape, dtype=feature.dtype)
        min_shape = (min(feature.shape[0], target_shape[0]), 
                     min(feature.shape[1], target_shape[1]))
        padded[:min_shape[0], :min_shape[1]] = feature[:min_shape[0], :min_shape[1]]
        return padded
    
    def predict(self, audio_path):
        """
        Run inference on audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            dict: Prediction results
        """
        import time
        
        # Extract features
        print(f"Processing: {os.path.basename(audio_path)}")
        start_time = time.time()
        features = self.extract_features(audio_path)
        feature_time = (time.time() - start_time) * 1000
        
        # Run inference
        features = features.to(self.device)
        
        with torch.no_grad():
            start_time = time.time()
            output = self.model(features)
            inference_time = (time.time() - start_time) * 1000
        
        # Get probabilities
        probs = F.softmax(output, dim=1)
        confidence, prediction = probs.max(1)
        
        confidence_value = confidence.item()
        predicted_class = self.class_names[prediction.item()]
        
        # Check confidence threshold
        is_confident = confidence_value >= self.confidence_threshold
        
        # Create results dictionary
        results = {
            'audio_file': os.path.basename(audio_path),
            'prediction': predicted_class if is_confident else 'UNCERTAIN',
            'confidence': confidence_value,
            'is_confident': is_confident,
            'threshold': self.confidence_threshold,
            'probabilities': {
                self.class_names[i]: probs[0][i].item()
                for i in range(len(self.class_names))
            },
            'timing': {
                'feature_extraction_ms': feature_time,
                'inference_ms': inference_time,
                'total_ms': feature_time + inference_time
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def predict_batch(self, audio_files):
        """
        Run inference on multiple audio files.
        
        Args:
            audio_files: List of audio file paths
        
        Returns:
            list: List of prediction results
        """
        results = []
        for audio_file in audio_files:
            result = self.predict(audio_file)
            results.append(result)
        return results


def print_results(results):
    """Print prediction results in a formatted way."""
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"Audio File: {results['audio_file']}")
    print(f"Timestamp: {results['timestamp']}")
    print("-"*70)
    
    if results['is_confident']:
        print(f"\nPrediction: {results['prediction'].upper()}")
        print(f"Confidence: {results['confidence']:.2%}")
        print(f"Status: ✓ CONFIDENT")
    else:
        print(f"\nPrediction: {results['prediction']}")
        print(f"Confidence: {results['confidence']:.2%}")
        print(f"Status: ✗ BELOW THRESHOLD ({results['threshold']:.2%})")
        print(f"\nNote: Prediction confidence is too low. This may be:")
        print("  - Background noise")
        print("  - Unclear audio")
        print("  - Audio not matching trained cry patterns")
    
    print(f"\nClass Probabilities:")
    sorted_probs = sorted(results['probabilities'].items(), 
                         key=lambda x: x[1], reverse=True)
    for class_name, prob in sorted_probs:
        bar_length = int(prob * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"  {class_name:15s} [{bar}] {prob:.2%}")
    
    print(f"\nTiming:")
    print(f"  Feature Extraction: {results['timing']['feature_extraction_ms']:.2f} ms")
    print(f"  Model Inference:    {results['timing']['inference_ms']:.2f} ms")
    print(f"  Total Time:         {results['timing']['total_ms']:.2f} ms")
    print("="*70)


def main():
    """Main testing function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test CryingSense CNN model on live audio recordings'
    )
    parser.add_argument('--model', type=str, 
                       default='../model/saved_models/cryingsense_cnn_best.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--audio', type=str, required=True,
                       help='Path to audio file or directory')
    parser.add_argument('--threshold', type=float, default=0.6,
                       help='Confidence threshold (default: 0.6)')
    parser.add_argument('--num-classes', type=int, default=5,
                       help='Number of classes in model (default: 5)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')
    
    args = parser.parse_args()
    
    print("="*70)
    print("CryingSense Live Audio Testing")
    print("="*70)
    
    # Check model exists
    if not os.path.exists(args.model):
        print(f"\nError: Model not found at {args.model}")
        print("Please train the model first or provide correct model path.")
        sys.exit(1)
    
    # Initialize predictor
    predictor = CryingSensePredictor(
        model_path=args.model,
        num_classes=args.num_classes,
        confidence_threshold=args.threshold
    )
    
    # Get audio files
    if os.path.isfile(args.audio):
        audio_files = [args.audio]
    elif os.path.isdir(args.audio):
        audio_files = [
            os.path.join(args.audio, f)
            for f in os.listdir(args.audio)
            if f.endswith(('.wav', '.mp3', '.flac'))
        ]
        if not audio_files:
            print(f"\nError: No audio files found in {args.audio}")
            sys.exit(1)
    else:
        print(f"\nError: Audio path not found: {args.audio}")
        sys.exit(1)
    
    print(f"\nProcessing {len(audio_files)} audio file(s)...")
    print("="*70)
    
    # Run predictions
    all_results = []
    for audio_file in audio_files:
        results = predictor.predict(audio_file)
        print_results(results)
        all_results.append(results)
        print()
    
    # Save results if output specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Summary
    if len(all_results) > 1:
        confident = sum(1 for r in all_results if r['is_confident'])
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Total files:       {len(all_results)}")
        print(f"Confident preds:   {confident} ({confident/len(all_results)*100:.1f}%)")
        print(f"Uncertain preds:   {len(all_results)-confident} ({(len(all_results)-confident)/len(all_results)*100:.1f}%)")
        print("="*70)


if __name__ == "__main__":
    main()
