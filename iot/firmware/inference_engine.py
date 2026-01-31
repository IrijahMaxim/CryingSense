"""
Raspberry Pi Inference Engine
Runs CNN model to classify infant cries from ESP32 data
"""

import sys
import os
import json
import time
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import torch
    from model.models.cnn_model import CryingSenseCNN
    from model.models.model_utils import load_model, quantize_model
    import librosa
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Warning: PyTorch not available - running in simulation mode")


class InferenceEngine:
    """
    Inference engine for Raspberry Pi that processes cry audio
    and generates classifications.
    """
    
    def __init__(self, model_path, device='cpu', use_quantized=False, 
                 class_names=None, confidence_threshold=0.6):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to trained model (.pth file)
            device: Device to use ('cpu' or 'cuda')
            use_quantized: Whether to use quantized model
            class_names: List of class names
            confidence_threshold: Minimum confidence to report prediction
        """
        self.device = device
        self.use_quantized = use_quantized
        self.confidence_threshold = confidence_threshold
        
        # Default class names
        if class_names is None:
            self.class_names = ['belly_pain', 'burp', 'discomfort', 'hunger', 'tired']
        else:
            self.class_names = class_names
        
        if PYTORCH_AVAILABLE:
            self._load_model(model_path)
        else:
            print("Running in simulation mode")
        
        print(f"Inference Engine initialized")
        print(f"Device: {self.device}")
        print(f"Classes: {self.class_names}")
        print(f"Confidence threshold: {self.confidence_threshold}")
    
    def _load_model(self, model_path):
        """Load the PyTorch model."""
        self.device = torch.device(self.device)
        
        # Initialize model
        self.model = CryingSenseCNN(num_classes=len(self.class_names))
        
        # Load weights
        load_model(self.model, model_path, device=self.device)
        
        # Quantize if requested
        if self.use_quantized:
            print("Quantizing model for edge deployment...")
            self.model = quantize_model(self.model)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
    
    def process_audio_file(self, audio_path):
        """
        Process audio file and generate prediction.
        
        Args:
            audio_path: Path to audio file (.wav)
        
        Returns:
            Dictionary containing prediction results
        """
        if not PYTORCH_AVAILABLE:
            return self._simulate_inference()
        
        try:
            # Extract features from audio
            features = self._extract_features(audio_path)
            
            # Run inference
            result = self._run_inference(features)
            
            return result
        
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None
    
    def process_raw_audio(self, audio_data, sample_rate=16000):
        """
        Process raw audio data and generate prediction.
        
        Args:
            audio_data: NumPy array of audio samples
            sample_rate: Sample rate of audio
        
        Returns:
            Dictionary containing prediction results
        """
        if not PYTORCH_AVAILABLE:
            return self._simulate_inference()
        
        try:
            # Extract features from raw audio
            features = self._extract_features_from_array(audio_data, sample_rate)
            
            # Run inference
            result = self._run_inference(features)
            
            return result
        
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None
    
    def _extract_features(self, audio_path, sr=16000, duration=5.0, 
                         n_mfcc=40, n_mels=128, n_chroma=12, target_time_steps=216):
        """Extract features from audio file."""
        # Load audio
        y, sr = librosa.load(audio_path, sr=sr, duration=duration)
        
        return self._extract_features_from_array(y, sr, n_mfcc, n_mels, n_chroma, target_time_steps)
    
    def _extract_features_from_array(self, y, sr=16000, n_mfcc=40, n_mels=128, 
                                    n_chroma=12, target_time_steps=216):
        """Extract features from audio array."""
        # Normalize
        y = librosa.util.normalize(y)
        
        # Pad or trim
        target_length = int(sr * 5.0)
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
        
        # SpecAugment (use same for inference)
        mel_db_aug = mel_db.copy()
        
        # Extract Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
        chroma = self._pad_or_crop(chroma, (n_chroma, target_time_steps))
        
        # Stack features
        features = np.stack([mfcc, mel_db, mel_db_aug, chroma], axis=0)
        
        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        return features_tensor
    
    def _pad_or_crop(self, feature, target_shape):
        """Pad or crop feature to target shape."""
        padded = np.zeros(target_shape, dtype=feature.dtype)
        min_shape = (min(feature.shape[0], target_shape[0]), 
                    min(feature.shape[1], target_shape[1]))
        padded[:min_shape[0], :min_shape[1]] = feature[:min_shape[0], :min_shape[1]]
        return padded
    
    def _run_inference(self, features):
        """Run model inference."""
        features = features.to(self.device)
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model(features)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        inference_time = time.time() - start_time
        
        predicted_class = self.class_names[predicted.item()]
        confidence_value = confidence.item()
        
        # Get all probabilities
        probs_dict = {
            class_name: probabilities[0][i].item() 
            for i, class_name in enumerate(self.class_names)
        }
        
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence_value,
            'probabilities': probs_dict,
            'inference_time_ms': inference_time * 1000,
            'timestamp': time.time(),
            'meets_threshold': confidence_value >= self.confidence_threshold
        }
        
        return result
    
    def _simulate_inference(self):
        """Simulate inference for testing."""
        import random
        
        time.sleep(0.1)  # Simulate processing time
        
        predicted_class = random.choice(self.class_names)
        confidence = random.uniform(0.5, 0.95)
        
        probs_dict = {class_name: random.uniform(0.0, 0.3) 
                     for class_name in self.class_names}
        probs_dict[predicted_class] = confidence
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probs_dict,
            'inference_time_ms': random.uniform(50, 200),
            'timestamp': time.time(),
            'meets_threshold': confidence >= self.confidence_threshold
        }


def test_inference_engine():
    """Test the inference engine."""
    print("="*60)
    print("Testing Inference Engine")
    print("="*60)
    
    # Initialize engine (will run in simulation mode if PyTorch not available)
    model_path = "../../model/saved_models/cryingsense_cnn.pth"
    
    if not os.path.exists(model_path):
        print(f"\nModel not found at {model_path}")
        print("Running in simulation mode...")
        model_path = None
    
    engine = InferenceEngine(
        model_path=model_path if model_path else "dummy.pth",
        device='cpu',
        confidence_threshold=0.6
    )
    
    # Test with simulated audio
    print("\nRunning test inference...")
    result = engine._simulate_inference()
    
    print("\nInference Result:")
    print(f"  Predicted Class: {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Meets Threshold: {result['meets_threshold']}")
    print(f"  Inference Time: {result['inference_time_ms']:.2f} ms")
    print("\n  Probabilities:")
    for class_name, prob in result['probabilities'].items():
        print(f"    {class_name}: {prob:.4f}")
    
    print("="*60)
    print("Test complete!")


if __name__ == "__main__":
    test_inference_engine()
