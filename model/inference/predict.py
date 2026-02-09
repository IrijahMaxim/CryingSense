"""
CryingSense Inference Script

Main inference script for real-time baby cry classification:
- Single file inference
- Batch inference
- Real-time streaming inference (placeholder)
- JSON output with predictions, confidence, and timing
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from model.inference.model_loader import ModelLoader
from model.inference.audio_preprocessor import AudioPreprocessor
from model.inference.feature_extractor import FeatureExtractor


class CryingSensePredictor:
    """Main predictor class for CryingSense inference."""
    
    def __init__(self, model_path, num_classes=5, device=None, 
                 confidence_threshold=0.70):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model
            num_classes: Number of classes (default: 5)
            device: Device to run inference on
            confidence_threshold: Minimum confidence for alerts (default: 0.70)
        """
        self.model_path = model_path
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        print("Initializing CryingSense Predictor...")
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model_loader = ModelLoader(model_path, num_classes, self.device)
        self.model = self.model_loader.load()
        
        # Initialize preprocessor and feature extractor
        self.preprocessor = AudioPreprocessor()
        self.feature_extractor = FeatureExtractor()
        
        # Class names (excluding 'noise' which is filtered out)
        self.class_names = ['belly_pain', 'burp', 'discomfort', 'hunger', 'tired']
        
        print(f"Predictor initialized on device: {self.device}")
        print(f"Confidence threshold: {self.confidence_threshold}")
    
    def predict_single(self, audio_path, return_all_probs=True):
        """
        Perform inference on a single audio file.
        
        Args:
            audio_path: Path to audio file
            return_all_probs: Whether to return all class probabilities
        
        Returns:
            Dictionary containing prediction results
        """
        start_time = time.time()
        
        # Preprocess audio
        audio = self.preprocessor.preprocess(audio_path)
        
        # Extract features
        features = self.feature_extractor.extract_features_for_inference(audio)
        
        # Convert to tensor and add batch dimension
        features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        
        # Inference
        inference_start = time.time()
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
        inference_time = (time.time() - inference_start) * 1000  # Convert to ms
        
        # Get prediction
        predicted_idx = np.argmax(probabilities)
        predicted_class = self.class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        # Build result
        result = {
            'prediction': predicted_class,
            'confidence': round(confidence, 4),
            'inference_time_ms': round(inference_time, 2),
            'total_time_ms': round((time.time() - start_time) * 1000, 2),
            'timestamp': datetime.now().isoformat(),
            'audio_file': os.path.basename(audio_path)
        }
        
        if return_all_probs:
            result['probabilities'] = {
                name: round(float(prob), 4) 
                for name, prob in zip(self.class_names, probabilities)
            }
        
        # Add alert flag if confidence exceeds threshold
        result['alert'] = confidence >= self.confidence_threshold
        
        return result
    
    def predict_batch(self, audio_dir, output_file=None, recursive=False):
        """
        Perform batch inference on multiple audio files.
        
        Args:
            audio_dir: Directory containing audio files
            output_file: Optional path to save results as JSON
            recursive: Whether to search recursively
        
        Returns:
            List of prediction results
        """
        print(f"Scanning directory: {audio_dir}")
        
        # Find all audio files
        audio_dir = Path(audio_dir)
        pattern = '**/*.wav' if recursive else '*.wav'
        audio_files = list(audio_dir.glob(pattern))
        
        if not audio_files:
            print("No audio files found!")
            return []
        
        print(f"Found {len(audio_files)} audio files")
        print("="*70)
        
        results = []
        for i, audio_path in enumerate(audio_files, 1):
            print(f"[{i}/{len(audio_files)}] Processing: {audio_path.name}")
            
            try:
                result = self.predict_single(str(audio_path))
                results.append(result)
                
                print(f"  Prediction: {result['prediction']} "
                      f"(confidence: {result['confidence']:.2%})")
                print(f"  Inference time: {result['inference_time_ms']:.2f}ms")
                
                if result['alert']:
                    print(f"  ⚠️  ALERT: High confidence detection!")
                
            except Exception as e:
                error_result = {
                    'audio_file': audio_path.name,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                results.append(error_result)
                print(f"  Error: {e}")
            
            print()
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {output_file}")
        
        # Print summary
        print("="*70)
        print("Batch Inference Summary")
        print("="*70)
        print(f"Total files: {len(audio_files)}")
        print(f"Successful: {len([r for r in results if 'prediction' in r])}")
        print(f"Errors: {len([r for r in results if 'error' in r])}")
        print(f"Alerts: {len([r for r in results if r.get('alert', False)])}")
        
        if results:
            avg_inference_time = np.mean([r['inference_time_ms'] for r in results if 'inference_time_ms' in r])
            print(f"Average inference time: {avg_inference_time:.2f}ms")
        
        print("="*70)
        
        return results
    
    def predict_streaming(self):
        """
        Placeholder for real-time streaming inference.
        
        This would integrate with microphone input for real-time monitoring.
        """
        print("Streaming inference not yet implemented.")
        print("This feature would:")
        print("  - Capture audio from microphone in real-time")
        print("  - Process audio in sliding windows")
        print("  - Provide continuous predictions")
        print("  - Trigger alerts when high-confidence cries detected")
        
        raise NotImplementedError("Streaming inference is not yet implemented")


def main():
    """Main function for command-line inference."""
    parser = argparse.ArgumentParser(
        description='CryingSense Inference - Baby Cry Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file inference
  python predict.py --audio test.wav --model ../saved_models/cryingsense_cnn.pth
  
  # Batch inference
  python predict.py --batch audio_dir/ --model ../saved_models/cryingsense_cnn.pth
  
  # Save batch results to JSON
  python predict.py --batch audio_dir/ --model ../saved_models/cryingsense_cnn.pth --output results.json
  
  # Use quantized model for edge deployment
  python predict.py --audio test.wav --model ../saved_models/cryingsense_cnn_quantized.pth
        """
    )
    
    # Input arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--audio', type=str, help='Path to single audio file')
    group.add_argument('--batch', type=str, help='Path to directory with audio files')
    group.add_argument('--stream', action='store_true', help='Real-time streaming mode')
    
    # Model arguments
    parser.add_argument('--model', type=str, 
                       default='../saved_models/cryingsense_cnn.pth',
                       help='Path to trained model (default: ../saved_models/cryingsense_cnn.pth)')
    parser.add_argument('--num-classes', type=int, default=5,
                       help='Number of classes (default: 5)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cpu/cuda, default: auto-detect)')
    
    # Output arguments
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for batch results (JSON format)')
    parser.add_argument('--confidence-threshold', type=float, default=0.70,
                       help='Confidence threshold for alerts (default: 0.70)')
    parser.add_argument('--recursive', action='store_true',
                       help='Search for audio files recursively in batch mode')
    
    args = parser.parse_args()
    
    # Set device if specified
    device = None
    if args.device:
        device = torch.device(args.device)
    
    print("="*70)
    print("CryingSense - Baby Cry Classification System")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Device: {args.device if args.device else 'auto-detect'}")
    print("="*70)
    print()
    
    # Initialize predictor
    predictor = CryingSensePredictor(
        model_path=args.model,
        num_classes=args.num_classes,
        device=device,
        confidence_threshold=args.confidence_threshold
    )
    
    print()
    print("="*70)
    
    # Run appropriate inference mode
    if args.audio:
        # Single file inference
        print("Running single file inference...")
        print("="*70)
        
        result = predictor.predict_single(args.audio)
        
        # Print results
        print(json.dumps(result, indent=2))
        
        print()
        print("="*70)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Inference time: {result['inference_time_ms']:.2f}ms")
        if result['alert']:
            print("⚠️  ALERT: High confidence detection!")
        print("="*70)
        
    elif args.batch:
        # Batch inference
        print("Running batch inference...")
        print("="*70)
        
        results = predictor.predict_batch(
            args.batch,
            output_file=args.output,
            recursive=args.recursive
        )
        
    elif args.stream:
        # Streaming inference
        print("Running streaming inference...")
        print("="*70)
        
        predictor.predict_streaming()


if __name__ == "__main__":
    main()
