"""
CryingSense Inference Script

Main inference script for real-time baby cry classification:
- Single file inference
- Batch inference
- Real-time streaming inference (placeholder)
- JSON output with predictions, confidence, and timing
- Database integration for storing classifications, sessions, and audio files
"""

import os
import sys
import json
import time
import argparse
import shutil
import uuid
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import librosa

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from model.inference.model_loader import ModelLoader
from model.inference.audio_preprocessor import AudioPreprocessor
from model.inference.feature_extractor import FeatureExtractor

# Database imports (optional - only used if database saving is enabled)
try:
    from database.services import CryClassificationService, SessionService
    from database.models import AudioFile, AudioMetadata
    from database.repository import AudioFileRepository
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    print("Warning: Database modules not available. Running without database integration.")


class CryingSensePredictor:
    """Main predictor class for CryingSense inference."""
    
    # Classes that are trained but not returned as cry predictions
    # Noise is an "invisible" class - model learns it but doesn't predict it as a cry type
    INVISIBLE_CLASSES = ['noise']
    
    def __init__(self, model_path, num_classes=6, device=None, 
                 confidence_threshold=0.70, save_to_db=False, 
                 device_id=None, device_source="esp32",
                 audio_storage_dir=None, session_id=None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model
            num_classes: Number of classes (default: 6)
            device: Device to run inference on
            confidence_threshold: Minimum confidence for alerts (default: 0.70)
            save_to_db: Whether to save results to database (default: False)
            device_id: Device identifier for database tracking (default: None)
            device_source: Device type - 'esp32' or 'android' (default: 'esp32')
            audio_storage_dir: Directory to store audio files (default: None = project_root/audio_storage)
            session_id: Optional session ID for grouping predictions (default: None)
        """
        self.model_path = model_path
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.save_to_db = save_to_db
        self.device_id = device_id
        self.device_source = device_source
        self.session_id = session_id
        
        # Initialize components
        print("Initializing CryingSense Predictor...")
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model_loader = ModelLoader(model_path, num_classes, self.device)
        self.model = self.model_loader.load()
        
        # Initialize preprocessor and feature extractor
        self.preprocessor = AudioPreprocessor()
        self.feature_extractor = FeatureExtractor()
        
        # Class names - must match the sorted order from training (alphabetical)
        # Includes noise as trained class, but it's "invisible" for predictions
        self.class_names = ['belly_pain', 'burp', 'discomfort', 'hunger', 'noise', 'tired']
        
        # Cry-only class names (excludes invisible classes)
        self.cry_class_names = [c for c in self.class_names if c not in self.INVISIBLE_CLASSES]
        
        # Initialize database services if enabled
        self.classification_service = None
        self.session_service = None
        self.audio_file_repo = None
        
        if self.save_to_db:
            if not DATABASE_AVAILABLE:
                print("Warning: Database requested but not available. Continuing without database.")
                self.save_to_db = False
            else:
                self.classification_service = CryClassificationService()
                self.session_service = SessionService()
                self.audio_file_repo = AudioFileRepository()
                print("Database integration enabled")
        
        # Setup audio storage directory
        if audio_storage_dir:
            self.audio_storage_dir = Path(audio_storage_dir)
        else:
            # Default to project_root/audio_storage
            project_root = Path(__file__).parent.parent.parent
            self.audio_storage_dir = project_root / "audio_storage"
        
        if self.save_to_db and self.audio_storage_dir:
            self.audio_storage_dir.mkdir(parents=True, exist_ok=True)
            print(f"Audio storage directory: {self.audio_storage_dir}")
        
        print(f"Predictor initialized on device: {self.device}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"Cry classes: {self.cry_class_names}")
        if self.save_to_db:
            print(f"Device ID: {self.device_id}")
            print(f"Device Source: {self.device_source}")
            if self.session_id:
                print(f"Session ID: {self.session_id}")
    
    def start_new_session(self):
        """
        Start a new monitoring session and update instance session_id.
        
        Returns:
            str: New session ID if successful, None otherwise
        """
        if not self.save_to_db or not self.session_service:
            print("Database integration not enabled")
            return None
        
        if not self.device_id:
            print("Device ID required to start a session")
            return None
        
        try:
            session_id = self.session_service.start_session(
                device_id=self.device_id,
                device_source=self.device_source
            )
            
            if session_id:
                self.session_id = session_id
                print(f"✓ New session started: {session_id}")
                return session_id
            else:
                print("Failed to start session")
                return None
                
        except Exception as e:
            print(f"Error starting session: {e}")
            return None
    
    def end_current_session(self):
        """
        End the current monitoring session.
        
        Returns:
            bool: True if successful
        """
        if not self.save_to_db or not self.session_service:
            print("Database integration not enabled")
            return False
        
        if not self.session_id:
            print("No active session")
            return False
        
        try:
            success = self.session_service.end_session(self.session_id)
            if success:
                print(f"✓ Session ended: {self.session_id}")
                self.session_id = None
            return success
            
        except Exception as e:
            print(f"Error ending session: {e}")
            return False
    
    def _save_audio_file(self, audio_path, classification_id=None):
        """
        Save audio file to storage directory and create database record.
        
        Args:
            audio_path: Path to the original audio file
            classification_id: Optional classification ID to link
            
        Returns:
            str: File ID if successful, None otherwise
        """
        if not self.save_to_db or not self.audio_file_repo:
            return None
        
        try:
            # Generate unique file ID
            file_id = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            # Get audio metadata
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            file_size = os.path.getsize(audio_path)
            
            # Copy audio file to storage directory
            storage_path = self.audio_storage_dir / f"{file_id}.wav"
            shutil.copy2(audio_path, storage_path)
            
            # Create AudioFile record
            audio_file = AudioFile(
                file_id=file_id,
                original_filename=os.path.basename(audio_path),
                file_path=str(storage_path),
                audio_metadata=AudioMetadata(
                    sample_rate=int(sr),
                    duration_seconds=float(duration),
                    channels=1 if len(y.shape) == 1 else y.shape[0],
                    bit_depth=16,
                    file_size_bytes=file_size
                ),
                mime_type="audio/wav",
                device_id=self.device_id,
                session_id=self.session_id,
                classification_id=classification_id
            )
            
            # Save to database
            result = self.audio_file_repo.create(audio_file)
            if result:
                print(f"  Audio file saved: {file_id}")
                return file_id
            
            return None
            
        except Exception as e:
            print(f"  Warning: Failed to save audio file: {e}")
            return None
    
    def _save_to_database(self, audio_path, result, mfcc_features=None):
        """
        Save classification result and audio file to database.
        
        Args:
            audio_path: Path to the audio file
            result: Prediction result dictionary
            mfcc_features: Optional MFCC features array
            
        Returns:
            str: Classification ID if successful, None otherwise
        """
        if not self.save_to_db or not self.classification_service:
            return None
        
        try:
            # Only save actual cry detections (not noise)
            if not result['is_cry']:
                print(f"  Skipping database save (no cry detected)")
                return None
            
            # Get audio metadata
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Convert MFCC features to list format if provided
            mfcc_list = None
            if mfcc_features is not None:
                if isinstance(mfcc_features, np.ndarray):
                    mfcc_list = mfcc_features.tolist()
            
            # Save classification to database
            classification_id = self.classification_service.save_classification(
                predicted_class=result['prediction'],
                confidence_score=result['confidence'],
                device_source=self.device_source,
                duration_seconds=duration,
                sample_rate=int(sr),
                device_id=self.device_id,
                session_id=self.session_id,
                all_probabilities=result.get('probabilities', {}),
                mfcc_features=mfcc_list,
                model_version="1.0.0"
            )
            
            if classification_id:
                print(f"  Classification saved to DB: {classification_id}")
                
                # Save audio file if classification was saved successfully
                audio_file_id = self._save_audio_file(audio_path, classification_id)
                result['classification_id'] = classification_id
                result['audio_file_id'] = audio_file_id
                
                return classification_id
            
            return None
            
        except Exception as e:
            print(f"  Warning: Failed to save to database: {e}")
            return None
    
    def predict_single(self, audio_path, return_all_probs=True, save_to_db=None):
        """
        Perform inference on a single audio file.
        
        Args:
            audio_path: Path to audio file
            return_all_probs: Whether to return all class probabilities
            save_to_db: Override instance setting for saving to database (optional)
        
        Returns:
            Dictionary containing prediction results
        """
        start_time = time.time()
        
        # Determine if we should save to database
        should_save_db = save_to_db if save_to_db is not None else self.save_to_db
        
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
        
        # Check if prediction is an invisible class (e.g., noise)
        is_cry = predicted_class not in self.INVISIBLE_CLASSES
        
        # Build result
        result = {
            'is_cry': is_cry,
            'prediction': predicted_class if is_cry else 'no_cry_detected',
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
            # Also include cry-only probabilities (excluding noise)
            result['cry_probabilities'] = {
                name: round(float(prob), 4)
                for name, prob in zip(self.class_names, probabilities)
                if name not in self.INVISIBLE_CLASSES
            }
        
        # Add alert flag - only alert for actual cry predictions above threshold
        result['alert'] = is_cry and confidence >= self.confidence_threshold
        
        # Save to database if enabled
        if should_save_db:
            self._save_to_database(audio_path, result, mfcc_features=features)
        
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
  
  # Save results to database with device tracking
  python predict.py --audio test.wav --model ../saved_models/cryingsense_cnn.pth \\
    --save-to-db --device-id ESP32-001 --device-source esp32
  
  # Run inference with session tracking
  python predict.py --audio test.wav --model ../saved_models/cryingsense_cnn.pth \\
    --save-to-db --device-id ESP32-001 --session-id session-123
        """
    )
    
    # Input arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--audio', type=str, help='Path to single audio file')
    group.add_argument('--batch', type=str, help='Path to directory with audio files')
    group.add_argument('--stream', action='store_true', help='Real-time streaming mode')
    
    # Model arguments
    parser.add_argument('--model', type=str, 
                       default='../saved_models/cryingsense_cnn_best.pth',
                       help='Path to trained model (default: ../saved_models/cryingsense_cnn_best.pth)')
    parser.add_argument('--num-classes', type=int, default=6,
                       help='Number of classes (default: 6)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cpu/cuda, default: auto-detect)')
    
    # Output arguments
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for batch results (JSON format)')
    parser.add_argument('--confidence-threshold', type=float, default=0.70,
                       help='Confidence threshold for alerts (default: 0.70)')
    parser.add_argument('--recursive', action='store_true',
                       help='Search for audio files recursively in batch mode')
    
    # Database arguments
    parser.add_argument('--save-to-db', action='store_true',
                       help='Save results to database')
    parser.add_argument('--device-id', type=str, default=None,
                       help='Device identifier for database tracking')
    parser.add_argument('--device-source', type=str, default='esp32',
                       choices=['esp32', 'android'],
                       help='Device source type (default: esp32)')
    parser.add_argument('--session-id', type=str, default=None,
                       help='Session ID for grouping predictions')
    parser.add_argument('--audio-storage-dir', type=str, default=None,
                       help='Directory to store audio files (default: project_root/audio_storage)')
    
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
    if args.save_to_db:
        print(f"Database: Enabled")
        print(f"Device ID: {args.device_id}")
        print(f"Device Source: {args.device_source}")
        if args.session_id:
            print(f"Session ID: {args.session_id}")
    print("="*70)
    print()
    
    # Initialize predictor
    predictor = CryingSensePredictor(
        model_path=args.model,
        num_classes=args.num_classes,
        device=device,
        confidence_threshold=args.confidence_threshold,
        save_to_db=args.save_to_db,
        device_id=args.device_id,
        device_source=args.device_source,
        audio_storage_dir=args.audio_storage_dir,
        session_id=args.session_id
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
