"""
Live Continuous Audio Monitoring for CryingSense

Continuously records audio from the microphone and runs real-time inference
to detect and classify baby cries. Shows live sound levels and predictions.

Features:
- Continuous microphone monitoring
- Real-time sound level visualization
- Automatic cry detection and classification
- Alert system when cry is detected with high confidence
"""

import os
import sys
import time
import wave
import numpy as np
import pyaudio
import torch
import torch.nn.functional as F
import librosa
from datetime import datetime
from collections import deque
import threading
import queue

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.models.cnn_model import CryingSenseCNN

# Import sound level utilities
from sound_level_meter import calculate_db, create_level_bar, get_level_status


class LiveMonitor:
    """Continuous live audio monitoring with real-time inference."""
    
    def __init__(self, model_path, num_classes=5, sample_rate=16000, 
                 chunk_duration=5.0, confidence_threshold=0.6, device_index=None):
        """
        Initialize live monitor.
        
        Args:
            model_path: Path to trained model checkpoint
            num_classes: Number of output classes
            sample_rate: Audio sample rate (default: 16000 Hz)
            chunk_duration: Duration of each audio chunk for inference (default: 5.0s)
            confidence_threshold: Minimum confidence for alerts
            device_index: Audio device index (None for default)
        """
        self.model_path = model_path
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.confidence_threshold = confidence_threshold
        self.device_index = device_index
        
        # Audio parameters
        self.chunk_size = 1024
        self.channels = 1
        self.audio_format = pyaudio.paInt16
        self.target_samples = int(sample_rate * chunk_duration)
        
        # PyTorch device
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Class names (sorted alphabetically as in training)
        # Always use 6 classes - noise is trained but treated as "invisible" during inference
        self.class_names = ['belly_pain', 'burp', 'discomfort', 'hunger', 'noise', 'tired']
        self.invisible_classes = ['noise']  # Classes that don't trigger alerts
        
        # Feature extraction parameters
        self.n_fft = 1024
        self.hop_length = 512
        self.n_mfcc = 40
        self.n_mels = 128
        self.n_chroma = 12
        self.target_time_steps = int(np.ceil((sample_rate * chunk_duration) / self.hop_length))
        
        # State tracking
        self.running = False
        self.audio_buffer = deque(maxlen=self.target_samples)
        self.peak_db = -100
        self.last_prediction = None
        self.prediction_history = deque(maxlen=10)
        self.alert_count = 0
        
        # Load model
        self.model = self._load_model()
        
        # PyAudio
        self.audio = pyaudio.PyAudio()
    
    def _load_model(self):
        """Load trained model."""
        print(f"Loading model from: {self.model_path}")
        print(f"Device: {self.torch_device}")
        
        model = CryingSenseCNN(num_classes=self.num_classes).to(self.torch_device)
        
        # Initialize _fc1 by running dummy forward pass
        dummy_input = torch.randn(1, 4, 128, 216).to(self.torch_device)
        with torch.no_grad():
            _ = model(dummy_input)
        
        checkpoint = torch.load(self.model_path, map_location=self.torch_device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"Validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2%}")
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        return model
    
    def _pad_or_crop(self, feature, target_shape):
        """Pad or crop feature to target shape."""
        padded = np.zeros(target_shape, dtype=feature.dtype)
        min_shape = (min(feature.shape[0], target_shape[0]), 
                     min(feature.shape[1], target_shape[1]))
        padded[:min_shape[0], :min_shape[1]] = feature[:min_shape[0], :min_shape[1]]
        return padded
    
    def _extract_features(self, audio_data):
        """Extract features from audio buffer."""
        # Convert int16 to float
        y = audio_data.astype(np.float32) / 32768.0
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=self.n_mfcc,
                                    n_fft=self.n_fft, hop_length=self.hop_length)
        mfcc = self._pad_or_crop(mfcc, (self.n_mfcc, self.target_time_steps))
        
        # Extract Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=self.sample_rate, n_mels=self.n_mels,
                                            n_fft=self.n_fft, hop_length=self.hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = self._pad_or_crop(mel_db, (self.n_mels, self.target_time_steps))
        
        # Extract Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sample_rate, n_chroma=self.n_chroma,
                                            n_fft=self.n_fft, hop_length=self.hop_length)
        chroma = self._pad_or_crop(chroma, (self.n_chroma, self.target_time_steps))
        
        # Combine features (4 channels, 128 height, time width)
        combined = np.zeros((4, 128, self.target_time_steps))
        combined[0, :self.n_mfcc, :] = mfcc
        combined[1, :self.n_mels, :] = mel_db
        combined[2, :self.n_chroma, :] = chroma
        # Channel 3: delta MFCC (same method as training)
        delta_mfcc = np.zeros_like(mfcc)
        delta_mfcc[:, 1:] = mfcc[:, 1:] - mfcc[:, :-1]
        combined[3, :self.n_mfcc, :] = delta_mfcc
        
        return combined
    
    def _run_inference(self, features):
        """Run model inference on features."""
        features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.torch_device)
        
        with torch.no_grad():
            output = self.model(features_tensor)
            probs = F.softmax(output, dim=1).cpu().numpy()[0]
        
        predicted_idx = np.argmax(probs)
        predicted_class = self.class_names[predicted_idx]
        confidence = probs[predicted_idx]
        
        # Check if this is an invisible class (noise)
        is_cry = predicted_class not in self.invisible_classes
        display_prediction = predicted_class if is_cry else 'no_cry_detected'
        
        return {
            'prediction': display_prediction,
            'raw_prediction': predicted_class,
            'is_cry': is_cry,
            'confidence': confidence,
            'probabilities': {name: prob for name, prob in zip(self.class_names, probs)},
            'timestamp': datetime.now()
        }
    
    def _draw_display(self, current_db, prediction=None):
        """Draw the live monitoring display."""
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 70)
        print("  CRYINGSENSE LIVE MONITOR")
        print("=" * 70)
        print()
        
        # Audio level meter
        print("  AUDIO INPUT")
        print("  " + "-" * 66)
        
        bar = create_level_bar(current_db, width=50)
        status, symbol = get_level_status(current_db)
        
        print(f"  Level: [{bar}] {current_db:5.1f} dB")
        print(f"  Peak:  {self.peak_db:5.1f} dB  |  Status: [{symbol}] {status}")
        print()
        
        # Buffer status
        buffer_pct = len(self.audio_buffer) / self.target_samples * 100
        buffer_bar = "█" * int(buffer_pct / 5) + "·" * (20 - int(buffer_pct / 5))
        print(f"  Buffer: [{buffer_bar}] {buffer_pct:5.1f}%  ({len(self.audio_buffer)}/{self.target_samples} samples)")
        print()
        
        # Prediction section
        print("  PREDICTION")
        print("  " + "-" * 66)
        
        if prediction:
            pred_class = prediction['prediction']
            raw_class = prediction.get('raw_prediction', pred_class)
            is_cry = prediction.get('is_cry', True)
            confidence = prediction['confidence']
            is_alert = is_cry and confidence >= self.confidence_threshold
            
            # Show prediction
            if not is_cry:
                print(f"  Current: {raw_class} -> NO CRY DETECTED ({confidence:.1%})")
                print(f"  Status: Environmental/background noise")
            elif is_alert:
                print(f"  >>> DETECTED: {pred_class.upper()} ({confidence:.1%}) <<<")
                print(f"  ALERT STATUS: ⚠️  HIGH CONFIDENCE DETECTION!")
            else:
                print(f"  Current: {pred_class} ({confidence:.1%})")
                print(f"  Status: Below threshold ({self.confidence_threshold:.0%})")            
            print()
            
            # Class probabilities
            print("  Class Probabilities:")
            sorted_probs = sorted(prediction['probabilities'].items(), 
                                 key=lambda x: x[1], reverse=True)
            for class_name, prob in sorted_probs:
                bar_len = int(prob * 30)
                prob_bar = "█" * bar_len + "·" * (30 - bar_len)
                marker = " <--" if class_name == pred_class else ""
                print(f"    {class_name:12s} [{prob_bar}] {prob:5.1%}{marker}")
        else:
            print("  Waiting for audio buffer to fill...")
            print()
            print("  Speak or make sounds into the microphone")
        
        print()
        
        # Statistics
        print("  " + "-" * 66)
        print(f"  Alerts: {self.alert_count}  |  "
              f"Threshold: {self.confidence_threshold:.0%}  |  "
              f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print()
        print("=" * 70)
        print("  Press Ctrl+C to stop")
        print("=" * 70)
    
    def run(self, save_detections=False, output_dir='detections'):
        """
        Start continuous live monitoring.
        
        Args:
            save_detections: Whether to save audio when cry is detected
            output_dir: Directory to save detection audio files
        """
        if save_detections:
            os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "=" * 70)
        print("Starting CryingSense Live Monitor")
        print("=" * 70)
        print(f"Sample Rate: {self.sample_rate} Hz")
        print(f"Chunk Duration: {self.chunk_duration}s")
        print(f"Confidence Threshold: {self.confidence_threshold:.0%}")
        print(f"Save Detections: {save_detections}")
        print("=" * 70)
        print("\nInitializing audio stream...")
        
        # Open audio stream
        try:
            stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size
            )
        except Exception as e:
            print(f"Error opening audio stream: {e}")
            print("\nTry running with --list-devices to see available devices")
            return
        
        self.running = True
        last_inference_time = 0
        inference_interval = 1.0  # Run inference every 1 second
        current_prediction = None
        
        print("Monitoring started. Press Ctrl+C to stop.\n")
        time.sleep(1)
        
        try:
            while self.running:
                # Read audio chunk
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                except Exception as e:
                    print(f"Error reading audio: {e}")
                    continue
                
                # Convert to numpy and add to buffer
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                self.audio_buffer.extend(audio_chunk)
                
                # Calculate current dB level
                current_db = calculate_db(audio_chunk)
                if current_db > self.peak_db:
                    self.peak_db = current_db
                
                # Run inference when buffer is full and interval has passed
                current_time = time.time()
                if (len(self.audio_buffer) >= self.target_samples and 
                    current_time - last_inference_time >= inference_interval):
                    
                    # Get audio from buffer
                    audio_data = np.array(list(self.audio_buffer))
                    
                    # Extract features and run inference
                    try:
                        features = self._extract_features(audio_data)
                        current_prediction = self._run_inference(features)
                        self.prediction_history.append(current_prediction)
                        
                        # Check for alert
                        if current_prediction['confidence'] >= self.confidence_threshold:
                            self.alert_count += 1
                            
                            # Save detection audio if enabled
                            if save_detections:
                                self._save_detection(audio_data, current_prediction, output_dir)
                        
                    except Exception as e:
                        print(f"Inference error: {e}")
                    
                    last_inference_time = current_time
                
                # Update display
                self._draw_display(current_db, current_prediction)
                
                # Small delay for display refresh
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\n\nStopping live monitor...")
        finally:
            self.running = False
            stream.stop_stream()
            stream.close()
            
            # Print summary
            print("\n" + "=" * 70)
            print("MONITORING SESSION SUMMARY")
            print("=" * 70)
            print(f"Total Alerts: {self.alert_count}")
            print(f"Peak Audio Level: {self.peak_db:.1f} dB")
            if self.prediction_history:
                # Count predictions by class
                class_counts = {}
                for pred in self.prediction_history:
                    cls = pred['prediction']
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                print(f"Prediction Distribution:")
                for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
                    print(f"  {cls}: {count}")
            print("=" * 70)
    
    def _save_detection(self, audio_data, prediction, output_dir):
        """Save detected audio clip."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pred_class = prediction['prediction']
        confidence = int(prediction['confidence'] * 100)
        filename = f"detection_{timestamp}_{pred_class}_{confidence}pct.wav"
        filepath = os.path.join(output_dir, filename)
        
        # Save as WAV
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())
        
        print(f"\n  [SAVED] {filename}")
    
    def list_devices(self):
        """List available audio devices."""
        print("\n" + "=" * 60)
        print("Available Audio Input Devices")
        print("=" * 60)
        info = self.audio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        
        default_device = self.audio.get_default_input_device_info()
        default_idx = default_device.get('index')
        
        for i in range(num_devices):
            device_info = self.audio.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels') > 0:
                marker = " [DEFAULT]" if i == default_idx else ""
                print(f"  Device {i}: {device_info.get('name')}{marker}")
        
        print("=" * 60)
    
    def close(self):
        """Clean up resources."""
        self.running = False
        self.audio.terminate()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='CryingSense Live Continuous Monitoring',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start live monitoring with default settings
  python live.py --model ../model/saved_models/cryingsense_cnn_best.pth
  
  # Monitor and save detected cries
  python live.py --model ../model/saved_models/cryingsense_cnn_best.pth --save
  
  # Set custom confidence threshold
  python live.py --model ../model/saved_models/cryingsense_cnn_best.pth --threshold 0.8
  
  # Use specific audio device
  python live.py --model ../model/saved_models/cryingsense_cnn_best.pth --device 1
        """
    )
    
    parser.add_argument('--model', type=str,
                       default='../model/saved_models/cryingsense_cnn_best.pth',
                       help='Path to trained model (default: ../model/saved_models/cryingsense_cnn_best.pth)')
    parser.add_argument('--num-classes', type=int, default=5,
                       help='Number of classes in model (default: 5)')
    parser.add_argument('--threshold', type=float, default=0.6,
                       help='Confidence threshold for alerts (default: 0.6)')
    parser.add_argument('--duration', type=float, default=5.0,
                       help='Audio chunk duration for inference (default: 5.0s)')
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='Audio sample rate (default: 16000)')
    parser.add_argument('--device', type=int, default=None,
                       help='Audio device index (default: system default)')
    parser.add_argument('--save', action='store_true',
                       help='Save audio when cry is detected')
    parser.add_argument('--output', type=str, default='detections',
                       help='Output directory for saved detections (default: detections)')
    parser.add_argument('--list-devices', action='store_true',
                       help='List available audio devices and exit')
    
    args = parser.parse_args()
    
    # Check model exists
    if not args.list_devices and not os.path.exists(args.model):
        print(f"\nError: Model not found at {args.model}")
        print("Please train the model first or provide correct model path.")
        sys.exit(1)
    
    # Initialize monitor
    monitor = LiveMonitor(
        model_path=args.model,
        num_classes=args.num_classes,
        sample_rate=args.sample_rate,
        chunk_duration=args.duration,
        confidence_threshold=args.threshold,
        device_index=args.device
    )
    
    try:
        if args.list_devices:
            monitor.list_devices()
        else:
            monitor.run(save_detections=args.save, output_dir=args.output)
    finally:
        monitor.close()


if __name__ == "__main__":
    main()
