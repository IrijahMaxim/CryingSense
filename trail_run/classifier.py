"""
Real-Time Cry Classifier for CryingSense

Classifies audio using the trained CNN model.
Handles feature extraction and inference.
"""

import sys
import os
import threading
import time
import logging
from typing import Optional, Dict, List, Tuple, Callable
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import librosa

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from .audio_buffer import AudioBuffer, RecordingBuffer
    from . import config
except ImportError:
    from audio_buffer import AudioBuffer, RecordingBuffer
    import config

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract acoustic features for CNN input."""
    
    def __init__(self):
        """Initialize with config parameters."""
        self.sample_rate = config.SAMPLE_RATE
        self.n_mfcc = config.N_MFCC
        self.n_mels = config.N_MELS
        self.n_chroma = config.N_CHROMA
        self.n_fft = config.N_FFT
        self.hop_length = config.HOP_LENGTH
        self.duration = config.DURATION
        
        # Target dimensions
        self.target_time_steps = int(np.ceil((self.sample_rate * self.duration) / self.hop_length))
        self.target_height = max(self.n_mfcc, self.n_mels, self.n_chroma)
    
    def extract(self, audio: np.ndarray) -> torch.Tensor:
        """
        Extract features from audio and prepare for model input.
        
        Args:
            audio: Audio samples (int16 or float32)
        
        Returns:
            Feature tensor ready for model (1, 4, height, width)
        """
        # Convert to float32 and normalize
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        
        # Pad or crop to target duration
        target_samples = int(self.duration * self.sample_rate)
        if len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
        elif len(audio) > target_samples:
            audio = audio[:target_samples]
        
        # Extract features
        mfcc = self._extract_mfcc(audio)
        mel = self._extract_mel_spectrogram(audio)
        chroma = self._extract_chroma(audio)
        delta_mfcc = self._compute_delta(mfcc)
        
        # Pad to unified height
        mfcc = self._pad_feature(mfcc, self.target_height)
        mel = self._pad_feature(mel, self.target_height)
        chroma = self._pad_feature(chroma, self.target_height)
        delta_mfcc = self._pad_feature(delta_mfcc, self.target_height)
        
        # Stack into 4-channel array
        features = np.stack([mfcc, mel, chroma, delta_mfcc], axis=0)
        
        # Convert to tensor and add batch dimension
        tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        return tensor
    
    def _extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features."""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return self._pad_time(mfcc)
    
    def _extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract Mel spectrogram features."""
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return self._pad_time(mel_db)
    
    def _extract_chroma(self, audio: np.ndarray) -> np.ndarray:
        """Extract Chroma features."""
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_chroma=self.n_chroma
        )
        return self._pad_time(chroma)
    
    def _compute_delta(self, feature: np.ndarray) -> np.ndarray:
        """Compute delta (first derivative) of feature."""
        return librosa.feature.delta(feature)
    
    def _pad_time(self, feature: np.ndarray) -> np.ndarray:
        """Pad or crop to target time steps."""
        if feature.shape[1] < self.target_time_steps:
            pad_width = self.target_time_steps - feature.shape[1]
            feature = np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')
        elif feature.shape[1] > self.target_time_steps:
            feature = feature[:, :self.target_time_steps]
        return feature
    
    def _pad_feature(self, feature: np.ndarray, target_height: int) -> np.ndarray:
        """Pad feature to target height."""
        if feature.shape[0] < target_height:
            pad_height = target_height - feature.shape[0]
            feature = np.pad(feature, ((0, pad_height), (0, 0)), mode='constant')
        return feature


class CryClassifier:
    """
    Real-time cry classifier using trained CNN model.
    
    Continuously monitors audio buffer and classifies when cry is detected.
    """
    
    def __init__(self, audio_buffer: AudioBuffer, model_path: str = None):
        """
        Initialize classifier.
        
        Args:
            audio_buffer: Audio buffer to monitor
            model_path: Path to trained model (default from config)
        """
        self.audio_buffer = audio_buffer
        self.model_path = model_path or str(config.MODEL_PATH)
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        
        # Class configuration
        self.class_names = config.CLASS_NAMES
        self.ignore_classes = config.IGNORE_CLASSES
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD
        
        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._current_prediction: Optional[Dict] = None
        self._prediction_lock = threading.Lock()
        
        # Recording
        self.recording_buffer = RecordingBuffer(
            sample_rate=config.SAMPLE_RATE,
            max_duration=30.0
        )
        
        # Detection state
        self._cry_detected = False
        self._cry_start_time = 0
        self._consecutive_cry_count = 0
        self._silence_start_time = 0
        
        # Callbacks
        self._on_cry_start: Optional[Callable] = None
        self._on_cry_end: Optional[Callable] = None
        self._on_classification: Optional[Callable] = None
        
        # Statistics
        self._classifications_made = 0
        self._cries_detected = 0
        
        logger.info(f"Classifier initialized on {self.device}")
        logger.info(f"Ignoring classes: {self.ignore_classes}")
    
    def _load_model(self) -> torch.nn.Module:
        """Load the trained CNN model."""
        try:
            from model.models.cnn_model import CryingSenseCNN
            
            model = CryingSenseCNN(
                num_classes=config.NUM_CLASSES,
                in_channels=4,
                dropout_rate=0.3,
                use_gap=True
            )
            
            # Load weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            
            model = model.to(self.device)
            model.eval()
            
            logger.info(f"Model loaded from {self.model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def start(self) -> None:
        """Start the classifier monitoring thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._classify_loop, daemon=True)
        self._thread.start()
        logger.info("Classifier started")
    
    def stop(self) -> None:
        """Stop the classifier."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("Classifier stopped")
    
    def _classify_loop(self) -> None:
        """Main classification loop."""
        logger.info("Classification loop started")
        self._last_amplitude_log = 0
        
        while self._running:
            try:
                # Wait for enough audio
                buffer_duration = self.audio_buffer.duration
                if buffer_duration < config.DETECTION_WINDOW:
                    if time.time() - self._last_amplitude_log > 10.0:
                        logger.debug(f"Waiting for audio... Buffer: {buffer_duration:.2f}s / {config.DETECTION_WINDOW:.2f}s")
                        self._last_amplitude_log = time.time()
                    time.sleep(0.1)
                    continue
                
                # Get detection window audio
                window_samples = int(config.DETECTION_WINDOW * config.SAMPLE_RATE)
                audio = self.audio_buffer.read_latest(window_samples)
                
                # Check amplitude (simple cry detection)
                amplitude = np.abs(audio).mean()
                is_loud = amplitude > config.AMPLITUDE_THRESHOLD
                
                # Lightweight telemetry logging
                current_time = time.time()
                if not hasattr(self, '_last_amplitude_log'):
                    self._last_amplitude_log = 0
                log_interval = 10.0 if not config.DEBUG_MODE else 5.0
                if current_time - self._last_amplitude_log > log_interval:
                    logger.info(f"Audio level: {amplitude:.1f} (threshold: {config.AMPLITUDE_THRESHOLD}, cry detected: {is_loud})")
                    self._last_amplitude_log = current_time
                
                if is_loud:
                    self._consecutive_cry_count += 1
                    self._silence_start_time = 0
                    
                    # Confirm cry after consecutive detections
                    if (self._consecutive_cry_count >= config.CRY_DETECTION_CONSECUTIVE 
                        and not self._cry_detected):
                        self._start_cry_detection()
                else:
                    self._consecutive_cry_count = 0
                    
                    # Check for cry end (silence)
                    if self._cry_detected:
                        if self._silence_start_time == 0:
                            self._silence_start_time = time.time()
                        elif time.time() - self._silence_start_time > config.POST_CRY_SILENCE:
                            self._end_cry_detection()
                
                # Add to recording if active
                if self.recording_buffer.is_recording:
                    self.recording_buffer.append(audio)
                
                # Periodic classification during cry (throttled to every 2 seconds)
                if self._cry_detected:
                    if not hasattr(self, '_last_classification_time'):
                        self._last_classification_time = 0
                    current_time = time.time()
                    if current_time - self._last_classification_time >= 2.0:
                        self._classify_current_audio()
                        self._last_classification_time = current_time
                
                time.sleep(0.1)  # 100ms loop
                
            except Exception as e:
                logger.error(f"Classification loop error: {e}")
                time.sleep(0.5)
    
    def _start_cry_detection(self) -> None:
        """Start cry detection and recording."""
        logger.info("Cry detected - starting recording")
        self._cry_detected = True
        self._cry_start_time = time.time()
        self._cries_detected += 1
        
        # Get pre-buffer audio
        pre_buffer_samples = int(config.PRE_CRY_BUFFER * config.SAMPLE_RATE)
        pre_audio = self.audio_buffer.read(
            config.PRE_CRY_BUFFER, 
            offset=config.DETECTION_WINDOW
        )
        
        self.recording_buffer.start(pre_audio)
        
        if self._on_cry_start:
            self._on_cry_start()
    
    def _end_cry_detection(self) -> None:
        """End cry detection and perform final classification."""
        logger.info("Cry ended - performing final classification")
        
        # Get the full recording
        recording = self.recording_buffer.stop()
        
        if len(recording) > 0:
            # Classify the full recording
            result = self._classify_audio(recording)
            
            if result and not self._should_ignore(result['class']):
                result['audio'] = recording
                result['duration'] = len(recording) / config.SAMPLE_RATE
                
                if self._on_cry_end:
                    self._on_cry_end(result)
        
        self._cry_detected = False
        self._silence_start_time = 0
    
    def _classify_current_audio(self) -> None:
        """Classify current audio in buffer."""
        # Get full duration audio
        full_samples = int(config.DURATION * config.SAMPLE_RATE)
        audio = self.audio_buffer.read_latest(full_samples)
        
        result = self._classify_audio(audio)
        
        if result:
            logger.info(f"Classification: {result['class']} ({result['confidence']:.2%}) - Ignored: {self._should_ignore(result['class'])}")
            with self._prediction_lock:
                self._current_prediction = result
    
    def _classify_audio(self, audio: np.ndarray) -> Optional[Dict]:
        """
        Classify audio samples.
        
        Args:
            audio: Audio samples
        
        Returns:
            Classification result dict or None
        """
        try:
            if config.DEBUG_MODE and logger.isEnabledFor(logging.DEBUG):
                audio_stats = {
                    'dtype': audio.dtype,
                    'min': float(np.min(audio)),
                    'max': float(np.max(audio)),
                    'mean': float(np.mean(np.abs(audio))),
                    'std': float(np.std(audio)),
                    'samples': len(audio)
                }
                logger.debug(
                    "Audio stats - dtype: %s, range: [%.1f, %.1f], mean_abs: %.1f, std: %.1f",
                    audio_stats['dtype'],
                    audio_stats['min'],
                    audio_stats['max'],
                    audio_stats['mean'],
                    audio_stats['std'],
                )
            
            # Extract features
            features = self.feature_extractor.extract(audio)
            features = features.to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(features)
                probabilities = F.softmax(outputs, dim=1)
            
            probs = probabilities.cpu().numpy()[0]
            predicted_idx = int(np.argmax(probs))
            predicted_class = self.class_names[predicted_idx]
            confidence = float(probs[predicted_idx])
            
            # Build all probabilities dict
            all_probs = {
                name: float(probs[i]) 
                for i, name in enumerate(self.class_names)
            }
            
            if config.DEBUG_MODE and logger.isEnabledFor(logging.DEBUG):
                probs_str = ", ".join([
                    f"{name}: {prob:.1%}" for name, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                ])
                logger.debug(f"All probabilities: {probs_str}")
            
            result = {
                'class': predicted_class,
                'confidence': confidence,
                'probabilities': all_probs,
                'timestamp': time.time(),
            }
            
            self._classifications_made += 1
            
            # Always call callback to update display (database filtering handled elsewhere)
            if self._on_classification:
                is_ignored = self._should_ignore(predicted_class)
                if config.DEBUG_MODE and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Sending classification to callback: %s (%.2f%%) [Ignored: %s]",
                        predicted_class,
                        confidence * 100.0,
                        is_ignored,
                    )
                self._on_classification(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return None
    
    def _should_ignore(self, class_name: str) -> bool:
        """Check if class should be ignored."""
        return class_name.lower() in [c.lower() for c in self.ignore_classes]
    
    def classify_now(self) -> Optional[Dict]:
        """
        Perform immediate classification on current buffer.
        
        Returns:
            Classification result
        """
        full_samples = int(config.DURATION * config.SAMPLE_RATE)
        audio = self.audio_buffer.read_latest(full_samples)
        return self._classify_audio(audio)
    
    def on_cry_start(self, callback: Callable) -> None:
        """Set callback for cry start."""
        self._on_cry_start = callback
    
    def on_cry_end(self, callback: Callable) -> None:
        """Set callback for cry end with result."""
        self._on_cry_end = callback
    
    def on_classification(self, callback: Callable) -> None:
        """Set callback for each classification."""
        self._on_classification = callback

    def force_offline_reset(self) -> None:
        """Reset live detection state when audio source is offline."""
        self._consecutive_cry_count = 0
        self._silence_start_time = 0
        self._cry_detected = False

        # Drop unfinished recording when source disappears.
        if self.recording_buffer.is_recording:
            self.recording_buffer.stop()

        with self._prediction_lock:
            self._current_prediction = None
    
    @property
    def current_prediction(self) -> Optional[Dict]:
        """Get current prediction."""
        with self._prediction_lock:
            return self._current_prediction
    
    @property
    def is_cry_detected(self) -> bool:
        """Whether cry is currently being detected."""
        return self._cry_detected
    
    @property
    def is_running(self) -> bool:
        """Whether classifier is running."""
        return self._running
    
    def get_stats(self) -> Dict:
        """Get classifier statistics."""
        return {
            "running": self._running,
            "cry_detected": self._cry_detected,
            "cries_detected": self._cries_detected,
            "classifications_made": self._classifications_made,
            "recording": self.recording_buffer.is_recording,
            "recording_duration": self.recording_buffer.duration,
        }
