"""
ESP32 Feature Extraction Module (Lightweight)
Extracts basic audio features for transmission to Raspberry Pi
"""

import math


try:
    import array
    MICROPYTHON = True
except:
    MICROPYTHON = False


class LightweightFeatureExtractor:
    """
    Lightweight feature extraction for ESP32.
    Extracts basic features that can be transmitted to Raspberry Pi.
    """
    
    def __init__(self, sample_rate=16000, frame_size=512, hop_length=256):
        """
        Initialize feature extractor.
        
        Args:
            sample_rate: Audio sample rate
            frame_size: Frame size for analysis
            hop_length: Hop length between frames
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_length = hop_length
    
    def extract_basic_features(self, audio_samples):
        """
        Extract basic audio features suitable for ESP32 processing.
        
        Args:
            audio_samples: List or array of audio samples (normalized -1 to 1)
        
        Returns:
            Dictionary containing extracted features
        """
        features = {}
        
        # Time-domain features
        features['rms_energy'] = self._calculate_rms_energy(audio_samples)
        features['zero_crossing_rate'] = self._calculate_zcr(audio_samples)
        features['mean_amplitude'] = sum(abs(s) for s in audio_samples) / len(audio_samples)
        
        # Spectral centroid (simplified)
        features['spectral_centroid'] = self._calculate_spectral_centroid(audio_samples)
        
        # Temporal envelope
        features['envelope'] = self._calculate_envelope(audio_samples)
        
        # Duration and sample count
        features['duration'] = len(audio_samples) / self.sample_rate
        features['num_samples'] = len(audio_samples)
        
        return features
    
    def _calculate_rms_energy(self, samples):
        """Calculate Root Mean Square energy."""
        sum_squares = sum(s * s for s in samples)
        return math.sqrt(sum_squares / len(samples))
    
    def _calculate_zcr(self, samples):
        """Calculate Zero Crossing Rate."""
        zero_crossings = 0
        for i in range(1, len(samples)):
            if (samples[i] >= 0 and samples[i-1] < 0) or \
               (samples[i] < 0 and samples[i-1] >= 0):
                zero_crossings += 1
        return zero_crossings / len(samples)
    
    def _calculate_spectral_centroid(self, samples):
        """
        Calculate simplified spectral centroid.
        This is a lightweight approximation suitable for ESP32.
        """
        # Use simplified FFT-like approach with frame analysis
        num_frames = (len(samples) - self.frame_size) // self.hop_length + 1
        
        if num_frames <= 0:
            return 0.0
        
        centroids = []
        for i in range(num_frames):
            start = i * self.hop_length
            end = start + self.frame_size
            frame = samples[start:end]
            
            # Simplified spectral centroid for this frame
            weighted_sum = 0
            total_magnitude = 0
            
            for freq_idx, magnitude in enumerate(frame):
                abs_mag = abs(magnitude)
                weighted_sum += freq_idx * abs_mag
                total_magnitude += abs_mag
            
            if total_magnitude > 0:
                centroids.append(weighted_sum / total_magnitude)
        
        return sum(centroids) / len(centroids) if centroids else 0.0
    
    def _calculate_envelope(self, samples, window_size=256):
        """Calculate temporal envelope."""
        envelope = []
        
        for i in range(0, len(samples), window_size):
            window = samples[i:i + window_size]
            if window:
                envelope_value = max(abs(s) for s in window)
                envelope.append(envelope_value)
        
        return envelope
    
    def create_feature_packet(self, features, device_id="ESP32_001"):
        """
        Create a compact feature packet for transmission.
        
        Args:
            features: Dictionary of extracted features
            device_id: Device identifier
        
        Returns:
            Dictionary formatted for transmission
        """
        import time
        
        packet = {
            'device_id': device_id,
            'timestamp': time.time(),
            'features': {
                'rms_energy': round(features['rms_energy'], 4),
                'zcr': round(features['zero_crossing_rate'], 4),
                'mean_amplitude': round(features['mean_amplitude'], 4),
                'spectral_centroid': round(features['spectral_centroid'], 4),
                'duration': round(features['duration'], 2),
                'num_samples': features['num_samples']
            },
            # Include simplified envelope (first 20 values)
            'envelope': [round(v, 3) for v in features['envelope'][:20]]
        }
        
        return packet


def test_feature_extraction():
    """Test function for feature extraction."""
    import random
    import time
    
    print("="*50)
    print("Testing Feature Extraction Module")
    print("="*50)
    
    # Generate test audio (5 seconds of simulated cry sound)
    sample_rate = 16000
    duration = 5.0
    num_samples = int(sample_rate * duration)
    
    # Simulate audio with varying amplitude (like a cry)
    samples = []
    for i in range(num_samples):
        # Create a wave pattern with some randomness
        t = i / sample_rate
        amplitude = 0.5 * math.sin(2 * math.pi * 440 * t)  # 440 Hz tone
        amplitude += 0.2 * random.uniform(-1, 1)  # Add noise
        samples.append(amplitude)
    
    print(f"Generated {len(samples)} test samples ({duration} seconds)")
    
    # Extract features
    extractor = LightweightFeatureExtractor(sample_rate=sample_rate)
    
    start_time = time.time()
    features = extractor.extract_basic_features(samples)
    extraction_time = time.time() - start_time
    
    print(f"\nFeature Extraction Time: {extraction_time:.4f} seconds")
    print("\nExtracted Features:")
    for key, value in features.items():
        if key != 'envelope':
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {len(value)} values")
    
    # Create packet
    packet = extractor.create_feature_packet(features)
    print(f"\nPacket created for device: {packet['device_id']}")
    print(f"Packet size estimate: ~{len(str(packet))} bytes")
    
    print("="*50)
    print("Test complete!")


if __name__ == "__main__":
    test_feature_extraction()
