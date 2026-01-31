"""
ESP32 Main Controller
Coordinates audio capture, feature extraction, and communication
"""

import time
import gc


# Import modules (some may not be available in simulation)
try:
    from audio_capture import AudioCapture
    from feature_extraction import LightweightFeatureExtractor
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    print("Warning: Running in simulation mode")


class CryingSenseESP32:
    """
    Main controller for ESP32-based cry detection transmitter.
    """
    
    def __init__(self, device_id="ESP32_001", sample_rate=16000, 
                 capture_duration=5.0, detection_threshold=0.1):
        """
        Initialize ESP32 controller.
        
        Args:
            device_id: Unique device identifier
            sample_rate: Audio sample rate
            capture_duration: Duration to capture audio (seconds)
            detection_threshold: Minimum RMS energy to consider as cry
        """
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.capture_duration = capture_duration
        self.detection_threshold = detection_threshold
        
        # Initialize components
        if MODULES_AVAILABLE:
            self.audio_capture = AudioCapture(
                sample_rate=sample_rate,
                duration=capture_duration
            )
            self.feature_extractor = LightweightFeatureExtractor(
                sample_rate=sample_rate
            )
        
        print(f"CryingSense ESP32 initialized: {device_id}")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Capture duration: {capture_duration} seconds")
    
    def detect_cry_event(self):
        """
        Monitor and detect cry events.
        
        Returns:
            Feature packet if cry detected, None otherwise
        """
        if not MODULES_AVAILABLE:
            return self._simulate_detection()
        
        # Step 1: Capture audio
        print("\nListening for cry...")
        audio_bytes = self.audio_capture.capture_audio()
        
        # Step 2: Convert to samples
        samples = self.audio_capture.bytes_to_array(audio_bytes)
        normalized = self.audio_capture.normalize_audio(samples)
        
        # Step 3: Quick energy check to filter out silence
        rms_energy = sum(s * s for s in normalized) / len(normalized)
        rms_energy = rms_energy ** 0.5
        
        print(f"RMS Energy: {rms_energy:.4f}")
        
        if rms_energy < self.detection_threshold:
            print("Energy too low - no cry detected")
            return None
        
        # Step 4: Extract features
        print("Cry detected! Extracting features...")
        features = self.feature_extractor.extract_basic_features(normalized)
        
        # Step 5: Create packet for transmission
        packet = self.feature_extractor.create_feature_packet(
            features, 
            device_id=self.device_id
        )
        
        return packet
    
    def _simulate_detection(self):
        """Simulate cry detection for testing."""
        import random
        
        print("\nSimulating cry detection...")
        time.sleep(1)
        
        # Randomly decide if cry is detected
        if random.random() > 0.3:  # 70% chance of detection
            print("Cry detected (simulated)!")
            return {
                'device_id': self.device_id,
                'timestamp': time.time(),
                'features': {
                    'rms_energy': round(random.uniform(0.15, 0.8), 4),
                    'zcr': round(random.uniform(0.05, 0.2), 4),
                    'mean_amplitude': round(random.uniform(0.1, 0.6), 4),
                    'spectral_centroid': round(random.uniform(2000, 5000), 4),
                    'duration': self.capture_duration,
                    'num_samples': int(self.sample_rate * self.capture_duration)
                },
                'envelope': [round(random.uniform(0.1, 0.9), 3) for _ in range(20)]
            }
        else:
            print("No cry detected (simulated)")
            return None
    
    def send_to_gateway(self, packet, gateway_url="http://raspberrypi.local:5000/api/cry-event"):
        """
        Send feature packet to Raspberry Pi gateway.
        
        Args:
            packet: Feature packet to send
            gateway_url: URL of the Raspberry Pi gateway
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Import HTTP client
            try:
                from http_client import HTTPClient
                client = HTTPClient()
                response = client.post(gateway_url, packet)
                print(f"Sent to gateway: {response}")
                return True
            except ImportError:
                print(f"[SIMULATION] Would send packet to: {gateway_url}")
                print(f"[SIMULATION] Packet: {packet}")
                return True
        except Exception as e:
            print(f"Error sending to gateway: {e}")
            return False
    
    def run_continuous(self, interval=2.0):
        """
        Run continuous cry detection loop.
        
        Args:
            interval: Seconds to wait between detections
        """
        print("="*50)
        print("Starting continuous cry detection...")
        print(f"Device ID: {self.device_id}")
        print(f"Detection interval: {interval} seconds")
        print("="*50)
        
        detection_count = 0
        
        try:
            while True:
                # Detect cry event
                packet = self.detect_cry_event()
                
                if packet:
                    detection_count += 1
                    print(f"\n[Detection #{detection_count}] Cry event detected!")
                    
                    # Send to gateway
                    success = self.send_to_gateway(packet)
                    
                    if success:
                        print("Successfully transmitted to gateway")
                    else:
                        print("Failed to transmit to gateway")
                
                # Garbage collection for memory management on ESP32
                gc.collect()
                
                # Wait before next detection
                print(f"\nWaiting {interval} seconds...")
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\n\nStopping cry detection...")
            print(f"Total detections: {detection_count}")
        
        finally:
            if MODULES_AVAILABLE and hasattr(self, 'audio_capture'):
                self.audio_capture.deinit()


def main():
    """Main entry point for ESP32 firmware."""
    # Configuration
    DEVICE_ID = "ESP32_001"
    SAMPLE_RATE = 16000
    CAPTURE_DURATION = 5.0
    DETECTION_THRESHOLD = 0.1
    DETECTION_INTERVAL = 2.0
    
    # Initialize controller
    controller = CryingSenseESP32(
        device_id=DEVICE_ID,
        sample_rate=SAMPLE_RATE,
        capture_duration=CAPTURE_DURATION,
        detection_threshold=DETECTION_THRESHOLD
    )
    
    # Run continuous detection
    controller.run_continuous(interval=DETECTION_INTERVAL)


if __name__ == "__main__":
    main()
