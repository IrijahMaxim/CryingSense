"""
ESP32 Audio Capture Module
Captures audio from microphone and prepares it for processing
"""

import time
import array


try:
    # For real ESP32 hardware
    from machine import Pin, I2S
    HARDWARE_AVAILABLE = True
except ImportError:
    # For development/testing on regular Python
    HARDWARE_AVAILABLE = False
    print("Warning: Running in simulation mode (no ESP32 hardware detected)")


class AudioCapture:
    """
    Handles audio capture from ESP32 microphone.
    """
    
    def __init__(self, sample_rate=16000, bits_per_sample=16, 
                 buffer_size=4096, duration=5.0):
        """
        Initialize audio capture.
        
        Args:
            sample_rate: Audio sample rate in Hz
            bits_per_sample: Bits per audio sample
            buffer_size: Size of audio buffer
            duration: Duration to capture in seconds
        """
        self.sample_rate = sample_rate
        self.bits_per_sample = bits_per_sample
        self.buffer_size = buffer_size
        self.duration = duration
        self.total_samples = int(sample_rate * duration)
        
        if HARDWARE_AVAILABLE:
            self._init_hardware()
        else:
            print("Audio capture initialized in simulation mode")
    
    def _init_hardware(self):
        """Initialize I2S hardware for audio capture."""
        # I2S configuration for ESP32
        # Pin configuration may vary based on your specific ESP32 board
        SCK_PIN = 14  # Serial Clock
        WS_PIN = 15   # Word Select (LRCK)
        SD_PIN = 32   # Serial Data
        
        self.audio_in = I2S(
            0,
            sck=Pin(SCK_PIN),
            ws=Pin(WS_PIN),
            sd=Pin(SD_PIN),
            mode=I2S.RX,
            bits=self.bits_per_sample,
            format=I2S.MONO,
            rate=self.sample_rate,
            ibuf=self.buffer_size
        )
        
        print("I2S audio capture initialized")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Duration: {self.duration} seconds")
    
    def capture_audio(self):
        """
        Capture audio from microphone.
        
        Returns:
            bytearray containing captured audio data
        """
        if HARDWARE_AVAILABLE:
            return self._capture_hardware()
        else:
            return self._capture_simulated()
    
    def _capture_hardware(self):
        """Capture audio from actual hardware."""
        print("Capturing audio...")
        
        # Calculate number of bytes to capture
        bytes_per_sample = self.bits_per_sample // 8
        total_bytes = self.total_samples * bytes_per_sample
        
        # Allocate buffer
        audio_buffer = bytearray(total_bytes)
        
        # Read audio data
        bytes_read = 0
        while bytes_read < total_bytes:
            chunk = min(self.buffer_size, total_bytes - bytes_read)
            num_read = self.audio_in.readinto(audio_buffer[bytes_read:bytes_read + chunk])
            if num_read:
                bytes_read += num_read
        
        print(f"Captured {bytes_read} bytes ({bytes_read / (self.sample_rate * bytes_per_sample):.2f} seconds)")
        
        return audio_buffer
    
    def _capture_simulated(self):
        """Simulate audio capture for testing."""
        import random
        
        print("Simulating audio capture...")
        time.sleep(0.5)  # Simulate capture time
        
        bytes_per_sample = self.bits_per_sample // 8
        total_bytes = self.total_samples * bytes_per_sample
        
        # Generate random audio data (white noise) for simulation
        audio_buffer = bytearray(total_bytes)
        for i in range(0, total_bytes, bytes_per_sample):
            # Simulate 16-bit audio sample
            value = int(random.uniform(-32768, 32767))
            audio_buffer[i] = value & 0xFF
            audio_buffer[i + 1] = (value >> 8) & 0xFF
        
        print(f"Simulated {total_bytes} bytes")
        return audio_buffer
    
    def bytes_to_array(self, audio_bytes):
        """
        Convert byte array to integer array.
        
        Args:
            audio_bytes: bytearray of audio data
        
        Returns:
            array.array of 16-bit integers
        """
        # Convert bytes to 16-bit signed integers
        samples = array.array('h')  # 'h' = signed short (16-bit)
        
        for i in range(0, len(audio_bytes), 2):
            # Combine two bytes into one 16-bit sample
            sample = audio_bytes[i] | (audio_bytes[i + 1] << 8)
            # Convert to signed integer
            if sample >= 32768:
                sample -= 65536
            samples.append(sample)
        
        return samples
    
    def normalize_audio(self, samples):
        """
        Normalize audio samples to -1.0 to 1.0 range.
        
        Args:
            samples: array.array of audio samples
        
        Returns:
            list of normalized float values
        """
        max_val = max(abs(min(samples)), abs(max(samples)))
        if max_val == 0:
            return [0.0] * len(samples)
        
        return [float(s) / max_val for s in samples]
    
    def deinit(self):
        """Cleanup audio resources."""
        if HARDWARE_AVAILABLE and hasattr(self, 'audio_in'):
            self.audio_in.deinit()
            print("Audio capture deinitialized")


def test_audio_capture():
    """Test function for audio capture."""
    print("="*50)
    print("Testing Audio Capture Module")
    print("="*50)
    
    # Initialize capture
    capture = AudioCapture(sample_rate=16000, duration=5.0)
    
    # Capture audio
    audio_bytes = capture.capture_audio()
    print(f"Captured audio: {len(audio_bytes)} bytes")
    
    # Convert to array
    samples = capture.bytes_to_array(audio_bytes)
    print(f"Converted to samples: {len(samples)} samples")
    
    # Normalize
    normalized = capture.normalize_audio(samples)
    print(f"Normalized: {len(normalized)} values")
    print(f"Range: [{min(normalized):.3f}, {max(normalized):.3f}]")
    
    # Cleanup
    capture.deinit()
    
    print("="*50)
    print("Test complete!")


if __name__ == "__main__":
    test_audio_capture()
