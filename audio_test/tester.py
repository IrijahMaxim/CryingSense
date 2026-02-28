"""
Audio Input Tester for CryingSense

Test your microphone and visualize audio input without needing a trained model.
Useful for verifying hardware setup and audio quality before running inference.

Features:
- Real-time audio level monitoring
- Frequency spectrum visualization
- Audio quality indicators
- Recording test capability
"""

import os
import sys
import time
import wave
import numpy as np
import pyaudio
from datetime import datetime
from collections import deque

# Import sound level utilities
from sound_level_meter import calculate_db, create_level_bar, get_level_status


class AudioTester:
    """Audio input tester without model dependency."""
    
    def __init__(self, sample_rate=16000, device_index=None):
        """
        Initialize audio tester.
        
        Args:
            sample_rate: Audio sample rate (default: 16000 Hz)
            device_index: Audio device index (None for default)
        """
        self.sample_rate = sample_rate
        self.device_index = device_index
        
        # Audio parameters
        self.chunk_size = 1024
        self.channels = 1
        self.audio_format = pyaudio.paInt16
        
        # State tracking
        self.running = False
        self.peak_db = -100
        self.min_db = 0
        self.avg_db_history = deque(maxlen=100)
        self.clipping_count = 0
        self.silence_count = 0
        self.total_chunks = 0
        
        # Spectrum analyzer bands
        self.freq_bands = [
            (0, 100, "Sub-bass"),
            (100, 300, "Bass"),
            (300, 1000, "Low-mid"),
            (1000, 3000, "Mid"),
            (3000, 6000, "High-mid"),
            (6000, 8000, "High"),
        ]
        
        # PyAudio
        self.audio = pyaudio.PyAudio()
    
    def _compute_spectrum(self, audio_chunk):
        """Compute frequency spectrum from audio chunk."""
        # Apply window and FFT
        windowed = audio_chunk * np.hanning(len(audio_chunk))
        fft = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(audio_chunk), 1/self.sample_rate)
        
        # Compute power in each band
        band_powers = []
        for low, high, name in self.freq_bands:
            mask = (freqs >= low) & (freqs < high)
            if np.any(mask):
                power = np.mean(fft[mask])
                # Convert to dB
                db = 20 * np.log10(power + 1e-10) - 60
                band_powers.append((name, db))
            else:
                band_powers.append((name, -60))
        
        return band_powers
    
    def _draw_display(self, current_db, spectrum=None):
        """Draw the audio tester display."""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 70)
        print("  CRYINGSENSE AUDIO TESTER")
        print("=" * 70)
        print()
        
        # Audio level meter
        print("  AUDIO LEVEL")
        print("  " + "-" * 66)
        
        bar = create_level_bar(current_db, width=50)
        status, symbol = get_level_status(current_db)
        
        print(f"  Level: [{bar}] {current_db:6.1f} dB")
        print(f"  Peak:  {self.peak_db:6.1f} dB  |  Min: {self.min_db:6.1f} dB")
        print(f"  Status: [{symbol}] {status}")
        print()
        
        # Average level
        if self.avg_db_history:
            avg_db = np.mean(list(self.avg_db_history))
            avg_bar = create_level_bar(avg_db, width=50)
            print(f"  Avg:   [{avg_bar}] {avg_db:6.1f} dB")
        print()
        
        # Spectrum analyzer
        if spectrum:
            print("  FREQUENCY SPECTRUM")
            print("  " + "-" * 66)
            
            max_bar_len = 40
            for name, db in spectrum:
                # Normalize dB to 0-1 range (roughly -60 to 0 dB)
                normalized = max(0, min(1, (db + 60) / 60))
                bar_len = int(normalized * max_bar_len)
                bar = "█" * bar_len + "·" * (max_bar_len - bar_len)
                print(f"  {name:10s} [{bar}] {db:5.1f} dB")
            print()
        
        # Quality indicators
        print("  QUALITY INDICATORS")
        print("  " + "-" * 66)
        
        # Clipping indicator
        clip_pct = (self.clipping_count / max(1, self.total_chunks)) * 100
        clip_status = "⚠️  CLIPPING DETECTED" if clip_pct > 1 else "✓ OK"
        print(f"  Clipping: {clip_status} ({clip_pct:.1f}%)")
        
        # Silence indicator
        silence_pct = (self.silence_count / max(1, self.total_chunks)) * 100
        if silence_pct > 80:
            silence_status = "⚠️  MOSTLY SILENT"
        elif silence_pct > 50:
            silence_status = "Low audio"
        else:
            silence_status = "✓ Audio detected"
        print(f"  Silence:  {silence_status} ({silence_pct:.1f}%)")
        
        # Sample rate / format
        print(f"  Format:   {self.sample_rate} Hz, 16-bit mono")
        print()
        
        # Statistics
        print("  " + "-" * 66)
        print(f"  Chunks: {self.total_chunks}  |  "
              f"Runtime: {self.total_chunks * self.chunk_size / self.sample_rate:.1f}s  |  "
              f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print()
        print("=" * 70)
        print("  Press Ctrl+C to stop  |  Press 'r' then Enter to record test clip")
        print("=" * 70)
    
    def run(self):
        """Start continuous audio testing."""
        print("\n" + "=" * 70)
        print("Starting CryingSense Audio Tester")
        print("=" * 70)
        print(f"Sample Rate: {self.sample_rate} Hz")
        print(f"Chunk Size: {self.chunk_size}")
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
        print("Audio stream opened. Monitoring started.\n")
        time.sleep(1)
        
        try:
            while self.running:
                # Read audio chunk
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                except Exception as e:
                    print(f"Error reading audio: {e}")
                    continue
                
                # Convert to numpy
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                self.total_chunks += 1
                
                # Calculate dB level
                current_db = calculate_db(audio_chunk)
                self.avg_db_history.append(current_db)
                
                # Update peak/min
                if current_db > self.peak_db:
                    self.peak_db = current_db
                if current_db < self.min_db and current_db > -60:
                    self.min_db = current_db
                
                # Check for clipping
                if np.max(np.abs(audio_chunk)) > 32000:
                    self.clipping_count += 1
                
                # Check for silence
                if current_db < -50:
                    self.silence_count += 1
                
                # Compute spectrum
                spectrum = self._compute_spectrum(audio_chunk.astype(np.float32))
                
                # Update display
                self._draw_display(current_db, spectrum)
                
                # Small delay
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\n\nStopping audio tester...")
        finally:
            self.running = False
            stream.stop_stream()
            stream.close()
            
            # Print summary
            self._print_summary()
    
    def _print_summary(self):
        """Print session summary."""
        print("\n" + "=" * 70)
        print("AUDIO TEST SUMMARY")
        print("=" * 70)
        
        runtime = self.total_chunks * self.chunk_size / self.sample_rate
        print(f"Total Runtime: {runtime:.1f} seconds")
        print(f"Peak Level: {self.peak_db:.1f} dB")
        print(f"Min Level: {self.min_db:.1f} dB")
        
        if self.avg_db_history:
            print(f"Average Level: {np.mean(list(self.avg_db_history)):.1f} dB")
        
        clip_pct = (self.clipping_count / max(1, self.total_chunks)) * 100
        silence_pct = (self.silence_count / max(1, self.total_chunks)) * 100
        
        print()
        print("Quality Assessment:")
        
        issues = []
        if clip_pct > 5:
            issues.append(f"  ⚠️  Clipping detected ({clip_pct:.1f}%) - Lower input gain")
        if silence_pct > 80:
            issues.append(f"  ⚠️  Low audio ({silence_pct:.1f}% silent) - Check microphone")
        if self.peak_db < -40:
            issues.append(f"  ⚠️  Very low levels (peak {self.peak_db:.1f} dB) - Increase gain")
        
        if issues:
            for issue in issues:
                print(issue)
        else:
            print("  ✓ Audio quality looks good!")
        
        print("=" * 70)
    
    def record_test(self, duration=5.0, output_dir='recordings'):
        """Record a test audio clip."""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nRecording {duration}s test clip...")
        
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
            return None
        
        frames = []
        num_chunks = int(self.sample_rate * duration / self.chunk_size)
        
        for i in range(num_chunks):
            data = stream.read(self.chunk_size, exception_on_overflow=False)
            frames.append(data)
            
            # Progress
            progress = (i + 1) / num_chunks
            bar_len = int(progress * 40)
            bar = "█" * bar_len + "·" * (40 - bar_len)
            remaining = duration - (i + 1) * self.chunk_size / self.sample_rate
            print(f"\r  Recording: [{bar}] {remaining:.1f}s remaining", end="", flush=True)
        
        stream.stop_stream()
        stream.close()
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_recording_{timestamp}.wav"
        filepath = os.path.join(output_dir, filename)
        
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
        
        print(f"\n  Saved: {filepath}")
        
        # Analyze recording
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        db = calculate_db(audio_data)
        peak = np.max(np.abs(audio_data)) / 32768.0
        
        print(f"  Level: {db:.1f} dB | Peak: {peak:.3f}")
        
        return filepath
    
    def list_devices(self):
        """List available audio devices."""
        print("\n" + "=" * 60)
        print("Available Audio Input Devices")
        print("=" * 60)
        info = self.audio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        
        try:
            default_device = self.audio.get_default_input_device_info()
            default_idx = default_device.get('index')
        except:
            default_idx = -1
        
        for i in range(num_devices):
            device_info = self.audio.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels') > 0:
                marker = " [DEFAULT]" if i == default_idx else ""
                sr = int(device_info.get('defaultSampleRate'))
                print(f"  Device {i}: {device_info.get('name')}{marker}")
                print(f"            Sample Rate: {sr} Hz, Channels: {device_info.get('maxInputChannels')}")
        
        print("=" * 60)
    
    def close(self):
        """Clean up resources."""
        self.running = False
        self.audio.terminate()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='CryingSense Audio Input Tester (no model required)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start live audio testing
  python tester.py
  
  # Record a 5-second test clip
  python tester.py --record
  
  # Record a 10-second clip
  python tester.py --record --duration 10
  
  # Use specific audio device
  python tester.py --device 1
  
  # List available audio devices
  python tester.py --list-devices
        """
    )
    
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='Audio sample rate (default: 16000)')
    parser.add_argument('--device', type=int, default=None,
                       help='Audio device index (default: system default)')
    parser.add_argument('--record', action='store_true',
                       help='Record a test audio clip instead of live monitoring')
    parser.add_argument('--duration', type=float, default=5.0,
                       help='Recording duration in seconds (default: 5.0)')
    parser.add_argument('--output', type=str, default='recordings',
                       help='Output directory for recordings (default: recordings)')
    parser.add_argument('--list-devices', action='store_true',
                       help='List available audio devices and exit')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = AudioTester(
        sample_rate=args.sample_rate,
        device_index=args.device
    )
    
    try:
        if args.list_devices:
            tester.list_devices()
        elif args.record:
            tester.record_test(duration=args.duration, output_dir=args.output)
        else:
            tester.run()
    finally:
        tester.close()


if __name__ == "__main__":
    main()
