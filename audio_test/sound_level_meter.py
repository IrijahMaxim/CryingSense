"""
Sound Level Meter for CryingSense

Real-time audio visualization tool showing:
- Sound level in dB
- Visual level meter
- Frequency spectrum analysis
- Peak detection

Use this to monitor microphone input and verify audio quality
before recording or testing the model.
"""

import os
import sys
import time
import numpy as np
import pyaudio
from collections import deque
import threading


def calculate_db(audio_data):
    """
    Calculate dB level from audio data (utility function).
    
    Args:
        audio_data: Audio samples as numpy array (int16)
        
    Returns:
        float: dB level (typically -60 to 0 range)
    """
    rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
    if rms < 1:
        rms = 1
    db = 20 * np.log10(rms / 32767)
    return db


def create_level_bar(db_level, width=40):
    """
    Create ASCII level bar (utility function).
    
    Args:
        db_level: Current dB level
        width: Bar width in characters
        
    Returns:
        str: ASCII level bar
    """
    min_db = -60
    max_db = 0
    db_level = max(min_db, min(max_db, db_level))
    position = int((db_level - min_db) / (max_db - min_db) * width)
    
    bar = ""
    for i in range(width):
        if i < position:
            if i < width * 0.6:
                bar += "█"
            elif i < width * 0.8:
                bar += "▓"
            else:
                bar += "░"
        else:
            bar += "·"
    return bar


def get_level_status(db_level):
    """
    Get status description for dB level.
    
    Args:
        db_level: Current dB level
        
    Returns:
        tuple: (status_text, status_symbol)
    """
    if db_level > -10:
        return "LOUD", "!"
    elif db_level > -20:
        return "GOOD", "+"
    elif db_level > -40:
        return "NORMAL", "="
    elif db_level > -55:
        return "QUIET", "-"
    else:
        return "SILENT", "."


class SoundLevelMeter:
    """Real-time sound level meter with frequency visualization."""
    
    def __init__(self, sample_rate=16000, chunk_size=1024, channels=1):
        """
        Initialize sound level meter.
        
        Args:
            sample_rate: Sample rate in Hz (default: 16000)
            chunk_size: Number of samples per buffer (default: 1024)
            channels: Number of audio channels (default: 1)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.running = False
        
        # Level history for smoothing
        self.level_history = deque(maxlen=10)
        self.peak_level = -100
        self.peak_hold_time = 0
        
        # Frequency analysis
        self.freq_bins = 16  # Number of frequency bands to display
        self.freq_history = deque(maxlen=5)
        
    def list_devices(self):
        """List all available audio input devices."""
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
                print(f"    Max Input Channels: {device_info.get('maxInputChannels')}")
                print(f"    Default Sample Rate: {int(device_info.get('defaultSampleRate'))} Hz")
                print()
        print("=" * 60)
        return default_idx
    
    def calculate_db(self, audio_data):
        """
        Calculate dB level from audio data.
        
        Args:
            audio_data: Audio samples as numpy array
            
        Returns:
            float: dB level (typically -60 to 0 range)
        """
        # Calculate RMS (Root Mean Square)
        rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
        
        # Avoid log of zero
        if rms < 1:
            rms = 1
        
        # Convert to dB (reference: 32767 for 16-bit audio)
        db = 20 * np.log10(rms / 32767)
        
        return db
    
    def calculate_frequency_spectrum(self, audio_data):
        """
        Calculate frequency spectrum using FFT.
        
        Args:
            audio_data: Audio samples as numpy array
            
        Returns:
            numpy array: Magnitude for each frequency band
        """
        # Apply Hanning window
        windowed = audio_data * np.hanning(len(audio_data))
        
        # Compute FFT
        fft = np.fft.rfft(windowed)
        magnitude = np.abs(fft)
        
        # Split into frequency bands
        band_size = len(magnitude) // self.freq_bins
        bands = []
        
        for i in range(self.freq_bins):
            start = i * band_size
            end = start + band_size
            band_magnitude = np.mean(magnitude[start:end])
            # Convert to dB scale
            if band_magnitude < 1:
                band_magnitude = 1
            band_db = 20 * np.log10(band_magnitude / 32767)
            bands.append(band_db)
        
        return np.array(bands)
    
    def create_level_bar(self, db_level, width=50, show_peak=True):
        """
        Create ASCII level bar.
        
        Args:
            db_level: Current dB level
            width: Bar width in characters
            show_peak: Whether to show peak marker
            
        Returns:
            str: ASCII level bar
        """
        # Map dB to bar position (-60 to 0 dB range)
        min_db = -60
        max_db = 0
        
        # Clamp values
        db_level = max(min_db, min(max_db, db_level))
        
        # Calculate position
        position = int((db_level - min_db) / (max_db - min_db) * width)
        
        # Create bar with color zones
        bar = ""
        for i in range(width):
            if i < position:
                if i < width * 0.6:  # Green zone
                    bar += "█"
                elif i < width * 0.8:  # Yellow zone
                    bar += "▓"
                else:  # Red zone
                    bar += "░"
            else:
                bar += "·"
        
        # Add peak marker
        if show_peak and self.peak_level > min_db:
            peak_pos = int((self.peak_level - min_db) / (max_db - min_db) * width)
            peak_pos = min(peak_pos, width - 1)
            bar_list = list(bar)
            if peak_pos >= 0:
                bar_list[peak_pos] = "│"
            bar = "".join(bar_list)
        
        return bar
    
    def create_spectrum_display(self, spectrum, height=8, width=48):
        """
        Create ASCII frequency spectrum display.
        
        Args:
            spectrum: Array of frequency band levels
            height: Display height in lines
            width: Display width in characters
            
        Returns:
            list: Lines of ASCII spectrum
        """
        min_db = -60
        max_db = -10
        
        # Normalize spectrum to 0-height range
        normalized = []
        for level in spectrum:
            level = max(min_db, min(max_db, level))
            norm = int((level - min_db) / (max_db - min_db) * height)
            normalized.append(norm)
        
        # Calculate bar width
        bar_width = width // len(spectrum)
        
        # Create display lines
        lines = []
        for row in range(height, 0, -1):
            line = ""
            for band_level in normalized:
                if band_level >= row:
                    line += "█" * (bar_width - 1) + " "
                else:
                    line += "·" * (bar_width - 1) + " "
            lines.append(line)
        
        return lines
    
    def get_frequency_labels(self, width=48):
        """Get frequency labels for spectrum display."""
        labels = []
        nyquist = self.sample_rate / 2
        band_width = nyquist / self.freq_bins
        
        for i in range(self.freq_bins):
            freq = int((i + 0.5) * band_width)
            if freq >= 1000:
                labels.append(f"{freq//1000}k")
            else:
                labels.append(f"{freq}")
        
        bar_width = width // self.freq_bins
        label_line = ""
        for label in labels:
            label_line += label.center(bar_width)
        
        return label_line
    
    def run(self, device_index=None, duration=None):
        """
        Start the sound level meter.
        
        Args:
            device_index: Audio device index (None for default)
            duration: Run duration in seconds (None for infinite)
        """
        print("\n" + "=" * 60)
        print("CryingSense Sound Level Meter")
        print("=" * 60)
        print(f"Sample Rate: {self.sample_rate} Hz")
        print(f"Chunk Size: {self.chunk_size}")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")
        
        # Open audio stream
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size
            )
        except Exception as e:
            print(f"Error opening audio stream: {e}")
            print("\nTry running with --list-devices to see available devices")
            return
        
        self.running = True
        start_time = time.time()
        
        try:
            while self.running:
                # Check duration
                if duration and (time.time() - start_time) >= duration:
                    break
                
                # Read audio data
                try:
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                except Exception as e:
                    print(f"Error reading audio: {e}")
                    continue
                
                # Convert to numpy array
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # Calculate levels
                current_db = self.calculate_db(audio_data)
                self.level_history.append(current_db)
                smooth_db = np.mean(self.level_history)
                
                # Update peak
                if current_db > self.peak_level:
                    self.peak_level = current_db
                    self.peak_hold_time = time.time()
                elif time.time() - self.peak_hold_time > 2.0:  # Peak hold for 2 seconds
                    self.peak_level = max(self.peak_level - 1, -100)
                
                # Calculate frequency spectrum
                spectrum = self.calculate_frequency_spectrum(audio_data)
                self.freq_history.append(spectrum)
                smooth_spectrum = np.mean(self.freq_history, axis=0)
                
                # Clear screen and draw display
                self._draw_display(smooth_db, smooth_spectrum)
                
                # Small delay for display refresh
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\n\nStopping sound level meter...")
        finally:
            self.running = False
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
    
    def _draw_display(self, db_level, spectrum):
        """Draw the complete display."""
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 60)
        print("  SOUND LEVEL METER - CryingSense")
        print("=" * 60)
        print()
        
        # Level meter section
        print("  LEVEL METER")
        print("  " + "-" * 54)
        
        level_bar = self.create_level_bar(db_level)
        
        # dB scale
        print("  -60dB          -40dB          -20dB          0dB")
        print(f"  [{level_bar}]")
        
        # Current and peak levels
        peak_str = f"{self.peak_level:.1f}" if self.peak_level > -100 else "---"
        print(f"  Current: {db_level:6.1f} dB    Peak: {peak_str} dB")
        print()
        
        # Level classification
        if db_level > -10:
            status = "LOUD (clipping risk!)"
            status_color = "!"
        elif db_level > -20:
            status = "GOOD (strong signal)"
            status_color = "+"
        elif db_level > -40:
            status = "NORMAL"
            status_color = "="
        elif db_level > -55:
            status = "QUIET"
            status_color = "-"
        else:
            status = "VERY QUIET / SILENT"
            status_color = "."
        
        print(f"  Status: [{status_color}] {status}")
        print()
        
        # Frequency spectrum section
        print("  FREQUENCY SPECTRUM")
        print("  " + "-" * 54)
        
        spectrum_lines = self.create_spectrum_display(spectrum)
        for line in spectrum_lines:
            print(f"  {line}")
        
        # Frequency labels
        freq_labels = self.get_frequency_labels()
        print(f"  {freq_labels}")
        print(f"  {'Hz'.center(48)}")
        print()
        
        # Instructions
        print("=" * 60)
        print("  Press Ctrl+C to stop")
        print("=" * 60)
    
    def close(self):
        """Clean up resources."""
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()


class SimpleMeter:
    """Simplified sound level meter without screen clearing (for terminals that don't support it)."""
    
    def __init__(self, sample_rate=16000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio = pyaudio.PyAudio()
        self.running = False
        
    def run(self, device_index=None, duration=None):
        """Run simplified meter that updates in place."""
        print("\n" + "=" * 60)
        print("CryingSense Sound Level Meter (Simple Mode)")
        print("=" * 60)
        print("Press Ctrl+C to stop\n")
        
        try:
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size
            )
        except Exception as e:
            print(f"Error: {e}")
            return
        
        self.running = True
        start_time = time.time()
        peak_db = -100
        
        try:
            while self.running:
                if duration and (time.time() - start_time) >= duration:
                    break
                
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # Calculate dB
                rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
                if rms < 1:
                    rms = 1
                db = 20 * np.log10(rms / 32767)
                
                # Update peak
                if db > peak_db:
                    peak_db = db
                
                # Create simple bar
                bar_width = 40
                position = int((db + 60) / 60 * bar_width)
                position = max(0, min(bar_width, position))
                bar = "█" * position + "·" * (bar_width - position)
                
                # Print update
                print(f"\r  [{bar}] {db:6.1f} dB  (Peak: {peak_db:6.1f} dB)  ", end="", flush=True)
                
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\n\nStopped.")
        finally:
            stream.stop_stream()
            stream.close()
    
    def close(self):
        self.audio.terminate()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Real-time Sound Level Meter for CryingSense'
    )
    parser.add_argument('--device', type=int, default=None,
                       help='Audio device index (default: system default)')
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='Sample rate in Hz (default: 16000)')
    parser.add_argument('--duration', type=float, default=None,
                       help='Run duration in seconds (default: infinite)')
    parser.add_argument('--list-devices', action='store_true',
                       help='List available audio devices and exit')
    parser.add_argument('--simple', action='store_true',
                       help='Use simple mode (no screen clearing)')
    
    args = parser.parse_args()
    
    if args.simple:
        meter = SimpleMeter(sample_rate=args.sample_rate)
    else:
        meter = SoundLevelMeter(sample_rate=args.sample_rate)
    
    try:
        if args.list_devices:
            meter.list_devices()
        else:
            meter.run(device_index=args.device, duration=args.duration)
    finally:
        meter.close()


if __name__ == "__main__":
    main()
