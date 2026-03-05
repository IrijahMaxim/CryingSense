"""
Sound Level Meter V2 for CryingSense

Real-time audio visualization with PyQtGraph GUI (fast):
- Amplitude waveform display
- Sound level in dB
- Frequency spectrum analysis

Use this to monitor microphone input and verify audio quality.

Requirements: pip install pyqtgraph PyQt5
"""

import sys
import numpy as np
import pyaudio
from collections import deque

# PyQtGraph for fast real-time plotting
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont
import pyqtgraph as pg


class SoundLevelMeterV2(QMainWindow):
    """Real-time sound level meter with PyQtGraph visualization."""
    
    def __init__(self, sample_rate=16000, chunk_size=1024, channels=1, 
                 waveform_duration=7.0, device_index=None):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.waveform_duration = waveform_duration
        self.device_index = device_index
        
        # Audio setup
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Waveform buffer
        self.waveform_samples = int(sample_rate * waveform_duration)
        self.waveform_buffer = np.zeros(self.waveform_samples)
        self.time_axis = np.linspace(0, waveform_duration, self.waveform_samples)
        
        # Level tracking
        self.current_db = -60
        self.peak_db = -60
        
        # Frequency spectrum
        self.freq_bins = 128
        self.spectrum = np.zeros(self.freq_bins)
        self.freq_axis = np.linspace(0, sample_rate / 2, self.freq_bins)
        
        self._setup_ui()
        self._start_audio()
        
        # Update timer (20ms = 50fps)
        self.timer = QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(20)
    
    def _setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle('CryingSense Sound Level Meter V2')
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet("background-color: #1e1e1e;")
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(10)
        
        # Title
        title = QLabel('CryingSense Sound Level Meter')
        title.setFont(QFont('Arial', 16, QFont.Bold))
        title.setStyleSheet("color: #00BFFF; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Configure pyqtgraph
        pg.setConfigOptions(antialias=False, useOpenGL=False)
        
        # Waveform plot
        self.waveform_plot = pg.PlotWidget(title='Amplitude Waveform')
        self.waveform_plot.setLabel('left', 'Amplitude')
        self.waveform_plot.setLabel('bottom', 'Time (s)')
        self.waveform_plot.setYRange(-1, 1)
        self.waveform_plot.setXRange(0, self.waveform_duration)
        self.waveform_plot.showGrid(x=True, y=True, alpha=0.3)
        self.waveform_plot.addLine(y=0, pen=pg.mkPen('gray', width=1))
        self.waveform_curve = self.waveform_plot.plot(
            self.time_axis, self.waveform_buffer, 
            pen=pg.mkPen('#00BFFF', width=1)
        )
        layout.addWidget(self.waveform_plot, stretch=3)
        
        # Level meter section
        level_layout = QHBoxLayout()
        
        # dB label
        self.db_label = QLabel('-60.0 dB')
        self.db_label.setFont(QFont('Consolas', 24, QFont.Bold))
        self.db_label.setStyleSheet("color: #00FF00; padding: 10px;")
        self.db_label.setMinimumWidth(150)
        level_layout.addWidget(self.db_label)
        
        # Progress bar as level meter
        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 60)
        self.level_bar.setValue(0)
        self.level_bar.setTextVisible(False)
        self.level_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #333;
                border-radius: 5px;
                background-color: #2d2d2d;
                height: 30px;
            }
            QProgressBar::chunk {
                background-color: #00FF00;
                border-radius: 3px;
            }
        """)
        level_layout.addWidget(self.level_bar, stretch=1)
        
        # Peak label
        self.peak_label = QLabel('Peak: -60.0 dB')
        self.peak_label.setFont(QFont('Consolas', 12))
        self.peak_label.setStyleSheet("color: #FF6B6B; padding: 10px;")
        level_layout.addWidget(self.peak_label)
        
        layout.addLayout(level_layout)
        
        # Frequency spectrum
        self.spectrum_plot = pg.PlotWidget(title='Frequency Spectrum')
        self.spectrum_plot.setLabel('left', 'Magnitude (dB)')
        self.spectrum_plot.setLabel('bottom', 'Frequency (Hz)')
        self.spectrum_plot.setYRange(0, 80)
        self.spectrum_plot.setXRange(0, self.sample_rate / 2)
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Custom Y-axis labels to show dB scale (-80 to 0)
        y_axis = self.spectrum_plot.getAxis('left')
        y_axis.setTicks([[(0, '-80'), (20, '-60'), (40, '-40'), (60, '-20'), (80, '0')]])
        
        self.spectrum_bars = pg.BarGraphItem(
            x=self.freq_axis, 
            height=np.zeros(self.freq_bins), 
            width=(self.sample_rate / 2) / self.freq_bins * 0.8,
            brush='#FF6B6B'
        )
        self.spectrum_plot.addItem(self.spectrum_bars)
        layout.addWidget(self.spectrum_plot, stretch=2)
        
        # Status bar
        status = QLabel(f'Sample Rate: {self.sample_rate} Hz | Chunk: {self.chunk_size} | Duration: {self.waveform_duration}s')
        status.setStyleSheet("color: #888; padding: 5px;")
        status.setAlignment(Qt.AlignCenter)
        layout.addWidget(status)
    
    def _start_audio(self):
        """Start the audio stream."""
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size
            )
            print(f"Audio stream started (device: {self.device_index or 'default'})")
        except Exception as e:
            print(f"Error opening audio stream: {e}")
            self.stream = None
    
    def _update(self):
        """Update the display with new audio data."""
        if not self.stream:
            return
        
        try:
            # Read audio data
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            
            # Normalize to -1 to 1
            audio_normalized = audio_data / 32768.0
            
            # Update waveform buffer
            self.waveform_buffer = np.roll(self.waveform_buffer, -len(audio_normalized))
            self.waveform_buffer[-len(audio_normalized):] = audio_normalized
            
            # Calculate dB
            rms = np.sqrt(np.mean(audio_data ** 2))
            if rms < 1:
                rms = 1
            self.current_db = 20 * np.log10(rms / 32768.0)
            
            # Update peak
            if self.current_db > self.peak_db:
                self.peak_db = self.current_db
            
            # Calculate spectrum
            windowed = audio_data * np.hanning(len(audio_data))
            fft = np.fft.rfft(windowed)
            magnitude = np.abs(fft)
            
            # Resample to freq_bins
            indices = np.linspace(0, len(magnitude) - 1, self.freq_bins).astype(int)
            self.spectrum = np.zeros(self.freq_bins)
            for i, idx in enumerate(indices):
                mag = magnitude[idx]
                if mag < 1:
                    mag = 1
                self.spectrum[i] = 20 * np.log10(mag / 32768.0)
            
            # Update plots
            self.waveform_curve.setData(self.time_axis, self.waveform_buffer)
            
            # Update level meter
            db_display = max(-60, min(0, self.current_db))
            self.level_bar.setValue(int(db_display + 60))
            self.db_label.setText(f'{self.current_db:.1f} dB')
            self.peak_label.setText(f'Peak: {self.peak_db:.1f} dB')
            
            # Color based on level
            if db_display > -10:
                color = '#FF0000'
            elif db_display > -20:
                color = '#FFFF00'
            else:
                color = '#00FF00'
            self.db_label.setStyleSheet(f"color: {color}; padding: 10px;")
            self.level_bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 2px solid #333;
                    border-radius: 5px;
                    background-color: #2d2d2d;
                    height: 30px;
                }}
                QProgressBar::chunk {{
                    background-color: {color};
                    border-radius: 3px;
                }}
            """)
            
            # Update spectrum (shift from dB range -80 to 0 -> display range 0 to 80)
            spectrum_display = np.clip(self.spectrum + 80, 0, 80)
            self.spectrum_bars.setOpts(height=spectrum_display)
            
        except Exception as e:
            pass  # Ignore read errors
    
    def closeEvent(self, event):
        """Clean up on close."""
        self.timer.stop()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("Sound level meter closed.")
        event.accept()


def list_devices():
    """List available audio devices."""
    audio = pyaudio.PyAudio()
    print("\nAvailable Audio Input Devices:")
    print("=" * 50)
    info = audio.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    
    default_device = audio.get_default_input_device_info()
    default_idx = default_device.get('index')
    
    for i in range(num_devices):
        device_info = audio.get_device_info_by_host_api_device_index(0, i)
        if device_info.get('maxInputChannels') > 0:
            marker = " [DEFAULT]" if i == default_idx else ""
            print(f"  {i}: {device_info.get('name')}{marker}")
    print("=" * 50)
    audio.terminate()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Real-time Sound Level Meter V2 with GUI'
    )
    parser.add_argument('--device', type=int, default=None,
                       help='Audio device index (default: system default)')
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='Sample rate in Hz (default: 16000)')
    parser.add_argument('--duration', type=float, default=7.0,
                       help='Waveform display duration in seconds (default: 7.0)')
    parser.add_argument('--list-devices', action='store_true',
                       help='List available audio devices and exit')
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_devices()
        return
    
    app = QApplication(sys.argv)
    
    meter = SoundLevelMeterV2(
        sample_rate=args.sample_rate,
        waveform_duration=args.duration,
        device_index=args.device
    )
    meter.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
