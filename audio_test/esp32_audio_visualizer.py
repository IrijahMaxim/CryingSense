"""
ESP32 INMP441 Audio Visualizer

Real-time visualization of audio from ESP32 INMP441 microphone via Serial:
- Amplitude waveform display
- Frequency spectrum (FFT) analysis
- Sound level monitoring
- Baby cry detection visualization

Usage: python esp32_audio_visualizer.py [COM_PORT]
Example: python esp32_audio_visualizer.py COM3
"""

import sys
import numpy as np
import serial
import serial.tools.list_ports
from collections import deque
import struct
import re

# PyQtGraph for fast real-time plotting
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLabel, QPushButton, QComboBox)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import pyqtgraph as pg


class SerialReaderThread(QThread):
    """Thread for reading serial data from ESP32."""
    
    data_received = pyqtSignal(np.ndarray)  # Emits audio samples
    status_received = pyqtSignal(str)  # Emits status messages
    
    def __init__(self, port, baudrate=115200, sample_rate=16000):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.sample_rate = sample_rate
        self.running = False
        self.serial_conn = None
        self.buffer = deque(maxlen=2048)
        
    def run(self):
        """Run the serial reading loop."""
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=0.05)
            self.serial_conn.reset_input_buffer()
            self.running = True
            print(f"Connected to {self.port}")
            
            raw_samples = []
            
            while self.running:
                # Clear excessive backlog to reduce lag
                if self.serial_conn.in_waiting > 1000:
                    self.serial_conn.reset_input_buffer()
                
                if self.serial_conn.in_waiting > 0:
                    try:
                        line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                        
                        if line:
                            # Check if it's raw samples data
                            if line.startswith('SAMPLES:'):
                                try:
                                    samples_str = line.split(':', 1)[1]
                                    samples = [float(x.strip()) for x in samples_str.split(',') if x.strip()]
                                    if samples:
                                        samples_array = np.array(samples, dtype=np.float32)
                                        self.data_received.emit(samples_array)
                                        continue
                                except Exception as e:
                                    print(f"Error parsing samples: {e}")
                            
                            # Check if it's a status message
                            if any(keyword in line for keyword in ['CRYING', 'Status:', 'RMS:', 'Amp:', 'TOO LOUD', 'Baby', 'Ready', 'Quiet', 'Normal', 'LOUD CRY']):
                                self.status_received.emit(line)
                            
                            # Try to parse numeric data (Amplitude or RMS values)
                            # Format: "Amp: 1234 | Peak: 5678 | LED: 100 | Status: CRYING"
                            amp_match = re.search(r'(?:Amp|RMS):\s*([\d.]+)', line)
                            if amp_match:
                                amp_value = float(amp_match.group(1))
                                # Simulate samples from amplitude (for visualization when raw samples not available)
                                # Generate synthetic waveform based on amplitude
                                num_samples = 64
                                phase = np.random.rand() * 2 * np.pi
                                synthetic_samples = amp_value * np.sin(np.linspace(phase, phase + 2*np.pi, num_samples))
                                self.data_received.emit(synthetic_samples.astype(np.float32))
                            else:
                                # Try parsing as plain number
                                try:
                                    value = float(line)
                                    if -10000 < value < 10000:  # Reasonable range for audio samples
                                        raw_samples.append(value)
                                        
                                        # Emit when we have enough samples
                                        if len(raw_samples) >= 64:
                                            samples_array = np.array(raw_samples, dtype=np.float32)
                                            self.data_received.emit(samples_array)
                                            raw_samples = []
                                except ValueError:
                                    pass
                                    
                    except Exception as e:
                        print(f"Error parsing line: {e}")
                        
        except serial.SerialException as e:
            print(f"Serial error: {e}")
        finally:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
                
    def stop(self):
        """Stop the serial reading thread."""
        self.running = False
        self.wait()


class ESP32AudioVisualizer(QMainWindow):
    """Real-time ESP32 INMP441 audio visualizer."""
    
    def __init__(self, port=None, sample_rate=16000):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.port = port
        
        # Audio buffers
        self.waveform_duration = 2.0  # Show 2 seconds
        self.waveform_samples = int(sample_rate * self.waveform_duration)
        self.waveform_buffer = deque(maxlen=self.waveform_samples)
        
        # Initialize with zeros
        for _ in range(self.waveform_samples):
            self.waveform_buffer.append(0)
        
        # Frequency spectrum
        self.fft_size = 1024
        self.freq_bins = self.fft_size // 2
        self.spectrum = np.zeros(self.freq_bins)
        self.freq_axis = np.fft.rfftfreq(self.fft_size, 1/sample_rate)[:-1]
        
        # Status
        self.current_rms = 0
        self.peak_rms = 0
        self.status_text = "Initializing..."
        
        # Serial thread
        self.serial_thread = None
        
        self._setup_ui()
        
        # Update timer for UI refresh
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_plots)
        self.timer.start(50)  # 20 FPS
        
        # Auto-connect if port specified
        if port:
            self._connect_serial(port)
    
    def _setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle('ESP32 INMP441 Audio Visualizer')
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #1e1e1e;")
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(10)
        
        # Title and controls
        header_layout = QHBoxLayout()
        
        title = QLabel('🎤 ESP32 INMP441 Audio Visualizer')
        title.setFont(QFont('Arial', 18, QFont.Bold))
        title.setStyleSheet("color: #00BFFF; padding: 10px;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Port selection
        port_label = QLabel("Port:")
        port_label.setStyleSheet("color: white;")
        header_layout.addWidget(port_label)
        
        self.port_combo = QComboBox()
        self.port_combo.setStyleSheet("""
            QComboBox {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #555;
                padding: 5px;
                min-width: 100px;
            }
        """)
        self._refresh_ports()
        header_layout.addWidget(self.port_combo)
        
        # Connect button
        self.connect_btn = QPushButton('Connect')
        self.connect_btn.setStyleSheet("""
            QPushButton {
                background-color: #00AA00;
                color: white;
                border: none;
                padding: 8px 20px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #00CC00;
            }
            QPushButton:pressed {
                background-color: #008800;
            }
        """)
        self.connect_btn.clicked.connect(self._on_connect_clicked)
        header_layout.addWidget(self.connect_btn)
        
        main_layout.addLayout(header_layout)
        
        # Configure pyqtgraph
        pg.setConfigOptions(antialias=True, useOpenGL=False)
        
        # Waveform plot
        self.waveform_plot = pg.PlotWidget(title='<span style="color: #00BFFF; font-size: 14pt;">Audio Waveform</span>')
        self.waveform_plot.setBackground('#2d2d2d')
        self.waveform_plot.setLabel('left', 'Amplitude', color='#aaa', **{'font-size': '12pt'})
        self.waveform_plot.setLabel('bottom', 'Time (s)', color='#aaa', **{'font-size': '12pt'})
        self.waveform_plot.showGrid(x=True, y=True, alpha=0.3)
        self.waveform_plot.addLine(y=0, pen=pg.mkPen('#666', width=1))
        
        time_axis = np.linspace(0, self.waveform_duration, self.waveform_samples)
        self.waveform_curve = self.waveform_plot.plot(
            time_axis, np.zeros(self.waveform_samples),
            pen=pg.mkPen('#00BFFF', width=2)
        )
        main_layout.addWidget(self.waveform_plot, stretch=3)
        
        # RMS and Status display
        info_layout = QHBoxLayout()
        
        self.rms_label = QLabel('RMS: 0.0')
        self.rms_label.setFont(QFont('Consolas', 20, QFont.Bold))
        self.rms_label.setStyleSheet("color: #00FF00; padding: 10px; background-color: #2d2d2d; border-radius: 5px;")
        self.rms_label.setMinimumWidth(200)
        info_layout.addWidget(self.rms_label)
        
        self.peak_label = QLabel('Peak: 0.0')
        self.peak_label.setFont(QFont('Consolas', 16))
        self.peak_label.setStyleSheet("color: #FF6B6B; padding: 10px;")
        info_layout.addWidget(self.peak_label)
        
        info_layout.addStretch()
        
        self.status_label = QLabel('Status: Ready')
        self.status_label.setFont(QFont('Arial', 14))
        self.status_label.setStyleSheet("color: #FFD700; padding: 10px; background-color: #2d2d2d; border-radius: 5px;")
        info_layout.addWidget(self.status_label)
        
        main_layout.addLayout(info_layout)
        
        # Frequency spectrum
        self.spectrum_plot = pg.PlotWidget(title='<span style="color: #00FF88; font-size: 14pt;">Frequency Spectrum (FFT)</span>')
        self.spectrum_plot.setBackground('#2d2d2d')
        self.spectrum_plot.setLabel('left', 'Magnitude (dB)', color='#aaa', **{'font-size': '12pt'})
        self.spectrum_plot.setLabel('bottom', 'Frequency (Hz)', color='#aaa', **{'font-size': '12pt'})
        self.spectrum_plot.setYRange(-60, 20)
        self.spectrum_plot.setXRange(0, self.sample_rate / 2)
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Add frequency markers for baby cry range (300-600 Hz fundamental, 2-4 kHz harmonics)
        self.spectrum_plot.addLine(x=300, pen=pg.mkPen('#FF6B6B', width=1, style=Qt.DashLine))
        self.spectrum_plot.addLine(x=600, pen=pg.mkPen('#FF6B6B', width=1, style=Qt.DashLine))
        
        self.spectrum_curve = self.spectrum_plot.plot(
            self.freq_axis, np.zeros(self.freq_bins),
            pen=pg.mkPen('#00FF88', width=2),
            fillLevel=-60,
            fillBrush=pg.mkBrush('#00FF8844')
        )
        main_layout.addWidget(self.spectrum_plot, stretch=2)
        
    def _refresh_ports(self):
        """Refresh available COM ports."""
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combo.clear()
        self.port_combo.addItems(ports if ports else ["No ports found"])
        
        # Auto-select if only one port or if port was specified
        if self.port and self.port in ports:
            self.port_combo.setCurrentText(self.port)
        
    def _on_connect_clicked(self):
        """Handle connect button click."""
        if self.serial_thread and self.serial_thread.running:
            # Disconnect
            self._disconnect_serial()
        else:
            # Connect
            port = self.port_combo.currentText()
            if port and port != "No ports found":
                self._connect_serial(port)
    
    def _connect_serial(self, port):
        """Connect to ESP32 serial port."""
        try:
            self.serial_thread = SerialReaderThread(port, sample_rate=self.sample_rate)
            self.serial_thread.data_received.connect(self._on_data_received)
            self.serial_thread.status_received.connect(self._on_status_received)
            self.serial_thread.start()
            
            self.connect_btn.setText('Disconnect')
            self.connect_btn.setStyleSheet("""
                QPushButton {
                    background-color: #CC0000;
                    color: white;
                    border: none;
                    padding: 8px 20px;
                    font-weight: bold;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #FF0000;
                }
            """)
            self.status_label.setText(f'Status: Connected to {port}')
            self.status_label.setStyleSheet("color: #00FF00; padding: 10px; background-color: #2d2d2d; border-radius: 5px;")
            
        except Exception as e:
            self.status_label.setText(f'Status: Connection Error - {str(e)}')
            self.status_label.setStyleSheet("color: #FF0000; padding: 10px; background-color: #2d2d2d; border-radius: 5px;")
    
    def _disconnect_serial(self):
        """Disconnect from serial port."""
        if self.serial_thread:
            self.serial_thread.stop()
            self.serial_thread = None
        
        self.connect_btn.setText('Connect')
        self.connect_btn.setStyleSheet("""
            QPushButton {
                background-color: #00AA00;
                color: white;
                border: none;
                padding: 8px 20px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #00CC00;
            }
        """)
        self.status_label.setText('Status: Disconnected')
        self.status_label.setStyleSheet("color: #FFD700; padding: 10px; background-color: #2d2d2d; border-radius: 5px;")
    
    def _on_data_received(self, samples):
        """Handle received audio data."""
        # Add samples to waveform buffer
        for sample in samples:
            self.waveform_buffer.append(sample)
        
        # Calculate RMS
        if len(samples) > 0:
            self.current_rms = np.sqrt(np.mean(samples**2))
            self.peak_rms = max(self.peak_rms, self.current_rms)
    
    def _on_status_received(self, status):
        """Handle received status messages."""
        self.status_text = status
        
        # Update status label with color coding
        if 'CRYING LOUDLY' in status or 'TOO LOUD' in status:
            color = '#FF0000'
        elif 'crying' in status.lower():
            color = '#FFA500'
        else:
            color = '#00FF00'
        
        self.status_label.setText(f'Status: {status}')
        self.status_label.setStyleSheet(f"color: {color}; padding: 10px; background-color: #2d2d2d; border-radius: 5px;")
    
    def _update_plots(self):
        """Update the plots."""
        # Update waveform
        waveform_data = np.array(self.waveform_buffer)
        if len(waveform_data) > 0:
            # Normalize for display
            max_val = np.max(np.abs(waveform_data)) if np.max(np.abs(waveform_data)) > 0 else 1
            normalized = waveform_data / max_val if max_val > 0 else waveform_data
            
            time_axis = np.linspace(0, self.waveform_duration, len(normalized))
            self.waveform_curve.setData(time_axis, normalized)
            
            # Update RMS label
            self.rms_label.setText(f'RMS: {self.current_rms:.1f}')
            self.peak_label.setText(f'Peak: {self.peak_rms:.1f}')
            
            # Update spectrum (FFT)
            if len(waveform_data) >= self.fft_size:
                # Take last fft_size samples
                samples_for_fft = waveform_data[-self.fft_size:]
                
                # Apply Hanning window
                windowed = samples_for_fft * np.hanning(self.fft_size)
                
                # Compute FFT
                fft_result = np.fft.rfft(windowed)[:-1]
                magnitude = np.abs(fft_result)
                
                # Convert to dB
                magnitude_db = 20 * np.log10(magnitude + 1e-10)
                
                # Smooth the spectrum
                self.spectrum_curve.setData(self.freq_axis, magnitude_db)
    
    def closeEvent(self, event):
        """Handle window close event."""
        self._disconnect_serial()
        event.accept()


def main():
    """Main entry point."""
    # Check for COM port argument
    port = None
    if len(sys.argv) > 1:
        port = sys.argv[1]
    
    # List available ports
    print("Available COM ports:")
    ports = serial.tools.list_ports.comports()
    for p in ports:
        print(f"  - {p.device}: {p.description}")
    
    if not ports:
        print("No COM ports found!")
    
    if port:
        print(f"\nConnecting to {port}...")
    else:
        print("\nNo port specified. Use GUI to select a port.")
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create and show visualizer
    visualizer = ESP32AudioVisualizer(port=port)
    visualizer.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
