"""
WiFi Audio Receiver for CryingSense

Receives audio data from ESP32 over WiFi using UDP protocol.
Provides a simple server that listens for incoming audio packets.
"""

import socket
import struct
import threading
import time
import logging
from typing import Optional, Callable
import numpy as np

from audio_buffer import AudioBuffer
import config

logger = logging.getLogger(__name__)


class WiFiAudioReceiver:
    """
    UDP server for receiving audio from ESP32.
    
    Protocol:
    - Each packet contains a 12-byte header + audio samples
    - Header: [packet_id (4B), timestamp (4B), sample_count (2B), flags (2B)]
    - Audio: 16-bit signed integers, little-endian
    """
    
    HEADER_SIZE = 12
    HEADER_FORMAT = "<IIHh"  # packet_id, timestamp_ms, sample_count, flags
    
    # Flags
    FLAG_FIRST_PACKET = 0x01
    FLAG_LAST_PACKET = 0x02
    FLAG_CRY_DETECTED = 0x04
    
    def __init__(self, audio_buffer: AudioBuffer, 
                 host: str = None, port: int = None):
        """
        Initialize WiFi receiver.
        
        Args:
            audio_buffer: Buffer to write received audio to
            host: Host to bind to (default from config)
            port: Port to listen on (default from config)
        """
        self.audio_buffer = audio_buffer
        self.host = host or config.WIFI_HOST
        self.port = port or config.WIFI_PORT
        
        self._socket: Optional[socket.socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Connection tracking
        self._connected_device: Optional[str] = None
        self._last_packet_time = 0
        self._packets_received = 0
        self._bytes_received = 0
        self._errors = 0
        
        # Callbacks
        self._on_connect: Optional[Callable] = None
        self._on_disconnect: Optional[Callable] = None
        self._on_error: Optional[Callable] = None
        
        # Device ID from ESP32
        self.device_id: Optional[str] = None
    
    def start(self) -> bool:
        """
        Start the receiver server.
        
        Returns:
            True if started successfully
        """
        if self._running:
            logger.warning("Receiver already running")
            return False
        
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind((self.host, self.port))
            self._socket.settimeout(1.0)  # Allow periodic checks
            
            self._running = True
            self._thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._thread.start()
            
            logger.info(f"WiFi receiver started on {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start receiver: {e}")
            self._cleanup()
            return False
    
    def stop(self) -> None:
        """Stop the receiver server."""
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        
        self._cleanup()
        logger.info("WiFi receiver stopped")
    
    def _cleanup(self) -> None:
        """Clean up socket resources."""
        if self._socket:
            try:
                self._socket.close()
            except:
                pass
            self._socket = None
    
    def _receive_loop(self) -> None:
        """Main receive loop (runs in separate thread)."""
        logger.info("Receiver loop started")
        
        while self._running:
            try:
                data, addr = self._socket.recvfrom(config.UDP_BUFFER_SIZE)
                self._process_packet(data, addr)
                
            except socket.timeout:
                # Check for device disconnect
                if self._connected_device and time.time() - self._last_packet_time > 5.0:
                    self._handle_disconnect()
                continue
                
            except Exception as e:
                self._errors += 1
                if self._on_error:
                    self._on_error(e)
                logger.error(f"Receive error: {e}")
                time.sleep(0.1)
        
        logger.info("Receiver loop ended")
    
    def _process_packet(self, data: bytes, addr: tuple) -> None:
        """
        Process received UDP packet.
        
        Args:
            data: Raw packet data
            addr: Sender address (ip, port)
        """
        if len(data) < self.HEADER_SIZE:
            logger.warning(f"Packet too small: {len(data)} bytes")
            return
        
        # Parse header
        header = data[:self.HEADER_SIZE]
        packet_id, timestamp_ms, sample_count, flags = struct.unpack(
            self.HEADER_FORMAT, header
        )
        
        # Extract audio samples
        audio_data = data[self.HEADER_SIZE:]
        expected_bytes = sample_count * 2  # 16-bit samples
        
        if len(audio_data) < expected_bytes:
            logger.warning(f"Audio data truncated: {len(audio_data)} < {expected_bytes}")
            sample_count = len(audio_data) // 2
        
        # Convert to numpy array
        samples = np.frombuffer(audio_data[:sample_count * 2], dtype=np.int16)
        
        # Track connection
        device_addr = f"{addr[0]}:{addr[1]}"
        if self._connected_device != device_addr:
            self._handle_connect(device_addr, flags)
        
        self._last_packet_time = time.time()
        self._packets_received += 1
        self._bytes_received += len(data)
        
        # Write to buffer
        self.audio_buffer.write(samples)
        
        # Check for device ID in first packet
        if flags & self.FLAG_FIRST_PACKET:
            # Device ID might be appended after audio
            remaining = audio_data[sample_count * 2:]
            if remaining:
                try:
                    self.device_id = remaining.decode('utf-8').strip('\x00')
                    logger.info(f"Device ID: {self.device_id}")
                except:
                    pass
    
    def _handle_connect(self, device_addr: str, flags: int) -> None:
        """Handle new device connection."""
        self._connected_device = device_addr
        logger.info(f"Device connected: {device_addr}")
        
        if self._on_connect:
            self._on_connect(device_addr)
    
    def _handle_disconnect(self) -> None:
        """Handle device disconnection."""
        logger.info(f"Device disconnected: {self._connected_device}")
        
        if self._on_disconnect:
            self._on_disconnect(self._connected_device)
        
        self._connected_device = None
        self.device_id = None
    
    def on_connect(self, callback: Callable) -> None:
        """Set connection callback."""
        self._on_connect = callback
    
    def on_disconnect(self, callback: Callable) -> None:
        """Set disconnection callback."""
        self._on_disconnect = callback
    
    def on_error(self, callback: Callable) -> None:
        """Set error callback."""
        self._on_error = callback
    
    @property
    def is_connected(self) -> bool:
        """Whether a device is currently connected."""
        return (self._connected_device is not None and 
                time.time() - self._last_packet_time < 5.0)
    
    @property
    def is_running(self) -> bool:
        """Whether the receiver is running."""
        return self._running
    
    def get_stats(self) -> dict:
        """Get receiver statistics."""
        return {
            "running": self._running,
            "connected": self.is_connected,
            "connected_device": self._connected_device,
            "device_id": self.device_id,
            "packets_received": self._packets_received,
            "bytes_received": self._bytes_received,
            "errors": self._errors,
            "last_packet_time": self._last_packet_time,
        }


class SerialAudioReceiver:
    """
    Fallback receiver for serial connection (USB).
    
    Parses the SAMPLES: format from ESP32 serial output.
    """
    
    def __init__(self, audio_buffer: AudioBuffer, port: str = "COM3", 
                 baudrate: int = 115200):
        """
        Initialize serial receiver.
        
        Args:
            audio_buffer: Buffer to write received audio to
            port: Serial port name
            baudrate: Serial baudrate
        """
        self.audio_buffer = audio_buffer
        self.port = port
        self.baudrate = baudrate
        
        self._serial = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Statistics
        self._lines_received = 0
        self._samples_received = 0
    
    def start(self) -> bool:
        """Start the serial receiver."""
        try:
            import serial
            self._serial = serial.Serial(self.port, self.baudrate, timeout=1.0)
            self._running = True
            self._thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._thread.start()
            logger.info(f"Serial receiver started on {self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start serial receiver: {e}")
            return False
    
    def stop(self) -> None:
        """Stop the serial receiver."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._serial:
            self._serial.close()
            self._serial = None
    
    def _receive_loop(self) -> None:
        """Main receive loop."""
        while self._running:
            try:
                line = self._serial.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith("SAMPLES:"):
                    self._process_samples(line[8:])
                    self._lines_received += 1
            except Exception as e:
                logger.error(f"Serial error: {e}")
                time.sleep(0.1)
    
    def _process_samples(self, data: str) -> None:
        """Process SAMPLES: line from ESP32."""
        try:
            values = [int(v) for v in data.split(',') if v.strip()]
            if values:
                # Upsample from 32 samples to approximate real sample rate
                # ESP32 sends 32 samples per ~32ms, we need 16000/s
                samples = np.array(values, dtype=np.int16)
                
                # Simple upsampling via interpolation
                target_samples = int(len(samples) * (config.SAMPLE_RATE * 0.032 / 32))
                if target_samples > len(samples):
                    indices = np.linspace(0, len(samples) - 1, target_samples)
                    samples = np.interp(indices, np.arange(len(samples)), samples).astype(np.int16)
                
                self.audio_buffer.write(samples)
                self._samples_received += len(samples)
        except Exception as e:
            logger.error(f"Sample parse error: {e}")
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    def get_stats(self) -> dict:
        return {
            "running": self._running,
            "port": self.port,
            "lines_received": self._lines_received,
            "samples_received": self._samples_received,
        }
