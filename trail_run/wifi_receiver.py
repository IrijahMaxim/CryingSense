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

try:
    from .audio_buffer import AudioBuffer
    from . import config
except ImportError:
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

            # Use an exclusive bind so only one receiver process can own the UDP port.
            # This avoids silent packet splitting when multiple app instances run.
            if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            self._socket.bind((self.host, self.port))
            self._socket.settimeout(1.0)  # Allow periodic checks
            
            self._running = True
            self._thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._thread.start()
            
            logger.info(f"WiFi receiver started on {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start receiver on {self.host}:{self.port}: {e}")
            logger.error("Tip: close other CryingSense instances using the same UDP port.")
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
    
    Parses binary packets from ESP32 serial output.

    Packet format:
    - Sync bytes: 0xAA 0x55
    - Header: [packet_id (4B), timestamp (4B), sample_count (2B), flags (2B)]
    - Audio: int16 PCM samples (sample_count * 2 bytes)
    """

    SYNC_BYTES = b"\xAA\x55"
    HEADER_SIZE = 12
    HEADER_FORMAT = "<IIHH"  # packet_id, timestamp_ms, sample_count, flags
    MAX_SAMPLES_PER_PACKET = 4096
    
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
        self._rx_buffer = bytearray()
        
        # Connection tracking
        self._connected = False
        self._last_data_time = 0
        
        # Statistics
        self._packets_received = 0
        self._samples_received = 0
        self._bytes_received = 0
        self._errors = 0
        
        # Callbacks
        self._on_connect: Optional[Callable] = None
        self._on_disconnect: Optional[Callable] = None
        self._on_error: Optional[Callable] = None
    
    def start(self) -> bool:
        """Start the serial receiver."""
        try:
            import serial
            self._serial = serial.Serial(self.port, self.baudrate, timeout=0.2)
            self._running = True
            self._thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._thread.start()
            logger.info(f"Serial receiver started on {self.port}")
            
            # Mark as connected and trigger callback
            self._connected = True
            self._last_data_time = time.time()
            if self._on_connect:
                self._on_connect(self.port)
            
            return True
        except Exception as e:
            logger.error(f"Failed to start serial receiver: {e}")
            if self._on_error:
                self._on_error(e)
            return False
    
    def stop(self) -> None:
        """Stop the serial receiver."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._serial:
            self._serial.close()
            self._serial = None
        
        # Trigger disconnect callback
        if self._connected and self._on_disconnect:
            self._on_disconnect(self.port)
        self._connected = False
    
    def _receive_loop(self) -> None:
        """Main receive loop."""
        while self._running:
            try:
                chunk = self._serial.read(4096)
                if not chunk:
                    continue

                self._bytes_received += len(chunk)
                self._rx_buffer.extend(chunk)
                self._process_buffer()
            except Exception as e:
                logger.error(f"Serial error: {e}")
                self._errors += 1
                if self._on_error:
                    self._on_error(e)
                time.sleep(0.1)

    def _process_buffer(self) -> None:
        """Parse as many complete packets as possible from the receive buffer."""
        while True:
            sync_idx = self._rx_buffer.find(self.SYNC_BYTES)

            if sync_idx == -1:
                # Keep only a small tail for sync-byte overlap across reads.
                if len(self._rx_buffer) > 1:
                    self._rx_buffer = self._rx_buffer[-1:]
                return

            if sync_idx > 0:
                # Drop noise/log bytes before sync.
                del self._rx_buffer[:sync_idx]

            if len(self._rx_buffer) < 2 + self.HEADER_SIZE:
                return

            header_start = 2
            header_end = header_start + self.HEADER_SIZE

            try:
                packet_id, timestamp_ms, sample_count, flags = struct.unpack(
                    self.HEADER_FORMAT,
                    self._rx_buffer[header_start:header_end]
                )
            except struct.error:
                del self._rx_buffer[:2]
                self._errors += 1
                continue

            if sample_count == 0 or sample_count > self.MAX_SAMPLES_PER_PACKET:
                # Invalid header; resync after first sync byte.
                del self._rx_buffer[:1]
                self._errors += 1
                continue

            payload_bytes = sample_count * 2
            packet_size = 2 + self.HEADER_SIZE + payload_bytes

            if len(self._rx_buffer) < packet_size:
                return

            payload_start = header_end
            payload_end = payload_start + payload_bytes
            payload = self._rx_buffer[payload_start:payload_end]

            samples = np.frombuffer(payload, dtype=np.int16)
            if samples.size:
                self.audio_buffer.write(samples)
                self._samples_received += samples.size
                self._packets_received += 1
                self._last_data_time = time.time()

            del self._rx_buffer[:packet_size]
    
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
        """Whether serial port is connected and receiving data."""
        return (self._connected and self._running and 
                time.time() - self._last_data_time < 5.0)
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    def get_stats(self) -> dict:
        return {
            "running": self._running,
            "connected": self.is_connected,
            "port": self.port,
            "packets_received": self._packets_received,
            "samples_received": self._samples_received,
            "bytes_received": self._bytes_received,
            "errors": self._errors,
        }


class MicrophoneAudioReceiver:
    """
    Computer microphone receiver for testing without ESP32.
    
    Uses sounddevice to capture audio from system microphone.
    """
    
    def __init__(self, audio_buffer: AudioBuffer, device_index: int = None,
                 sample_rate: int = None):
        """
        Initialize microphone receiver.
        
        Args:
            audio_buffer: Buffer to write received audio to
            device_index: Microphone device index (None for default)
            sample_rate: Sample rate (default from config)
        """
        self.audio_buffer = audio_buffer
        self.device_index = device_index
        self.sample_rate = sample_rate or config.SAMPLE_RATE
        
        self._stream = None
        self._running = False
        
        # Connection tracking
        self._connected = False
        self._last_data_time = 0
        
        # Statistics
        self._samples_received = 0
        self._chunks_received = 0
        
        # Callbacks
        self._on_connect: Optional[Callable] = None
        self._on_disconnect: Optional[Callable] = None
        self._on_error: Optional[Callable] = None
    
    def start(self) -> bool:
        """Start the microphone receiver."""
        try:
            import sounddevice as sd
            
            # List available devices
            devices = sd.query_devices()
            logger.info(f"Available audio devices: {len(devices)}")
            
            # Get default input device if not specified
            if self.device_index is None:
                default_device = sd.query_devices(kind='input')
                logger.info(f"Using default microphone: {default_device['name']}")
            else:
                device_info = sd.query_devices(self.device_index)
                logger.info(f"Using microphone #{self.device_index}: {device_info['name']}")
            
            # Create input stream
            self._stream = sd.InputStream(
                device=self.device_index,
                channels=1,
                samplerate=self.sample_rate,
                dtype='int16',
                blocksize=512,  # Match ESP32 buffer size
                callback=self._audio_callback
            )
            
            self._stream.start()
            self._running = True
            self._connected = True
            self._last_data_time = time.time()
            
            logger.info(f"Microphone started at {self.sample_rate}Hz")
            
            # Trigger connect callback
            if self._on_connect:
                self._on_connect("Computer Microphone")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start microphone: {e}")
            if self._on_error:
                self._on_error(e)
            return False
    
    def stop(self) -> None:
        """Stop the microphone receiver."""
        self._running = False
        
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        
        # Trigger disconnect callback
        if self._connected and self._on_disconnect:
            self._on_disconnect("Computer Microphone")
        self._connected = False
        
        logger.info("Microphone stopped")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Audio callback called by sounddevice."""
        if status:
            logger.warning(f"Microphone status: {status}")
            if self._on_error:
                self._on_error(status)
        
        if not self._running:
            return
        
        # Convert to 1D int16 array
        samples = indata[:, 0].astype(np.int16)
        
        # Write to buffer
        self.audio_buffer.write(samples)
        
        # Update statistics
        self._samples_received += len(samples)
        self._chunks_received += 1
        self._last_data_time = time.time()
    
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
        """Whether microphone is connected and receiving data."""
        return (self._connected and self._running and 
                time.time() - self._last_data_time < 5.0)
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    def get_stats(self) -> dict:
        return {
            "running": self._running,
            "connected": self.is_connected,
            "device": "Computer Microphone",
            "sample_rate": self.sample_rate,
            "chunks_received": self._chunks_received,
            "samples_received": self._samples_received,
            "bytes_received": self._samples_received * 2,
        }
