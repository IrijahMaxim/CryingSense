"""
Thread-Safe Audio Buffer for CryingSense

Circular buffer for real-time audio streaming with thread safety.
Supports concurrent read/write operations from WiFi receiver and classifier.
"""

import threading
import numpy as np
from collections import deque
from typing import Optional, Tuple
import time


class AudioBuffer:
    """
    Thread-safe circular audio buffer for real-time streaming.
    
    Features:
    - Lock-free reads for visualization
    - Thread-safe writes from network receiver
    - Automatic overflow handling
    - Pre-cry buffer preservation
    """
    
    def __init__(self, max_duration: float, sample_rate: int = 16000):
        """
        Initialize audio buffer.
        
        Args:
            max_duration: Maximum buffer duration in seconds
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        
        # Main circular buffer
        self._buffer = np.zeros(self.max_samples, dtype=np.int16)
        self._write_index = 0
        self._samples_written = 0
        
        # Thread synchronization
        self._lock = threading.RLock()
        self._data_event = threading.Event()
        
        # Timestamp tracking
        self._last_write_time = time.time()
        self._start_time = time.time()
        
        # Statistics
        self._total_samples_received = 0
        self._overflow_count = 0
    
    def write(self, samples: np.ndarray) -> int:
        """
        Write samples to buffer (thread-safe).
        
        Args:
            samples: Audio samples to write (int16 array)
        
        Returns:
            Number of samples written
        """
        if len(samples) == 0:
            return 0
        
        # Convert to int16 if necessary
        if samples.dtype != np.int16:
            samples = samples.astype(np.int16)
        
        with self._lock:
            n_samples = len(samples)
            
            # Handle overflow (wrap around)
            if n_samples >= self.max_samples:
                # Only keep the last max_samples
                samples = samples[-self.max_samples:]
                n_samples = self.max_samples
                self._overflow_count += 1
            
            # Calculate write positions
            end_index = self._write_index + n_samples
            
            if end_index <= self.max_samples:
                # Simple case: no wrap
                self._buffer[self._write_index:end_index] = samples
            else:
                # Wrap around
                first_part = self.max_samples - self._write_index
                self._buffer[self._write_index:] = samples[:first_part]
                self._buffer[:n_samples - first_part] = samples[first_part:]
            
            # Update indices
            self._write_index = end_index % self.max_samples
            self._samples_written = min(self._samples_written + n_samples, self.max_samples)
            self._total_samples_received += n_samples
            self._last_write_time = time.time()
        
        # Signal that new data is available
        self._data_event.set()
        
        return n_samples
    
    def read(self, duration: float, offset: float = 0.0) -> np.ndarray:
        """
        Read samples from buffer (thread-safe).
        
        Args:
            duration: Duration in seconds to read
            offset: Offset from current position in seconds (0 = most recent)
        
        Returns:
            Audio samples as int16 array
        """
        n_samples = int(duration * self.sample_rate)
        offset_samples = int(offset * self.sample_rate)
        
        with self._lock:
            available = self._samples_written
            
            if available == 0:
                return np.zeros(n_samples, dtype=np.int16)
            
            # Clamp to available samples
            total_needed = n_samples + offset_samples
            if total_needed > available:
                n_samples = max(0, available - offset_samples)
                if n_samples <= 0:
                    return np.zeros(int(duration * self.sample_rate), dtype=np.int16)
            
            # Calculate read start position
            read_start = (self._write_index - offset_samples - n_samples) % self.max_samples
            
            # Read samples
            if read_start + n_samples <= self.max_samples:
                result = self._buffer[read_start:read_start + n_samples].copy()
            else:
                # Wrap around read
                first_part = self.max_samples - read_start
                result = np.concatenate([
                    self._buffer[read_start:],
                    self._buffer[:n_samples - first_part]
                ])
            
            return result
    
    def read_latest(self, n_samples: int) -> np.ndarray:
        """
        Read the most recent N samples.
        
        Args:
            n_samples: Number of samples to read
        
        Returns:
            Audio samples as int16 array
        """
        with self._lock:
            available = min(n_samples, self._samples_written)
            
            if available == 0:
                return np.zeros(n_samples, dtype=np.int16)
            
            read_start = (self._write_index - available) % self.max_samples
            
            if read_start + available <= self.max_samples:
                result = self._buffer[read_start:read_start + available].copy()
            else:
                first_part = self.max_samples - read_start
                result = np.concatenate([
                    self._buffer[read_start:],
                    self._buffer[:available - first_part]
                ])
            
            # Pad if necessary
            if len(result) < n_samples:
                result = np.pad(result, (n_samples - len(result), 0), mode='constant')
            
            return result
    
    def get_all(self) -> np.ndarray:
        """
        Get all available samples in chronological order.
        
        Returns:
            All buffered samples as int16 array
        """
        with self._lock:
            if self._samples_written == 0:
                return np.zeros(0, dtype=np.int16)
            
            if self._samples_written < self.max_samples:
                # Buffer not full yet
                return self._buffer[:self._samples_written].copy()
            else:
                # Buffer is full, need to reorder
                return np.concatenate([
                    self._buffer[self._write_index:],
                    self._buffer[:self._write_index]
                ])
    
    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.fill(0)
            self._write_index = 0
            self._samples_written = 0
            self._data_event.clear()
    
    def wait_for_data(self, timeout: float = None) -> bool:
        """
        Wait for new data to be available.
        
        Args:
            timeout: Maximum time to wait in seconds
        
        Returns:
            True if data is available, False if timeout
        """
        result = self._data_event.wait(timeout)
        self._data_event.clear()
        return result
    
    @property
    def duration(self) -> float:
        """Current buffer duration in seconds."""
        with self._lock:
            return self._samples_written / self.sample_rate
    
    @property
    def is_full(self) -> bool:
        """Whether buffer is at capacity."""
        with self._lock:
            return self._samples_written >= self.max_samples
    
    @property
    def samples_available(self) -> int:
        """Number of samples currently in buffer."""
        with self._lock:
            return self._samples_written
    
    @property
    def time_since_last_write(self) -> float:
        """Time since last write in seconds."""
        return time.time() - self._last_write_time
    
    def get_stats(self) -> dict:
        """Get buffer statistics."""
        with self._lock:
            return {
                "samples_available": self._samples_written,
                "duration_seconds": self._samples_written / self.sample_rate,
                "buffer_utilization": self._samples_written / self.max_samples,
                "total_samples_received": self._total_samples_received,
                "overflow_count": self._overflow_count,
                "last_write_time": self._last_write_time,
            }


class RecordingBuffer:
    """
    Buffer specifically for recording detected cries.
    
    Accumulates audio during cry detection and provides
    the complete recording when cry ends.
    """
    
    def __init__(self, sample_rate: int = 16000, max_duration: float = 30.0):
        """
        Initialize recording buffer.
        
        Args:
            sample_rate: Audio sample rate
            max_duration: Maximum recording duration
        """
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        
        self._chunks: list = []
        self._total_samples = 0
        self._lock = threading.Lock()
        self._recording = False
        self._start_time = None
    
    def start(self, pre_buffer: np.ndarray = None) -> None:
        """
        Start recording.
        
        Args:
            pre_buffer: Optional audio to include before trigger point
        """
        with self._lock:
            self._chunks = []
            self._total_samples = 0
            self._recording = True
            self._start_time = time.time()
            
            if pre_buffer is not None and len(pre_buffer) > 0:
                self._chunks.append(pre_buffer.copy())
                self._total_samples = len(pre_buffer)
    
    def append(self, samples: np.ndarray) -> bool:
        """
        Append samples to recording.
        
        Args:
            samples: Audio samples to append
        
        Returns:
            True if appended, False if buffer full or not recording
        """
        with self._lock:
            if not self._recording:
                return False
            
            if self._total_samples + len(samples) > self.max_samples:
                # Would exceed max duration
                return False
            
            self._chunks.append(samples.copy())
            self._total_samples += len(samples)
            return True
    
    def stop(self) -> np.ndarray:
        """
        Stop recording and return the complete audio.
        
        Returns:
            Complete recording as int16 array
        """
        with self._lock:
            self._recording = False
            
            if not self._chunks:
                return np.zeros(0, dtype=np.int16)
            
            recording = np.concatenate(self._chunks)
            self._chunks = []
            self._total_samples = 0
            
            return recording
    
    @property
    def is_recording(self) -> bool:
        """Whether currently recording."""
        return self._recording
    
    @property
    def duration(self) -> float:
        """Current recording duration in seconds."""
        with self._lock:
            return self._total_samples / self.sample_rate
