"""
Real-Time Waveform Display for CryingSense

Displays live audio waveform using pygame.
Shows classification results and system status.
"""

import sys
import threading
import time
from typing import Optional, Dict, Tuple
from pathlib import Path
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from audio_buffer import AudioBuffer
import config

# Try to import pygame (optional for headless mode)
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not available. Display will run in headless mode.")


class WaveformDisplay:
    """
    Real-time waveform visualization with pygame.
    
    Features:
    - Live waveform display
    - Classification result display
    - Connection status
    - Recording indicator
    """
    
    def __init__(self, audio_buffer: AudioBuffer):
        """
        Initialize display.
        
        Args:
            audio_buffer: Audio buffer to visualize
        """
        self.audio_buffer = audio_buffer
        
        # Display dimensions
        self.width = config.WAVEFORM_WIDTH
        self.height = config.WAVEFORM_HEIGHT
        self.total_height = self.height + 150  # Extra space for info
        
        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Display state
        self._connected = False
        self._recording = False
        self._cry_detected = False
        self._current_prediction: Optional[Dict] = None
        self._status_text = "Initializing..."
        
        # Pygame surfaces
        self._screen = None
        self._clock = None
        self._font = None
        self._font_large = None
        
        # Lock for thread-safe updates
        self._lock = threading.Lock()
    
    def start(self) -> bool:
        """
        Start the display window.
        
        Returns:
            True if started successfully
        """
        if not PYGAME_AVAILABLE:
            print("Running in headless mode (no display)")
            return False
        
        if self._running:
            return False
        
        self._running = True
        self._thread = threading.Thread(target=self._display_loop, daemon=True)
        self._thread.start()
        
        return True
    
    def stop(self) -> None:
        """Stop the display."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
    
    def _display_loop(self) -> None:
        """Main display loop (runs in separate thread)."""
        try:
            pygame.init()
            self._screen = pygame.display.set_mode((self.width, self.total_height))
            pygame.display.set_caption("CryingSense - Real-Time Monitor")
            self._clock = pygame.time.Clock()
            self._font = pygame.font.Font(None, 24)
            self._font_large = pygame.font.Font(None, 48)
            
            while self._running:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._running = False
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self._running = False
                            return
                
                # Draw frame
                self._draw_frame()
                
                pygame.display.flip()
                self._clock.tick(config.DISPLAY_FPS)
            
        except Exception as e:
            print(f"Display error: {e}")
        finally:
            pygame.quit()
    
    def _draw_frame(self) -> None:
        """Draw a single frame."""
        # Background
        self._screen.fill(config.COLOR_BACKGROUND)
        
        # Draw waveform
        self._draw_waveform()
        
        # Draw info panel
        self._draw_info_panel()
        
        # Draw status bar
        self._draw_status_bar()
    
    def _draw_waveform(self) -> None:
        """Draw the audio waveform."""
        # Get audio samples for display
        display_samples = int(2.0 * config.SAMPLE_RATE)  # 2 seconds
        audio = self.audio_buffer.read_latest(display_samples)
        
        if len(audio) == 0:
            return
        
        # Downsample for display
        step = max(1, len(audio) // self.width)
        display_audio = audio[::step][:self.width]
        
        # Normalize to display height
        if np.max(np.abs(display_audio)) > 0:
            normalized = display_audio / 32768.0
        else:
            normalized = np.zeros(len(display_audio))
        
        # Center line
        center_y = self.height // 2
        
        # Choose color based on state
        with self._lock:
            if self._cry_detected:
                color = config.COLOR_WAVEFORM_ALERT
            else:
                color = config.COLOR_WAVEFORM
        
        # Draw waveform
        points = []
        for i, sample in enumerate(normalized):
            y = int(center_y - sample * (self.height // 2 - 10))
            y = max(5, min(self.height - 5, y))
            points.append((i, y))
        
        if len(points) > 1:
            pygame.draw.lines(self._screen, color, False, points, 2)
        
        # Draw center line
        pygame.draw.line(
            self._screen, 
            (100, 100, 100), 
            (0, center_y), 
            (self.width, center_y), 
            1
        )
        
        # Draw amplitude threshold lines
        threshold_y = int(center_y - (config.AMPLITUDE_THRESHOLD / 32768.0) * (self.height // 2))
        pygame.draw.line(
            self._screen,
            (80, 80, 100),
            (0, threshold_y),
            (self.width, threshold_y),
            1
        )
        pygame.draw.line(
            self._screen,
            (80, 80, 100),
            (0, 2 * center_y - threshold_y),
            (self.width, 2 * center_y - threshold_y),
            1
        )
    
    def _draw_info_panel(self) -> None:
        """Draw the information panel below waveform."""
        panel_y = self.height + 10
        
        with self._lock:
            prediction = self._current_prediction
            cry_detected = self._cry_detected
            recording = self._recording
        
        # Status indicator
        if cry_detected:
            status_color = config.COLOR_CRY_DETECTED
            status_text = "CRY DETECTED"
        else:
            status_color = config.COLOR_LISTENING
            status_text = "LISTENING"
        
        # Draw status badge
        pygame.draw.rect(
            self._screen,
            status_color,
            (10, panel_y, 150, 30),
            border_radius=5
        )
        text = self._font.render(status_text, True, (0, 0, 0))
        self._screen.blit(text, (20, panel_y + 7))
        
        # Recording indicator
        if recording:
            pygame.draw.circle(self._screen, (255, 50, 50), (180, panel_y + 15), 8)
            text = self._font.render("REC", True, (255, 100, 100))
            self._screen.blit(text, (195, panel_y + 7))
        
        # Classification result
        if prediction:
            class_name = prediction['class'].upper().replace('_', ' ')
            confidence = prediction['confidence']
            
            # Don't show noise/speech
            if prediction['class'] not in config.IGNORE_CLASSES:
                # Class name
                text = self._font_large.render(class_name, True, (255, 255, 255))
                self._screen.blit(text, (10, panel_y + 40))
                
                # Confidence bar
                bar_x = 10
                bar_y = panel_y + 90
                bar_width = 300
                bar_height = 20
                
                # Background
                pygame.draw.rect(
                    self._screen,
                    (50, 50, 60),
                    (bar_x, bar_y, bar_width, bar_height),
                    border_radius=3
                )
                
                # Fill
                fill_width = int(bar_width * confidence)
                fill_color = self._confidence_color(confidence)
                pygame.draw.rect(
                    self._screen,
                    fill_color,
                    (bar_x, bar_y, fill_width, bar_height),
                    border_radius=3
                )
                
                # Percentage text
                pct_text = f"{confidence:.1%}"
                text = self._font.render(pct_text, True, (255, 255, 255))
                self._screen.blit(text, (bar_x + bar_width + 10, bar_y + 2))
                
                # All probabilities (smaller)
                probs_y = panel_y + 40
                probs_x = 350
                for cls_name, prob in sorted(
                    prediction['probabilities'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                ):
                    if cls_name not in config.IGNORE_CLASSES:
                        color = (200, 200, 200) if cls_name == prediction['class'] else (120, 120, 120)
                        text = self._font.render(f"{cls_name}: {prob:.1%}", True, color)
                        self._screen.blit(text, (probs_x, probs_y))
                        probs_y += 20
    
    def _draw_status_bar(self) -> None:
        """Draw status bar at bottom."""
        bar_y = self.total_height - 25
        
        # Connection status
        with self._lock:
            connected = self._connected
            status_text = self._status_text
        
        if connected:
            pygame.draw.circle(self._screen, (0, 255, 100), (15, bar_y + 10), 5)
            text = self._font.render("Connected", True, (100, 255, 100))
        else:
            pygame.draw.circle(self._screen, (255, 100, 100), (15, bar_y + 10), 5)
            text = self._font.render("Disconnected", True, (255, 100, 100))
        
        self._screen.blit(text, (25, bar_y + 2))
        
        # Status text
        text = self._font.render(status_text, True, (150, 150, 150))
        self._screen.blit(text, (150, bar_y + 2))
        
        # Buffer info
        buffer_info = f"Buffer: {self.audio_buffer.duration:.1f}s"
        text = self._font.render(buffer_info, True, (150, 150, 150))
        self._screen.blit(text, (self.width - 120, bar_y + 2))
    
    def _confidence_color(self, confidence: float) -> Tuple[int, int, int]:
        """Get color based on confidence level."""
        if confidence >= 0.9:
            return (50, 255, 100)
        elif confidence >= 0.7:
            return (255, 220, 50)
        elif confidence >= 0.5:
            return (255, 150, 50)
        else:
            return (255, 80, 80)
    
    def set_connected(self, connected: bool) -> None:
        """Update connection status."""
        with self._lock:
            self._connected = connected
    
    def set_recording(self, recording: bool) -> None:
        """Update recording status."""
        with self._lock:
            self._recording = recording
    
    def set_cry_detected(self, detected: bool) -> None:
        """Update cry detection status."""
        with self._lock:
            self._cry_detected = detected
    
    def set_prediction(self, prediction: Dict) -> None:
        """Update current prediction."""
        with self._lock:
            self._current_prediction = prediction
    
    def set_status(self, text: str) -> None:
        """Update status text."""
        with self._lock:
            self._status_text = text
    
    @property
    def is_running(self) -> bool:
        """Whether display is running."""
        return self._running


class TerminalDisplay:
    """
    Fallback terminal-based display for headless mode.
    
    Prints status updates to terminal.
    """
    
    def __init__(self, audio_buffer: AudioBuffer):
        """Initialize terminal display."""
        self.audio_buffer = audio_buffer
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # State
        self._connected = False
        self._recording = False
        self._cry_detected = False
        self._current_prediction: Optional[Dict] = None
        self._lock = threading.Lock()
    
    def start(self) -> bool:
        """Start terminal display."""
        self._running = True
        self._thread = threading.Thread(target=self._display_loop, daemon=True)
        self._thread.start()
        return True
    
    def stop(self) -> None:
        """Stop terminal display."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
    
    def _display_loop(self) -> None:
        """Main display loop."""
        while self._running:
            self._print_status()
            time.sleep(0.5)
    
    def _print_status(self) -> None:
        """Print current status."""
        with self._lock:
            # Get audio amplitude
            audio = self.audio_buffer.read_latest(config.SAMPLE_RATE)
            amplitude = int(np.abs(audio).mean()) if len(audio) > 0 else 0
            
            # Build status line
            status_parts = []
            
            if self._connected:
                status_parts.append("🟢 Connected")
            else:
                status_parts.append("🔴 Waiting")
            
            status_parts.append(f"Amp: {amplitude:5d}")
            
            if self._cry_detected:
                status_parts.append("🔊 CRY")
            else:
                status_parts.append("   Listening")
            
            if self._recording:
                status_parts.append("⏺️ REC")
            
            if self._current_prediction:
                pred = self._current_prediction
                if pred['class'] not in config.IGNORE_CLASSES:
                    status_parts.append(
                        f"→ {pred['class'].upper()}: {pred['confidence']:.1%}"
                    )
            
            # Print with carriage return for overwrite
            status_line = " | ".join(status_parts)
            print(f"\r{status_line:<100}", end="", flush=True)
    
    def set_connected(self, connected: bool) -> None:
        with self._lock:
            self._connected = connected
    
    def set_recording(self, recording: bool) -> None:
        with self._lock:
            self._recording = recording
    
    def set_cry_detected(self, detected: bool) -> None:
        with self._lock:
            self._cry_detected = detected
    
    def set_prediction(self, prediction: Dict) -> None:
        with self._lock:
            self._current_prediction = prediction
    
    def set_status(self, text: str) -> None:
        pass  # Not used in terminal mode
    
    @property
    def is_running(self) -> bool:
        return self._running
