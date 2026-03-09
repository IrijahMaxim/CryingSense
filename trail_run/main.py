#!/usr/bin/env python3
"""
CryingSense Trail Run - Main Runner

Complete real-time cry detection and classification system:
1. Connects to ESP32 via WiFi to receive audio
2. Displays live waveform visualization
3. Ignores speech and noise classes
4. Records when cry is detected
5. Classifies cry type with CNN model
6. Sends results to MongoDB Atlas

Usage:
    python main.py                    # WiFi mode (UDP, default)
    python main.py --serial COM3      # Serial mode (USB)
    python main.py --wifi             # WiFi mode (UDP)
    python main.py --microphone       # Computer microphone mode
    python main.py --headless         # No display (terminal only)
    python main.py --no-db            # Skip database connection
"""

import sys
import os
import argparse
import logging
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

try:
    from . import config
    from .audio_buffer import AudioBuffer
    from .wifi_receiver import WiFiAudioReceiver, SerialAudioReceiver, MicrophoneAudioReceiver
    from .classifier import CryClassifier
    from .database_handler import DatabaseHandler
    from .waveform_display import WaveformDisplay, TerminalDisplay, PYGAME_AVAILABLE
except ImportError:
    import config
    from audio_buffer import AudioBuffer
    from wifi_receiver import WiFiAudioReceiver, SerialAudioReceiver, MicrophoneAudioReceiver
    from classifier import CryClassifier
    from database_handler import DatabaseHandler
    from waveform_display import WaveformDisplay, TerminalDisplay, PYGAME_AVAILABLE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.TRAIL_RUN_DIR / 'trail_run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CryingSenseSystem:
    """
    Main system orchestrator.
    
    Coordinates all components:
    - Audio receiver (WiFi/Serial)
    - Real-time classifier
    - Database handler
    - Waveform display
    """
    
    def __init__(self, use_serial: str = None, use_microphone: bool = False,
                 headless: bool = False, use_database: bool = True):
        """
        Initialize the system.
        
        Args:
            use_serial: Serial port name (None for WiFi mode)
            use_microphone: Use computer microphone instead of ESP32
            headless: Run without display
            use_database: Whether to use database
        """
        self.use_serial = use_serial
        self.use_microphone = use_microphone
        self.headless = headless
        self.use_database = use_database
        
        # Components
        self.audio_buffer: AudioBuffer = None
        self.receiver = None
        self.classifier: CryClassifier = None
        self.database: DatabaseHandler = None
        self.display = None
        
        # State
        self._running = False
        self._shutdown_event = False
        
        # Statistics
        self._start_time = None
        self._cries_saved = 0
        self._db_executor = None
    
    def initialize(self) -> bool:
        """
        Initialize all components.
        
        Returns:
            True if all components initialized successfully
        """
        logger.info("=" * 60)
        logger.info("CryingSense Trail Run System")
        logger.info("=" * 60)
        
        try:
            # 1. Create audio buffer
            logger.info("Initializing audio buffer...")
            self.audio_buffer = AudioBuffer(
                max_duration=config.BUFFER_DURATION,
                sample_rate=config.SAMPLE_RATE
            )
            logger.info(f"  Buffer: {config.BUFFER_DURATION}s @ {config.SAMPLE_RATE}Hz")
            
            # 2. Initialize receiver
            if self.use_microphone:
                logger.info("Initializing computer microphone...")
                self.receiver = MicrophoneAudioReceiver(
                    audio_buffer=self.audio_buffer
                )
            elif self.use_serial:
                logger.info(f"Initializing serial receiver on {self.use_serial}...")
                self.receiver = SerialAudioReceiver(
                    audio_buffer=self.audio_buffer,
                    port=self.use_serial
                )
            else:
                logger.info(f"Initializing WiFi receiver on port {config.WIFI_PORT}...")
                self.receiver = WiFiAudioReceiver(
                    audio_buffer=self.audio_buffer
                )
            
            # Set receiver callbacks
            self.receiver.on_connect(self._on_device_connect)
            self.receiver.on_disconnect(self._on_device_disconnect)
            
            # 3. Initialize classifier
            logger.info("Initializing CNN classifier...")
            self.classifier = CryClassifier(
                audio_buffer=self.audio_buffer,
                model_path=str(config.MODEL_PATH)
            )
            
            # Set classifier callbacks
            self.classifier.on_cry_start(self._on_cry_start)
            self.classifier.on_cry_end(self._on_cry_end)
            self.classifier.on_classification(self._on_classification)
            
            logger.info(f"  Model: {config.MODEL_PATH.name}")
            logger.info(f"  Classes: {config.CLASS_NAMES}")
            logger.info(f"  Ignoring: {config.IGNORE_CLASSES}")
            
            # 4. Initialize database
            if self.use_database:
                logger.info("Connecting to MongoDB Atlas...")
                self.database = DatabaseHandler()
                
                if self.database.connect():
                    self._db_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="db-save")
                    # Register device
                    device_id = self.database.register_device(
                        device_type=config.DEVICE_TYPE,
                        firmware_version="1.0.0"
                    )
                    logger.info(f"  Device ID: {device_id}")
                    
                    # Start session
                    session_id = self.database.start_session()
                    logger.info(f"  Session ID: {session_id}")
                else:
                    logger.warning("Database connection failed - continuing without DB")
                    self.database = None
            else:
                logger.info("Database disabled")
                self.database = None
            
            # 5. Initialize display
            if self.headless or not PYGAME_AVAILABLE:
                logger.info("Using terminal display")
                self.display = TerminalDisplay(self.audio_buffer)
            else:
                logger.info("Initializing waveform display")
                self.display = WaveformDisplay(self.audio_buffer)
            
            logger.info("Initialization complete!")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def start(self) -> None:
        """Start all components and run main loop."""
        if not self.initialize():
            logger.error("Failed to initialize system")
            return
        
        try:
            # Start components
            logger.info("Starting components...")
            
            if not self.receiver.start():
                logger.error("Failed to start receiver")
                return
            
            self.classifier.start()
            self.display.start()
            
            self._running = True
            self._start_time = time.time()
            
            logger.info("System running!")
            if self.use_microphone:
                logger.info("Listening on computer microphone")
            elif self.use_serial:
                logger.info(f"Listening on serial {self.use_serial}")
            else:
                logger.info(f"Listening on UDP port {config.WIFI_PORT}")
            logger.info("Press Ctrl+C to stop")
            print()
            
            # Main loop
            while self._running and not self._shutdown_event:
                try:
                    # Update display with current state
                    self.display.set_connected(self.receiver.is_connected)
                    self.display.set_cry_detected(self.classifier.is_cry_detected)
                    self.display.set_recording(self.classifier.recording_buffer.is_recording)
                    
                    prediction = self.classifier.current_prediction
                    if prediction:
                        self.display.set_prediction(prediction)
                    
                    # Check if display closed
                    if hasattr(self.display, 'is_running') and not self.display.is_running:
                        self._running = False
                        break
                    
                    time.sleep(0.1)
                    
                except KeyboardInterrupt:
                    break
            
        except Exception as e:
            logger.error(f"Runtime error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop all components and cleanup."""
        logger.info("\nShutting down...")
        self._running = False
        
        # Stop components in reverse order
        if self.display:
            self.display.stop()
        
        if self.classifier:
            self.classifier.stop()
        
        if self.receiver:
            self.receiver.stop()
        
        # End database session
        if self.database:
            self.database.end_session()
            self.database.disconnect()

        if self._db_executor:
            self._db_executor.shutdown(wait=False, cancel_futures=False)
            self._db_executor = None
        
        # Print summary
        self._print_summary()
    
    def _on_device_connect(self, device_addr: str) -> None:
        """Handle device connection."""
        logger.info(f"Device connected: {device_addr}")
        self.display.set_status(f"Connected: {device_addr}")
        
        # Update device registration with address
        if self.database:
            self.database.register_device(
                device_id=self.database.device_id,
                metadata={"last_address": device_addr}
            )
    
    def _on_device_disconnect(self, device_addr: str) -> None:
        """Handle device disconnection."""
        logger.info(f"Device disconnected: {device_addr}")
        self.display.set_status("Offline mode (no packets for 5s)")

        # WiFi listener offline fallback: stop using stale audio immediately.
        if not self.use_serial and not self.use_microphone:
            self.audio_buffer.clear()
            if self.classifier:
                self.classifier.force_offline_reset()
    
    def _on_cry_start(self) -> None:
        """Handle cry detection start."""
        logger.info("Cry detected - recording started")
        self.display.set_status("Recording cry...")
    
    def _on_cry_end(self, result: dict) -> None:
        """Handle cry detection end with classification result."""
        if result['class'] in config.IGNORE_CLASSES:
            logger.info(f"Ignored: {result['class']} ({result['confidence']:.1%})")
            self.display.set_status("Listening...")
            return
        
        logger.info(f"Cry classified: {result['class']} ({result['confidence']:.1%})")
        
        # Save to database
        if self.database and 'audio' in result:
            try:
                if self._db_executor:
                    self._db_executor.submit(self._save_cry_event_async, result)
                    logger.info("  Queued database save")
                else:
                    saved = self.database.save_cry_event(result, result['audio'])
                    self._cries_saved += 1
                    logger.info(f"  Saved to database (files: {saved})")
            except Exception as e:
                logger.error(f"  Database save failed: {e}")
        
        # Update display
        self.display.set_prediction(result)
        self.display.set_status(
            f"Saved: {result['class'].upper()} ({result['confidence']:.1%})"
        )
    
    def _on_classification(self, result: dict) -> None:
        """Handle periodic classification during cry."""
        # Update display with live prediction (show ALL predictions, including noise)
        self.display.set_prediction(result)

    def _save_cry_event_async(self, result: dict) -> None:
        """Persist cry event in background so UI/classifier stays responsive."""
        if not self.database:
            return
        try:
            saved = self.database.save_cry_event(result, result['audio'])
            self._cries_saved += 1
            logger.info(f"  Saved to database (files: {saved})")
        except Exception as e:
            logger.error(f"  Database save failed (async): {e}")
    
    def _print_summary(self) -> None:
        """Print session summary."""
        if not self._start_time:
            return
        
        duration = time.time() - self._start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        
        print()
        print("=" * 60)
        print("Session Summary")
        print("=" * 60)
        print(f"Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        if self.classifier:
            stats = self.classifier.get_stats()
            print(f"Cries Detected: {stats['cries_detected']}")
            print(f"Classifications: {stats['classifications_made']}")
        
        if self.database:
            stats = self.database.get_stats()
            print(f"Saved to Database: {stats['classifications_saved']} classifications")
            print(f"                   {stats['audio_files_saved']} audio files")
        
        if self.receiver:
            stats = self.receiver.get_stats()
            bytes_received = stats.get('bytes_received', 0)
            if bytes_received > 1024 * 1024:
                print(f"Data Received: {bytes_received / 1024 / 1024:.1f} MB")
            else:
                print(f"Data Received: {bytes_received / 1024:.1f} KB")
        
        print("=" * 60)


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print("\nReceived shutdown signal...")
    global system
    if system:
        system._shutdown_event = True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CryingSense Real-Time Cry Detection System"
    )
    
    parser.add_argument(
        '--serial', '-s',
        type=str,
        default=None,
        help='Serial port for USB connection (default: disabled, WiFi mode)'
    )

    parser.add_argument(
        '--wifi',
        action='store_true',
        help='Force WiFi mode (disables serial mode)'
    )
    
    parser.add_argument(
        '--microphone', '-m',
        action='store_true',
        help='Use computer microphone instead of ESP32'
    )
    
    parser.add_argument(
        '--headless', '-H',
        action='store_true',
        help='Run without graphical display (terminal only)'
    )
    
    parser.add_argument(
        '--no-db', '-n',
        action='store_true',
        help='Disable database connection'
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=config.WIFI_PORT,
        help=f'UDP port for WiFi mode (default: {config.WIFI_PORT})'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    # Update config if port specified
    if args.port != config.WIFI_PORT:
        config.WIFI_PORT = args.port

    # WiFi mode explicitly disables serial receiver.
    if args.wifi:
        args.serial = None
    
    if args.debug:
        config.DEBUG_MODE = True
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup signal handlers
    global system
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run system
    system = CryingSenseSystem(
        use_serial=args.serial,
        use_microphone=args.microphone,
        headless=args.headless,
        use_database=not args.no_db
    )
    
    system.start()


# Global reference for signal handler
system: CryingSenseSystem = None


if __name__ == "__main__":
    main()
