"""
CryingSense Trail Run Configuration

Central configuration for the real-time cry detection and classification system.
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
TRAIL_RUN_DIR = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "model" / "saved_models" / "cryingsense_cnn_best.pth"
RECORDINGS_DIR = TRAIL_RUN_DIR / "recordings"

# Create recordings directory if it doesn't exist
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
NUM_CLASSES = 6
CLASS_NAMES = ['belly_pain', 'burp', 'discomfort', 'hunger', 'noise', 'tired']
CRY_CLASSES = ['belly_pain', 'burp', 'discomfort', 'hunger', 'tired']  # Excludes noise/speech
IGNORE_CLASSES = ['noise', 'speech']  # Classes to ignore during detection
CONFIDENCE_THRESHOLD = 0.70  # Minimum confidence for valid prediction

# =============================================================================
# AUDIO CONFIGURATION
# =============================================================================
SAMPLE_RATE = 16000
DURATION = 5.0  # seconds
N_MFCC = 40
N_MELS = 128
N_CHROMA = 12
N_FFT = 1024
HOP_LENGTH = 512
BIT_DEPTH = 16
CHANNELS = 1

# Buffer settings
BUFFER_DURATION = 10.0  # Keep 10 seconds of audio in buffer
DETECTION_WINDOW = 2.0  # Window for cry detection (seconds)
MIN_CRY_DURATION = 0.5  # Minimum cry duration to trigger (seconds)

# =============================================================================
# WIFI / NETWORK CONFIGURATION
# =============================================================================
WIFI_HOST = "0.0.0.0"  # Listen on all interfaces
WIFI_PORT = 8888
ESP32_EXPECTED_SAMPLE_RATE = 16000
UDP_BUFFER_SIZE = 4096

# =============================================================================
# DATABASE CONFIGURATION (MongoDB Atlas)
# =============================================================================
MONGO_URI = "mongodb+srv://admin:adminpassword123@cryingsense.qpdshid.mongodb.net/cryingsense_db"
MONGO_DATABASE = "cryingsense_db"

# Collection names
COLLECTION_AUDIO_FILES = "audio_files"
COLLECTION_AUDIO_SESSIONS = "audio_sessions"
COLLECTION_CRY_CLASSIFICATIONS = "cry_classifications"
COLLECTION_DEVICE_REGISTRATIONS = "device_registrations"

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================
DEVICE_SOURCE = "esp32"
DEVICE_ID = None  # Will be set from ESP32 MAC address or auto-generated

# =============================================================================
# DISPLAY CONFIGURATION
# =============================================================================
WAVEFORM_WIDTH = 800
WAVEFORM_HEIGHT = 200
WAVEFORM_UPDATE_INTERVAL = 50  # ms
DISPLAY_FPS = 30

# Colors (RGB)
COLOR_BACKGROUND = (20, 20, 30)
COLOR_WAVEFORM = (0, 255, 100)
COLOR_WAVEFORM_ALERT = (255, 100, 100)
COLOR_TEXT = (255, 255, 255)
COLOR_CRY_DETECTED = (255, 50, 50)
COLOR_LISTENING = (100, 200, 100)

# =============================================================================
# DETECTION CONFIGURATION
# =============================================================================
AMPLITUDE_THRESHOLD = 500  # Minimum amplitude to consider as potential cry
CRY_DETECTION_CONSECUTIVE = 3  # Consecutive windows needed to confirm cry
POST_CRY_SILENCE = 2.0  # Seconds of silence after cry to stop recording
PRE_CRY_BUFFER = 1.0  # Seconds before cry to include in recording

# =============================================================================
# DEBUGGING
# =============================================================================
DEBUG_MODE = False
LOG_LEVEL = "INFO"
SAVE_ALL_AUDIO = False  # Save all audio, not just cries (for debugging)
