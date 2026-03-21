"""
CryingSense Raspberry Pi 3B+ — Central Configuration

All tunables live here. Override any value via environment variables or a .env file
placed next to this script.

Optimised for:
  - Broadcom BCM2837B0, Cortex-A53 @ 1.4 GHz
  - 1 GB LPDDR2 SDRAM
  - 32 GB Micro-SD storage
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ────────────────────────────────────────────────────────────────
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)

# ── Paths ────────────────────────────────────────────────────────────────────
RPI_DIR = Path(__file__).parent
SAVED_MODELS_DIR = RPI_DIR / "saved_models"
RECORDINGS_DIR = RPI_DIR / "recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Model file — auto-detect first available format
MODEL_PATH: str = os.getenv("MODEL_PATH", "")
if not MODEL_PATH:
    for candidate in [
        SAVED_MODELS_DIR / "cryingsense_model.onnx",
        SAVED_MODELS_DIR / "cryingsense_model.pt",
        SAVED_MODELS_DIR / "cryingsense_quantized.pth",
    ]:
        if candidate.exists():
            MODEL_PATH = str(candidate)
            break

# ── Model ────────────────────────────────────────────────────────────────────
NUM_CLASSES: int = int(os.getenv("NUM_CLASSES", "6"))
CLASS_NAMES: list = ["belly_pain", "burp", "discomfort", "hunger", "noise", "tired"]
CRY_CLASSES: list = [c for c in CLASS_NAMES if c != "noise"]
INVISIBLE_CLASSES: list = ["noise"]
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))

# ── Audio ────────────────────────────────────────────────────────────────────
SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))
DURATION: float = float(os.getenv("DURATION", "5.0"))
CHANNELS: int = 1
BIT_DEPTH: int = 16

# Feature extraction
N_MFCC: int = 40
N_MELS: int = 128
N_CHROMA: int = 12
N_FFT: int = 1024
HOP_LENGTH: int = 512

# Preprocessing
TOP_DB: int = 20          # silence-trim threshold
NOISE_REDUCE: bool = os.getenv("NOISE_REDUCE", "true").lower() == "true"

# Microphone / recording
MIC_DEVICE_INDEX: int = int(os.getenv("MIC_DEVICE_INDEX", "0"))
LISTEN_CHUNK: int = int(os.getenv("LISTEN_CHUNK", "1024"))

# ── Database (MongoDB Atlas) ────────────────────────────────────────────────
MONGO_URI: str = os.getenv(
    "MONGO_URI",
    "mongodb://localhost:27017",
)
MONGO_DATABASE: str = os.getenv("MONGO_DATABASE", "cryingsense_db")

# Collection names — matching the existing schema
COL_AUDIO_FILES: str = "audio_files"
COL_AUDIO_SESSIONS: str = "audio_sessions"
COL_CRY_CLASSIFICATIONS: str = "cry_classifications"
COL_DEVICE_REGISTRATIONS: str = "device_registrations"

# Pool sizes tuned for 1 GB RAM
MONGO_MAX_POOL: int = int(os.getenv("MONGO_MAX_POOL", "5"))
MONGO_MIN_POOL: int = int(os.getenv("MONGO_MIN_POOL", "1"))
MONGO_TIMEOUT_MS: int = int(os.getenv("MONGO_TIMEOUT_MS", "5000"))

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE_TYPE: str = os.getenv("DEVICE_TYPE", "raspberry_pi")
DEVICE_ID: str = os.getenv("DEVICE_ID", "")  # auto-detected at runtime if empty

# ── App Communication ───────────────────────────────────────────────────────
APP_API_HOST: str = os.getenv("APP_API_HOST", "0.0.0.0")
APP_API_PORT: int = int(os.getenv("APP_API_PORT", "8765"))

# ── Pipeline ─────────────────────────────────────────────────────────────────
PREDICTION_INTERVAL: float = float(os.getenv("PREDICTION_INTERVAL", "5.0"))
MAX_RECORDING_SECONDS: int = int(os.getenv("MAX_RECORDING_SECONDS", "30"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
