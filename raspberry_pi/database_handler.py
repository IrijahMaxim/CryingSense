"""
Database Handler — Raspberry Pi 3B+ Pipeline

Lightweight MongoDB client that writes to four collections:
  1. audio_files            — raw .wav binary (GridFS-style metadata)
  2. audio_sessions         — per-device session logs
  3. cry_classifications    — predicted cry + probabilities
  4. device_registrations   — device identity / heartbeat

Foreign-key relationships (stored as string references):
  - audio_files         → device_registrations.device_id
  - audio_sessions      → device_registrations.device_id
  - cry_classifications → audio_files.file_id,
                           audio_sessions.session_id,
                           device_registrations.device_id

Pool sizes are capped for 1 GB RAM (see config.py).
"""

import os
import uuid
import base64
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import PyMongoError

from config import (
    MONGO_URI,
    MONGO_DATABASE,
    MONGO_MAX_POOL,
    MONGO_MIN_POOL,
    MONGO_TIMEOUT_MS,
    COL_AUDIO_FILES,
    COL_AUDIO_SESSIONS,
    COL_CRY_CLASSIFICATIONS,
    COL_DEVICE_REGISTRATIONS,
    DEVICE_TYPE,
    SAMPLE_RATE,
    CHANNELS,
    BIT_DEPTH,
)

log = logging.getLogger(__name__)


class DatabaseHandler:
    """Thin MongoDB facade tuned for RPi 3B+."""

    def __init__(self):
        self._client: Optional[MongoClient] = None
        self._db = None

    # ── connection ───────────────────────────────────────────────────────

    def connect(self):
        if self._client is not None:
            return
        self._client = MongoClient(
            MONGO_URI,
            maxPoolSize=MONGO_MAX_POOL,
            minPoolSize=MONGO_MIN_POOL,
            connectTimeoutMS=MONGO_TIMEOUT_MS,
            serverSelectionTimeoutMS=MONGO_TIMEOUT_MS,
        )
        self._db = self._client[MONGO_DATABASE]
        self._ensure_indexes()
        log.info("MongoDB connected  [db=%s]", MONGO_DATABASE)

    def close(self):
        if self._client:
            self._client.close()
            self._client = None
            self._db = None

    # ── device registration ──────────────────────────────────────────────

    def register_device(self, device_id: str, mac_address: str = "") -> str:
        col = self._db[COL_DEVICE_REGISTRATIONS]
        now = datetime.utcnow()
        col.update_one(
            {"device_id": device_id},
            {
                "$set": {
                    "device_type": DEVICE_TYPE,
                    "mac_address": mac_address,
                    "is_active": True,
                    "last_seen": now,
                },
                "$setOnInsert": {
                    "device_id": device_id,
                    "registered_at": now,
                },
            },
            upsert=True,
        )
        log.info("Device registered  [id=%s]", device_id)
        return device_id

    def heartbeat(self, device_id: str):
        self._db[COL_DEVICE_REGISTRATIONS].update_one(
            {"device_id": device_id},
            {"$set": {"last_seen": datetime.utcnow()}},
        )

    # ── sessions ─────────────────────────────────────────────────────────

    def start_session(self, device_id: str) -> str:
        session_id = str(uuid.uuid4())
        self._db[COL_AUDIO_SESSIONS].insert_one(
            {
                "session_id": session_id,
                "device_id": device_id,
                "device_source": DEVICE_TYPE,
                "started_at": datetime.utcnow(),
                "ended_at": None,
                "is_active": True,
                "total_cries_detected": 0,
                "classification_ids": [],
            }
        )
        log.info("Session started  [session=%s]", session_id)
        return session_id

    def end_session(self, session_id: str):
        self._db[COL_AUDIO_SESSIONS].update_one(
            {"session_id": session_id},
            {"$set": {"ended_at": datetime.utcnow(), "is_active": False}},
        )
        log.info("Session ended  [session=%s]", session_id)

    # ── audio file ───────────────────────────────────────────────────────

    def save_audio_file(
        self,
        wav_path: str,
        device_id: str,
        session_id: str,
        duration_seconds: float,
        classification_id: Optional[str] = None,
    ) -> Optional[str]:
        """Store audio file metadata (+ optional base64 payload for small files)."""
        try:
            file_id = f"audio_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            file_size = os.path.getsize(wav_path) if os.path.isfile(wav_path) else 0

            doc: Dict[str, Any] = {
                "file_id": file_id,
                "original_filename": os.path.basename(wav_path),
                "file_path": wav_path,
                "audio_metadata": {
                    "sample_rate": SAMPLE_RATE,
                    "duration_seconds": round(duration_seconds, 3),
                    "channels": CHANNELS,
                    "bit_depth": BIT_DEPTH,
                    "file_size_bytes": file_size,
                },
                "mime_type": "audio/wav",
                "device_id": device_id,
                "session_id": session_id,
                "classification_id": classification_id,
                "uploaded_at": datetime.utcnow(),
            }

            # Embed small files (< 1 MB) as base64 for cloud portability
            if 0 < file_size < 1_048_576:
                with open(wav_path, "rb") as f:
                    doc["audio_data_base64"] = base64.b64encode(f.read()).decode()

            self._db[COL_AUDIO_FILES].insert_one(doc)
            log.info("Audio saved  [file_id=%s  size=%d]", file_id, file_size)
            return file_id

        except (PyMongoError, OSError) as exc:
            log.error("Failed to save audio: %s", exc)
            return None

    # ── cry classification ───────────────────────────────────────────────

    def save_classification(
        self,
        result: dict,
        device_id: str,
        session_id: str,
        audio_file_id: Optional[str] = None,
        model_version: str = "1.0.0",
    ) -> Optional[str]:
        """
        Persist a prediction result dict to cry_classifications.

        Parameters
        ----------
        result : dict from Predictor.predict()
        """
        try:
            doc = {
                "timestamp": datetime.utcnow(),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "device_source": DEVICE_TYPE,
                "device_id": device_id,
                "session_id": session_id,
                "audio_file_id": audio_file_id,
                "classification": {
                    "predicted_class": result["prediction"],
                    "confidence_score": result["confidence"],
                    "all_probabilities": result.get("probabilities", {}),
                    "model_version": model_version,
                },
                "is_verified": False,
                "verified_class": None,
            }
            inserted = self._db[COL_CRY_CLASSIFICATIONS].insert_one(doc)
            cid = str(inserted.inserted_id)

            # Link classification to session
            self._db[COL_AUDIO_SESSIONS].update_one(
                {"session_id": session_id},
                {
                    "$push": {"classification_ids": cid},
                    "$inc": {"total_cries_detected": 1},
                },
            )
            log.info("Classification saved  [id=%s  class=%s]", cid, result["prediction"])
            return cid

        except PyMongoError as exc:
            log.error("Failed to save classification: %s", exc)
            return None

    # ── indexes ──────────────────────────────────────────────────────────

    def _ensure_indexes(self):
        try:
            cls_col = self._db[COL_CRY_CLASSIFICATIONS]
            cls_col.create_index([("timestamp", DESCENDING)])
            cls_col.create_index([("device_id", ASCENDING), ("timestamp", DESCENDING)])

            self._db[COL_AUDIO_SESSIONS].create_index([("session_id", ASCENDING)], unique=True)
            self._db[COL_AUDIO_SESSIONS].create_index([("device_id", ASCENDING)])

            self._db[COL_AUDIO_FILES].create_index([("file_id", ASCENDING)], unique=True)

            self._db[COL_DEVICE_REGISTRATIONS].create_index(
                [("device_id", ASCENDING)], unique=True
            )
        except PyMongoError as exc:
            log.warning("Index creation issue: %s", exc)
