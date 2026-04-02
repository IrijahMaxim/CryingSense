"""Firebase persistence layer for the Raspberry Pi pipeline."""

import os
import uuid
import base64
import logging
from datetime import datetime
from typing import Optional, Dict, Any

import firebase_admin
from firebase_admin import credentials, firestore, storage

from config import (
    FIREBASE_PROJECT_ID,
    FIREBASE_CREDENTIALS_PATH,
    FIREBASE_STORAGE_BUCKET,
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
    """Thin Firebase facade used by the pipeline."""

    def __init__(self):
        self._app = None
        self._client = None
        self._db = None
        self._bucket = None

    # ── connection ───────────────────────────────────────────────────────

    def connect(self):
        if self._db is not None:
            return

        init_args: Dict[str, Any] = {}
        if FIREBASE_PROJECT_ID:
            init_args["projectId"] = FIREBASE_PROJECT_ID
        if FIREBASE_STORAGE_BUCKET:
            init_args["storageBucket"] = FIREBASE_STORAGE_BUCKET

        if FIREBASE_CREDENTIALS_PATH:
            cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
            self._app = firebase_admin.initialize_app(cred, init_args or None)
        else:
            self._app = firebase_admin.initialize_app(options=init_args or None)

        self._client = firestore.client(self._app)
        self._db = self._client
        if FIREBASE_STORAGE_BUCKET:
            self._bucket = storage.bucket(app=self._app)

        log.info("Firebase connected  [project=%s]", FIREBASE_PROJECT_ID or "default")

    def close(self):
        self._bucket = None
        self._client = None
        self._db = None
        if self._app is not None:
            try:
                firebase_admin.delete_app(self._app)
            except ValueError:
                pass
            self._app = None

    # ── device registration ──────────────────────────────────────────────

    def register_device(self, device_id: str, mac_address: str = "") -> str:
        col = self._db.collection(COL_DEVICE_REGISTRATIONS)
        now = datetime.utcnow()
        ref = col.document(device_id)
        existing = ref.get()
        data = {
            "device_type": DEVICE_TYPE,
            "mac_address": mac_address,
            "is_active": True,
            "last_seen": now,
        }
        if existing.exists:
            ref.set(data, merge=True)
        else:
            data.update({"device_id": device_id, "registered_at": now})
            ref.set(data, merge=True)
        log.info("Device registered  [id=%s]", device_id)
        return device_id

    def heartbeat(self, device_id: str):
        self._db.collection(COL_DEVICE_REGISTRATIONS).document(device_id).set(
            {"last_seen": datetime.utcnow()}, merge=True
        )

    # ── sessions ─────────────────────────────────────────────────────────

    def start_session(self, device_id: str) -> str:
        session_id = str(uuid.uuid4())
        self._db.collection(COL_AUDIO_SESSIONS).document(session_id).set(
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
        self._db.collection(COL_AUDIO_SESSIONS).document(session_id).set(
            {"ended_at": datetime.utcnow(), "is_active": False}, merge=True
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
        """Store audio metadata and optionally upload WAV to Firebase Storage."""
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

            if self._bucket is not None and os.path.isfile(wav_path):
                blob_path = f"audio/{device_id}/{session_id}/{file_id}.wav"
                blob = self._bucket.blob(blob_path)
                blob.upload_from_filename(wav_path, content_type="audio/wav")
                doc["storage_path"] = blob_path
            elif 0 < file_size < 1_048_576:
                with open(wav_path, "rb") as f:
                    doc["audio_data_base64"] = base64.b64encode(f.read()).decode()

            self._db.collection(COL_AUDIO_FILES).document(file_id).set(doc)
            log.info("Audio saved  [file_id=%s  size=%d]", file_id, file_size)
            return file_id

        except Exception as exc:
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
            cid = str(uuid.uuid4())
            doc = {
                "id": cid,
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
            self._db.collection(COL_CRY_CLASSIFICATIONS).document(cid).set(doc)

            # Link classification to session
            session_ref = self._db.collection(COL_AUDIO_SESSIONS).document(session_id)
            snap = session_ref.get()
            data = snap.to_dict() if snap.exists else {}
            ids = list(data.get("classification_ids", []))
            ids.append(cid)
            total = int(data.get("total_cries_detected", 0)) + 1
            session_ref.set(
                {
                    "classification_ids": ids,
                    "total_cries_detected": total,
                    "updated_at": datetime.utcnow(),
                },
                merge=True,
            )
            log.info("Classification saved  [id=%s  class=%s]", cid, result["prediction"])
            return cid

        except Exception as exc:
            log.error("Failed to save classification: %s", exc)
            return None
