"""
Database Handler for CryingSense Trail Run

Handles all MongoDB Atlas operations for storing:
- Audio files
- Audio sessions
- Cry classifications
- Device registrations
"""

import os
import sys
import logging
import uuid
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
import numpy as np
from io import BytesIO
import wave

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from . import config
except ImportError:
    import config

logger = logging.getLogger(__name__)


class DatabaseHandler:
    """
    Handles all database operations for the trail run system.
    
    Manages:
    - Device registration
    - Session management
    - Classification storage
    - Audio file storage
    """
    
    def __init__(self, mongo_uri: str = None, database_name: str = None):
        """
        Initialize database handler.
        
        Args:
            mongo_uri: MongoDB connection URI (default from config)
            database_name: Database name (default from config)
        """
        self.mongo_uri = mongo_uri or config.MONGO_URI
        self.database_name = database_name or config.MONGO_DATABASE
        
        self._client = None
        self._db = None
        self._connected = False
        
        # Current session
        self._session_id: Optional[str] = None
        self._device_id: Optional[str] = None
        
        # Statistics
        self._classifications_saved = 0
        self._audio_files_saved = 0
    
    def connect(self) -> bool:
        """
        Connect to MongoDB Atlas.
        
        Returns:
            True if connection successful
        """
        try:
            from pymongo import MongoClient
            from pymongo.server_api import ServerApi
            
            self._client = MongoClient(
                self.mongo_uri,
                server_api=ServerApi('1'),
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
            )
            
            # Test connection
            self._client.admin.command('ping')
            
            self._db = self._client[self.database_name]
            self._connected = True
            
            logger.info(f"Connected to MongoDB: {self.database_name}")
            return True
            
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            self._connected = False
            return False
    
    def disconnect(self) -> None:
        """Disconnect from MongoDB."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            self._connected = False
            logger.info("Disconnected from MongoDB")
    
    def register_device(self, device_id: str = None, device_type: str = "esp32",
                       mac_address: str = None, firmware_version: str = "1.0.0",
                       metadata: Dict = None) -> Optional[str]:
        """
        Register or update device in database.
        
        Args:
            device_id: Device identifier (generated if not provided)
            device_type: Device type (esp32, android, etc.)
            mac_address: Device MAC address
            firmware_version: Firmware version
            metadata: Additional device metadata
        
        Returns:
            Device ID if successful
        """
        if not self._connected:
            logger.warning("Not connected to database")
            return None
        
        try:
            # Generate device ID if not provided
            if not device_id:
                if mac_address:
                    device_id = hashlib.md5(mac_address.encode()).hexdigest()[:12]
                else:
                    device_id = str(uuid.uuid4())[:12]
            
            self._device_id = device_id
            
            collection = self._db[config.COLLECTION_DEVICE_REGISTRATIONS]
            
            # Check if device exists
            existing = collection.find_one({"device_id": device_id})
            
            doc = {
                "device_id": device_id,
                "device_type": device_type,
                "mac_address": mac_address,
                "firmware_version": firmware_version,
                "metadata": metadata or {},
                "updated_at": datetime.utcnow(),
            }
            
            if existing:
                # Update existing
                doc["last_seen"] = datetime.utcnow()
                collection.update_one(
                    {"device_id": device_id},
                    {"$set": doc}
                )
                logger.info(f"Device updated: {device_id}")
            else:
                # Create new
                doc["created_at"] = datetime.utcnow()
                doc["first_seen"] = datetime.utcnow()
                doc["last_seen"] = datetime.utcnow()
                doc["is_active"] = True
                collection.insert_one(doc)
                logger.info(f"Device registered: {device_id}")
            
            return device_id
            
        except Exception as e:
            logger.error(f"Device registration failed: {e}")
            return None
    
    def start_session(self, device_id: str = None) -> Optional[str]:
        """
        Start a new monitoring session.
        
        Args:
            device_id: Device ID (uses registered device if not provided)
        
        Returns:
            Session ID if successful
        """
        if not self._connected:
            logger.warning("Not connected to database")
            return None
        
        device_id = device_id or self._device_id
        if not device_id:
            logger.error("No device ID available")
            return None
        
        try:
            session_id = str(uuid.uuid4())
            self._session_id = session_id
            
            collection = self._db[config.COLLECTION_AUDIO_SESSIONS]
            
            doc = {
                "session_id": session_id,
                "device_id": device_id,
                "device_type": config.DEVICE_TYPE,
                "start_time": datetime.utcnow(),
                "end_time": None,
                "status": "active",
                "classification_count": 0,
                "audio_file_count": 0,
                "created_at": datetime.utcnow(),
            }
            
            collection.insert_one(doc)
            logger.info(f"Session started: {session_id}")
            
            return session_id
            
        except Exception as e:
            logger.error(f"Session start failed: {e}")
            return None
    
    def end_session(self, session_id: str = None) -> bool:
        """
        End a monitoring session.
        
        Args:
            session_id: Session ID (uses current session if not provided)
        
        Returns:
            True if successful
        """
        if not self._connected:
            return False
        
        session_id = session_id or self._session_id
        if not session_id:
            return False
        
        try:
            collection = self._db[config.COLLECTION_AUDIO_SESSIONS]
            
            result = collection.update_one(
                {"session_id": session_id},
                {
                    "$set": {
                        "end_time": datetime.utcnow(),
                        "status": "completed",
                        "updated_at": datetime.utcnow(),
                    }
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"Session ended: {session_id}")
                if session_id == self._session_id:
                    self._session_id = None
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Session end failed: {e}")
            return False
    
    def save_classification(self, predicted_class: str, confidence: float,
                           all_probabilities: Dict[str, float],
                           duration_seconds: float,
                           device_id: str = None, session_id: str = None,
                           audio_file_id: str = None) -> Optional[str]:
        """
        Save a cry classification result.
        
        Args:
            predicted_class: Predicted cry class
            confidence: Confidence score (0-1)
            all_probabilities: All class probabilities
            duration_seconds: Audio duration
            device_id: Device ID
            session_id: Session ID
            audio_file_id: Associated audio file ID
        
        Returns:
            Classification document ID if successful
        """
        if not self._connected:
            logger.warning("Not connected to database")
            return None
        
        device_id = device_id or self._device_id
        session_id = session_id or self._session_id
        
        try:
            collection = self._db[config.COLLECTION_CRY_CLASSIFICATIONS]
            
            doc = {
                "timestamp": datetime.utcnow(),
                "device_type": config.DEVICE_TYPE,
                "device_id": device_id,
                "session_id": session_id,
                "audio_metadata": {
                    "sample_rate": config.SAMPLE_RATE,
                    "duration_seconds": duration_seconds,
                    "channels": config.CHANNELS,
                    "bit_depth": config.BIT_DEPTH,
                },
                "classification": {
                    "predicted_class": predicted_class,
                    "confidence_score": round(confidence, 4),
                    "all_probabilities": {
                        k: round(v, 4) for k, v in all_probabilities.items()
                    },
                    "model_version": "1.0.0",
                },
                "audio_file_id": audio_file_id,
                "is_verified": False,
                "verified_class": None,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }
            
            result = collection.insert_one(doc)
            doc_id = str(result.inserted_id)
            
            self._classifications_saved += 1
            
            # Update session count
            if session_id:
                self._db[config.COLLECTION_AUDIO_SESSIONS].update_one(
                    {"session_id": session_id},
                    {"$inc": {"classification_count": 1}}
                )
            
            logger.info(f"Classification saved: {predicted_class} ({confidence:.1%})")
            return doc_id
            
        except Exception as e:
            logger.error(f"Classification save failed: {e}")
            return None
    
    def save_audio_file(self, audio: np.ndarray, filename: str = None,
                       device_id: str = None, session_id: str = None,
                       classification_id: str = None,
                       classification_label: str = None) -> Optional[str]:
        """
        Save audio file to database.
        
        Args:
            audio: Audio samples (int16)
            filename: Custom filename (generated if not provided)
            device_id: Device ID
            session_id: Session ID
            classification_id: Associated classification ID
            classification_label: Classification label for filename
        
        Returns:
            Audio file document ID if successful
        """
        if not self._connected:
            logger.warning("Not connected to database")
            return None
        
        device_id = device_id or self._device_id
        session_id = session_id or self._session_id
        
        try:
            # Generate filename
            if not filename:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                label = classification_label or "cry"
                filename = f"{label}_{timestamp}.wav"
            
            # Convert audio to WAV bytes
            wav_bytes = self._audio_to_wav_bytes(audio)
            
            collection = self._db[config.COLLECTION_AUDIO_FILES]
            
            doc = {
                "filename": filename,
                "device_id": device_id,
                "session_id": session_id,
                "classification_id": classification_id,
                "file_data": wav_bytes,
                "file_size_bytes": len(wav_bytes),
                "audio_metadata": {
                    "sample_rate": config.SAMPLE_RATE,
                    "duration_seconds": len(audio) / config.SAMPLE_RATE,
                    "channels": config.CHANNELS,
                    "bit_depth": config.BIT_DEPTH,
                    "format": "wav",
                },
                "created_at": datetime.utcnow(),
            }
            
            result = collection.insert_one(doc)
            doc_id = str(result.inserted_id)
            
            self._audio_files_saved += 1
            
            # Update session count
            if session_id:
                self._db[config.COLLECTION_AUDIO_SESSIONS].update_one(
                    {"session_id": session_id},
                    {"$inc": {"audio_file_count": 1}}
                )
            
            # Also save locally
            self._save_audio_locally(audio, filename, classification_label)
            
            logger.info(f"Audio file saved: {filename} ({len(wav_bytes)} bytes)")
            return doc_id
            
        except Exception as e:
            logger.error(f"Audio file save failed: {e}")
            return None
    
    def _audio_to_wav_bytes(self, audio: np.ndarray) -> bytes:
        """Convert numpy audio array to WAV bytes."""
        buffer = BytesIO()
        
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(config.CHANNELS)
            wav.setsampwidth(config.BIT_DEPTH // 8)
            wav.setframerate(config.SAMPLE_RATE)
            wav.writeframes(audio.astype(np.int16).tobytes())
        
        return buffer.getvalue()
    
    def _save_audio_locally(self, audio: np.ndarray, filename: str, 
                           classification_label: str = None) -> Optional[Path]:
        """Save audio file locally as backup."""
        try:
            # Create subdirectory for label
            if classification_label:
                save_dir = config.RECORDINGS_DIR / classification_label
            else:
                save_dir = config.RECORDINGS_DIR
            
            save_dir.mkdir(parents=True, exist_ok=True)
            filepath = save_dir / filename
            
            with wave.open(str(filepath), 'wb') as wav:
                wav.setnchannels(config.CHANNELS)
                wav.setsampwidth(config.BIT_DEPTH // 8)
                wav.setframerate(config.SAMPLE_RATE)
                wav.writeframes(audio.astype(np.int16).tobytes())
            
            return filepath
            
        except Exception as e:
            logger.error(f"Local save failed: {e}")
            return None
    
    def save_cry_event(self, classification_result: Dict, audio: np.ndarray) -> Dict[str, str]:
        """
        Save complete cry event (classification + audio).
        
        Args:
            classification_result: Result dict from classifier
            audio: Audio samples
        
        Returns:
            Dict with saved document IDs
        """
        result = {}
        
        # Save audio file first
        audio_id = self.save_audio_file(
            audio=audio,
            classification_label=classification_result['class'],
        )
        result['audio_file_id'] = audio_id
        
        # Save classification with audio reference
        classification_id = self.save_classification(
            predicted_class=classification_result['class'],
            confidence=classification_result['confidence'],
            all_probabilities=classification_result['probabilities'],
            duration_seconds=len(audio) / config.SAMPLE_RATE,
            audio_file_id=audio_id,
        )
        result['classification_id'] = classification_id
        
        return result
    
    @property
    def is_connected(self) -> bool:
        """Whether connected to database."""
        return self._connected
    
    @property
    def session_id(self) -> Optional[str]:
        """Current session ID."""
        return self._session_id
    
    @property
    def device_id(self) -> Optional[str]:
        """Current device ID."""
        return self._device_id
    
    def get_stats(self) -> Dict:
        """Get handler statistics."""
        return {
            "connected": self._connected,
            "device_id": self._device_id,
            "session_id": self._session_id,
            "classifications_saved": self._classifications_saved,
            "audio_files_saved": self._audio_files_saved,
        }
