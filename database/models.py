"""
Data Models for CryingSense Database

Defines Pydantic models for validation and MongoDB document structures.
These models represent the core data entities for infant cry classification.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from bson import ObjectId


class CryClass(str, Enum):
    """Classification categories for infant cries."""
    HUNGER = "hunger"
    TIRED = "tired"
    DISCOMFORT = "discomfort"
    BELLY_PAIN = "belly_pain"
    BURP = "burp"


class DeviceSource(str, Enum):
    """Source device for audio capture."""
    ESP32 = "esp32"
    ANDROID = "android"


class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic models."""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v, handler):
        if isinstance(v, ObjectId):
            return v
        if isinstance(v, str) and ObjectId.is_valid(v):
            return ObjectId(v)
        raise ValueError("Invalid ObjectId")
    
    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        return {"type": "string"}


class AudioMetadata(BaseModel):
    """Metadata about the captured audio."""
    
    sample_rate: int = Field(default=44100, description="Audio sample rate in Hz")
    duration_seconds: float = Field(..., description="Duration of audio in seconds")
    channels: int = Field(default=1, description="Number of audio channels")
    bit_depth: int = Field(default=16, description="Bit depth of audio")
    file_size_bytes: Optional[int] = Field(default=None, description="Size of audio file if stored")
    
    class Config:
        json_schema_extra = {
            "example": {
                "sample_rate": 44100,
                "duration_seconds": 3.5,
                "channels": 1,
                "bit_depth": 16,
                "file_size_bytes": 308700
            }
        }


class MFCCFeatures(BaseModel):
    """MFCC feature extraction results."""
    
    n_mfcc: int = Field(default=13, description="Number of MFCC coefficients")
    features: List[List[float]] = Field(..., description="MFCC feature matrix")
    frame_length: int = Field(default=2048, description="Frame length for STFT")
    hop_length: int = Field(default=512, description="Hop length for STFT")


class ClassificationResult(BaseModel):
    """Model classification output."""
    
    predicted_class: CryClass = Field(..., description="Predicted cry classification")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Model confidence (0-1)")
    all_probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Probability scores for all classes"
    )
    model_version: str = Field(default="1.0.0", description="Version of the model used")
    
    @field_validator("confidence_score")
    @classmethod
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence score must be between 0 and 1")
        return round(v, 4)
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_class": "hunger",
                "confidence_score": 0.9234,
                "all_probabilities": {
                    "hunger": 0.9234,
                    "tired": 0.0421,
                    "discomfort": 0.0198,
                    "belly_pain": 0.0095,
                    "burp": 0.0052
                },
                "model_version": "1.0.0"
            }
        }


class CryClassification(BaseModel):
    """
    Main document model for cry classification records.
    
    This is the primary collection for storing classification results
    from both ESP32/RaspberryPi and Android devices.
    """
    
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    
    # Timestamps
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the cry was recorded")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Source information
    device_source: DeviceSource = Field(..., description="Device that captured the audio")
    device_id: Optional[str] = Field(default=None, description="Unique identifier of the device")
    session_id: Optional[str] = Field(default=None, description="Session identifier for grouping")
    
    # Audio metadata
    audio_metadata: AudioMetadata = Field(..., description="Audio capture metadata")
    audio_file_path: Optional[str] = Field(default=None, description="Path to stored audio file (if any)")
    
    # Features
    mfcc_features: Optional[MFCCFeatures] = Field(default=None, description="Extracted MFCC features")
    
    # Classification results
    classification: ClassificationResult = Field(..., description="Model classification output")
    
    # Additional context
    notes: Optional[str] = Field(default=None, max_length=500, description="Optional notes")
    is_verified: bool = Field(default=False, description="Whether classification was verified by user")
    verified_class: Optional[CryClass] = Field(default=None, description="User-verified classification")
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        json_schema_extra = {
            "example": {
                "timestamp": "2024-01-15T10:30:00Z",
                "device_source": "esp32",
                "device_id": "ESP32-001",
                "audio_metadata": {
                    "sample_rate": 44100,
                    "duration_seconds": 3.5,
                    "channels": 1,
                    "bit_depth": 16
                },
                "classification": {
                    "predicted_class": "hunger",
                    "confidence_score": 0.9234,
                    "model_version": "1.0.0"
                }
            }
        }
    
    def to_mongo_dict(self) -> Dict[str, Any]:
        """Convert model to MongoDB document format."""
        data = self.model_dump(by_alias=True, exclude_none=True)
        if "_id" in data and data["_id"] is None:
            del data["_id"]
        return data


class AudioSession(BaseModel):
    """
    Session model for grouping continuous audio monitoring.
    
    Used when ESP32 runs continuously and groups multiple
    cry events within a monitoring session.
    """
    
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    
    # Session info
    session_id: str = Field(..., description="Unique session identifier")
    device_id: str = Field(..., description="Device that initiated the session")
    device_source: DeviceSource = Field(..., description="Type of device")
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = Field(default=None)
    is_active: bool = Field(default=True)
    
    # Stats
    total_cries_detected: int = Field(default=0)
    classification_ids: List[str] = Field(default_factory=list, description="References to classifications")
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class DeviceRegistration(BaseModel):
    """
    Device registration for ESP32 and mobile devices.
    
    Tracks registered devices in the system.
    """
    
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    
    device_id: str = Field(..., description="Unique device identifier")
    device_type: DeviceSource = Field(..., description="Type of device")
    
    # ESP32 specific
    mac_address: Optional[str] = Field(default=None, description="MAC address for ESP32")
    firmware_version: Optional[str] = Field(default=None)
    
    # Mobile specific
    platform: Optional[str] = Field(default=None, description="android/ios")
    app_version: Optional[str] = Field(default=None)
    
    # Status
    is_active: bool = Field(default=True)
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    
    # Timestamps
    registered_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class AudioFile(BaseModel):
    """
    Audio file storage model.
    
    Stores audio file metadata and base64-encoded audio data or file paths.
    For large files, consider using GridFS instead.
    """
    
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    
    # File identification
    file_id: str = Field(..., description="Unique file identifier")
    original_filename: Optional[str] = Field(default=None, description="Original filename")
    
    # Storage options (use ONE of these)
    file_path: Optional[str] = Field(default=None, description="Path to file on filesystem")
    file_url: Optional[str] = Field(default=None, description="URL to cloud storage (S3, Azure Blob)")
    audio_data_base64: Optional[str] = Field(default=None, description="Base64 encoded audio (small files only)")
    gridfs_id: Optional[str] = Field(default=None, description="GridFS file ID for large files")
    
    # Metadata
    audio_metadata: AudioMetadata = Field(..., description="Audio file metadata")
    mime_type: str = Field(default="audio/wav", description="MIME type of audio file")
    
    # References
    device_id: Optional[str] = Field(default=None, description="Device that captured audio")
    session_id: Optional[str] = Field(default=None, description="Related session")
    classification_id: Optional[str] = Field(default=None, description="Related classification")
    
    # Timestamps
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(default=None, description="Auto-delete after this date")
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        json_schema_extra = {
            "example": {
                "file_id": "audio_20240115_103000",
                "original_filename": "cry_sample.wav",
                "file_path": "/data/audio/cry_sample.wav",
                "audio_metadata": {
                    "sample_rate": 44100,
                    "duration_seconds": 3.5,
                    "channels": 1,
                    "bit_depth": 16,
                    "file_size_bytes": 308700
                },
                "mime_type": "audio/wav",
                "device_id": "ESP32-001"
            }
        }
    
    def to_mongo_dict(self) -> Dict[str, Any]:
        """Convert model to MongoDB document format."""
        data = self.model_dump(by_alias=True, exclude_none=True)
        if "_id" in data and data["_id"] is None:
            del data["_id"]
        return data
