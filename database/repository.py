"""
Repository Layer for CryingSense Database

Provides data access patterns and CRUD operations for all collections.
Follows the Repository pattern for clean separation of concerns.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from bson import ObjectId
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError, PyMongoError
from pymongo import ASCENDING, DESCENDING

from .config import get_collection
from .models import (
    CryClassification,
    AudioSession,
    DeviceRegistration,
    AudioFile,
    CryClass,
    DeviceSource,
)

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository with common CRUD operations."""

    def __init__(self, collection_name: str):
        self.collection_name = collection_name

    @property
    def collection(self) -> Collection:
        """Get the MongoDB collection."""
        return get_collection(self.collection_name)

    def find_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """Find a document by its ObjectId."""
        try:
            return self.collection.find_one({"_id": ObjectId(id)})
        except Exception as e:
            logger.error(f"Error finding document by ID: {e}")
            return None

    def delete_by_id(self, id: str) -> bool:
        """Delete a document by its ObjectId."""
        try:
            result = self.collection.delete_one({"_id": ObjectId(id)})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False

    def count(self, filter_query: Optional[Dict] = None) -> int:
        """Count documents matching a filter."""
        return self.collection.count_documents(filter_query or {})


class CryClassificationRepository(BaseRepository):
    """
    Repository for cry classification records.

    Handles all database operations for CryClassification documents.
    This is the primary repository for storing and querying classification results.
    """

    COLLECTION_NAME = "cry_classifications"

    def __init__(self):
        super().__init__(self.COLLECTION_NAME)
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """Create required indexes for optimal query performance."""
        try:
            # Time-based queries
            self.collection.create_index([("timestamp", DESCENDING)])
            self.collection.create_index([("created_at", DESCENDING)])

            # Device queries
            self.collection.create_index([("device_id", ASCENDING)])
            self.collection.create_index([("device_source", ASCENDING)])

            # Classification queries
            self.collection.create_index([("classification.predicted_class", ASCENDING)])

            # Compound index for common queries
            self.collection.create_index([
                ("device_id", ASCENDING),
                ("timestamp", DESCENDING)
            ])

            logger.info("CryClassification indexes created successfully")
        except PyMongoError as e:
            logger.warning(f"Error creating indexes: {e}")

    def create(self, classification: CryClassification) -> Optional[str]:
        """
        Insert a new classification record.

        Args:
            classification: CryClassification model instance.

        Returns:
            str: The inserted document's ID, or None if failed.
        """
        try:
            doc = classification.to_mongo_dict()
            doc["created_at"] = datetime.utcnow()
            doc["updated_at"] = datetime.utcnow()

            result = self.collection.insert_one(doc)
            logger.info(f"Created classification: {result.inserted_id}")
            return str(result.inserted_id)

        except PyMongoError as e:
            logger.error(f"Error creating classification: {e}")
            return None

    def find_recent(
        self,
        limit: int = 50,
        device_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find recent classifications.

        Args:
            limit: Maximum number of records to return.
            device_id: Optional filter by device.

        Returns:
            List of classification documents.
        """
        query = {}
        if device_id:
            query["device_id"] = device_id

        cursor = (
            self.collection
            .find(query)
            .sort("timestamp", DESCENDING)
            .limit(limit)
        )
        return list(cursor)

    def find_by_time_range(
        self,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        device_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find classifications within a time range.

        Args:
            start_time: Start of the time range.
            end_time: End of the time range (defaults to now).
            device_id: Optional filter by device.

        Returns:
            List of classification documents.
        """
        end_time = end_time or datetime.utcnow()

        query = {
            "timestamp": {
                "$gte": start_time,
                "$lte": end_time
            }
        }
        if device_id:
            query["device_id"] = device_id

        cursor = self.collection.find(query).sort("timestamp", DESCENDING)
        return list(cursor)

    def find_by_class(
        self,
        cry_class: CryClass,
        limit: int = 100,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Find classifications by predicted class.

        Args:
            cry_class: The CryClass to filter by.
            limit: Maximum number of records.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of classification documents.
        """
        query = {
            "classification.predicted_class": cry_class.value,
            "classification.confidence_score": {"$gte": min_confidence}
        }

        cursor = (
            self.collection
            .find(query)
            .sort("timestamp", DESCENDING)
            .limit(limit)
        )
        return list(cursor)

    def update_verification(
        self,
        id: str,
        verified_class: CryClass,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Update a classification with user verification.

        Args:
            id: Classification document ID.
            verified_class: User-verified classification.
            notes: Optional notes.

        Returns:
            bool: True if update was successful.
        """
        try:
            update_data = {
                "$set": {
                    "is_verified": True,
                    "verified_class": verified_class.value,
                    "updated_at": datetime.utcnow(),
                }
            }
            if notes:
                update_data["$set"]["notes"] = notes

            result = self.collection.update_one(
                {"_id": ObjectId(id)},
                update_data
            )
            return result.modified_count > 0

        except PyMongoError as e:
            logger.error(f"Error updating verification: {e}")
            return False

    def get_statistics(
        self,
        device_id: Optional[str] = None,
        days: int = 7,
    ) -> Dict[str, Any]:
        """
        Get classification statistics.

        Args:
            device_id: Optional filter by device.
            days: Number of days to analyze.

        Returns:
            Dictionary with statistics.
        """
        start_date = datetime.utcnow() - timedelta(days=days)

        match_stage = {"timestamp": {"$gte": start_date}}
        if device_id:
            match_stage["device_id"] = device_id

        pipeline = [
            {"$match": match_stage},
            {
                "$group": {
                    "_id": "$classification.predicted_class",
                    "count": {"$sum": 1},
                    "avg_confidence": {"$avg": "$classification.confidence_score"},
                }
            },
            {"$sort": {"count": -1}}
        ]

        results = list(self.collection.aggregate(pipeline))

        total = sum(r["count"] for r in results)
        return {
            "period_days": days,
            "total_classifications": total,
            "by_class": {
                r["_id"]: {
                    "count": r["count"],
                    "percentage": round(r["count"] / total * 100, 2) if total > 0 else 0,
                    "avg_confidence": round(r["avg_confidence"], 4),
                }
                for r in results
            },
        }

    def get_hourly_distribution(
        self,
        days: int = 7,
    ) -> List[Dict[str, Any]]:
        """
        Get hourly distribution of cries.

        Useful for understanding cry patterns throughout the day.

        Args:
            days: Number of days to analyze.

        Returns:
            List with hourly counts.
        """
        start_date = datetime.utcnow() - timedelta(days=days)

        match_stage = {"timestamp": {"$gte": start_date}}

        pipeline = [
            {"$match": match_stage},
            {
                "$group": {
                    "_id": {"$hour": "$timestamp"},
                    "count": {"$sum": 1},
                }
            },
            {"$sort": {"_id": 1}}
        ]

        return list(self.collection.aggregate(pipeline))


class AudioSessionRepository(BaseRepository):
    """Repository for audio monitoring sessions."""

    COLLECTION_NAME = "audio_sessions"

    def __init__(self):
        super().__init__(self.COLLECTION_NAME)
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """Create indexes for sessions."""
        try:
            self.collection.create_index([("session_id", ASCENDING)], unique=True)
            self.collection.create_index([("device_id", ASCENDING)])
            self.collection.create_index([("is_active", ASCENDING)])
            self.collection.create_index([("started_at", DESCENDING)])
        except PyMongoError as e:
            logger.warning(f"Error creating indexes: {e}")

    def create(self, session: AudioSession) -> Optional[str]:
        """Create a new audio session."""
        try:
            doc = session.model_dump(by_alias=True, exclude_none=True)
            if "_id" in doc and doc["_id"] is None:
                del doc["_id"]

            result = self.collection.insert_one(doc)
            return str(result.inserted_id)
        except DuplicateKeyError:
            logger.warning(f"Session already exists: {session.session_id}")
            return None
        except PyMongoError as e:
            logger.error(f"Error creating session: {e}")
            return None

    def find_active_sessions(self, device_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find all active sessions."""
        query = {"is_active": True}
        if device_id:
            query["device_id"] = device_id
        return list(self.collection.find(query))

    def end_session(self, session_id: str) -> bool:
        """End an active session."""
        try:
            result = self.collection.update_one(
                {"session_id": session_id},
                {
                    "$set": {
                        "is_active": False,
                        "ended_at": datetime.utcnow()
                    }
                }
            )
            return result.modified_count > 0
        except PyMongoError as e:
            logger.error(f"Error ending session: {e}")
            return False

    def add_classification_to_session(
        self,
        session_id: str,
        classification_id: str
    ) -> bool:
        """Add a classification reference to a session."""
        try:
            result = self.collection.update_one(
                {"session_id": session_id},
                {
                    "$push": {"classification_ids": classification_id},
                    "$inc": {"total_cries_detected": 1}
                }
            )
            return result.modified_count > 0
        except PyMongoError as e:
            logger.error(f"Error adding classification to session: {e}")
            return False


class DeviceRegistrationRepository(BaseRepository):
    """Repository for device registrations."""

    COLLECTION_NAME = "device_registrations"

    def __init__(self):
        super().__init__(self.COLLECTION_NAME)
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """Create indexes for devices."""
        try:
            self.collection.create_index([("device_id", ASCENDING)], unique=True)
            self.collection.create_index([("device_type", ASCENDING)])
            self.collection.create_index([("is_active", ASCENDING)])
        except PyMongoError as e:
            logger.warning(f"Error creating indexes: {e}")

    def register_device(self, device: DeviceRegistration) -> Optional[str]:
        """Register a new device or update existing."""
        try:
            doc = device.model_dump(by_alias=True, exclude_none=True)
            if "_id" in doc and doc["_id"] is None:
                del doc["_id"]

            result = self.collection.update_one(
                {"device_id": device.device_id},
                {"$set": doc},
                upsert=True
            )

            if result.upserted_id:
                return str(result.upserted_id)
            return device.device_id

        except PyMongoError as e:
            logger.error(f"Error registering device: {e}")
            return None

    def update_last_seen(self, device_id: str) -> bool:
        """Update the last seen timestamp for a device."""
        try:
            result = self.collection.update_one(
                {"device_id": device_id},
                {"$set": {"last_seen": datetime.utcnow()}}
            )
            return result.modified_count > 0
        except PyMongoError as e:
            logger.error(f"Error updating last seen: {e}")
            return False

    def find_by_type(self, device_type: DeviceSource) -> List[Dict[str, Any]]:
        """Find all devices of a specific type."""
        return list(self.collection.find({
            "device_type": device_type.value,
            "is_active": True
        }))

    def deactivate_device(self, device_id: str) -> bool:
        """Deactivate a device."""
        try:
            result = self.collection.update_one(
                {"device_id": device_id},
                {"$set": {"is_active": False}}
            )
            return result.modified_count > 0
        except PyMongoError as e:
            logger.error(f"Error deactivating device: {e}")
            return False


class AudioFileRepository(BaseRepository):
    """
    Repository for audio file storage.
    
    Handles storage and retrieval of audio files with their metadata.
    Supports multiple storage methods: filesystem paths, cloud URLs, base64, GridFS.
    """
    
    COLLECTION_NAME = "audio_files"
    
    def __init__(self):
        super().__init__(self.COLLECTION_NAME)
        self._ensure_indexes()
    
    def _ensure_indexes(self) -> None:
        """Create required indexes for optimal query performance."""
        try:
            # File identification
            self.collection.create_index([("file_id", ASCENDING)], unique=True)
            
            # Reference queries
            self.collection.create_index([("device_id", ASCENDING)])
            self.collection.create_index([("session_id", ASCENDING)])
            self.collection.create_index([("classification_id", ASCENDING)])
            
            # Time-based queries
            self.collection.create_index([("uploaded_at", DESCENDING)])
            
            # TTL index for auto-deletion (30 days default)
            self.collection.create_index(
                [("expires_at", ASCENDING)],
                expireAfterSeconds=0
            )
            
            logger.info("AudioFile indexes created successfully")
        except PyMongoError as e:
            logger.warning(f"Error creating indexes: {e}")
    
    def create(self, audio_file: AudioFile) -> Optional[str]:
        """
        Store a new audio file record.
        
        Args:
            audio_file: AudioFile model instance.
            
        Returns:
            str: File ID if successful, None otherwise.
        """
        try:
            doc = audio_file.to_mongo_dict()
            result = self.collection.insert_one(doc)
            return audio_file.file_id
        except DuplicateKeyError:
            logger.error(f"Audio file with file_id {audio_file.file_id} already exists")
            return None
        except PyMongoError as e:
            logger.error(f"Error storing audio file: {e}")
            return None
    
    def find_by_file_id(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Find an audio file by its file_id."""
        return self.collection.find_one({"file_id": file_id})
    
    def find_by_device(
        self,
        device_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Find all audio files from a specific device."""
        return list(self.collection.find(
            {"device_id": device_id}
        ).sort("uploaded_at", DESCENDING).limit(limit))
    
    def find_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Find all audio files from a specific session."""
        return list(self.collection.find(
            {"session_id": session_id}
        ).sort("uploaded_at", ASCENDING))
    
    def find_by_classification(self, classification_id: str) -> Optional[Dict[str, Any]]:
        """Find the audio file for a specific classification."""
        return self.collection.find_one({"classification_id": classification_id})
    
    def delete_by_file_id(self, file_id: str) -> bool:
        """Delete an audio file by its file_id."""
        try:
            result = self.collection.delete_one({"file_id": file_id})
            return result.deleted_count > 0
        except PyMongoError as e:
            logger.error(f"Error deleting audio file: {e}")
            return False
    
    def set_expiry(self, file_id: str, days: int = 30) -> bool:
        """Set expiry date for an audio file (for TTL auto-deletion)."""
        try:
            expires_at = datetime.utcnow() + timedelta(days=days)
            result = self.collection.update_one(
                {"file_id": file_id},
                {"$set": {"expires_at": expires_at}}
            )
            return result.modified_count > 0
        except PyMongoError as e:
            logger.error(f"Error setting expiry: {e}")
            return False
