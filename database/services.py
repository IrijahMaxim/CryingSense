"""
Database Service Layer for CryingSense

High-level service functions that coordinate database operations
and provide a clean API for the rest of the application.
"""

import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

from .models import (
    CryClassification,
    AudioSession,
    DeviceRegistration,
    AudioMetadata,
    ClassificationResult,
    MFCCFeatures,
    CryClass,
    DeviceSource,
)
from .repository import (
    CryClassificationRepository,
    AudioSessionRepository,
    DeviceRegistrationRepository,
)

logger = logging.getLogger(__name__)


class CryClassificationService:
    """
    Service for handling cry classification data.
    
    This is the main interface for storing and retrieving
    classification results from the Raspberry Pi and mobile devices.
    """
    
    def __init__(self):
        self.classification_repo = CryClassificationRepository()
        self.session_repo = AudioSessionRepository()
    
    def save_classification(
        self,
        predicted_class: str,
        confidence_score: float,
        device_source: str,
        duration_seconds: float,
        sample_rate: int = 44100,
        device_id: Optional[str] = None,
        session_id: Optional[str] = None,
        all_probabilities: Optional[Dict[str, float]] = None,
        mfcc_features: Optional[List[List[float]]] = None,
        model_version: str = "1.0.0",
    ) -> Optional[str]:
        """
        Save a new classification result.
        
        This is the primary method called by the Raspberry Pi
        after performing inference on captured audio.
        
        Args:
            predicted_class: The predicted cry class (hunger, tired, etc.)
            confidence_score: Model confidence score (0-1)
            device_source: Source device ('esp32' or 'android')
            duration_seconds: Duration of the audio clip
            sample_rate: Audio sample rate
            device_id: Optional device identifier
            session_id: Optional session ID for grouping
            all_probabilities: Optional dict of all class probabilities
            mfcc_features: Optional MFCC feature matrix
            model_version: Version of the model used
            
        Returns:
            str: The created document ID, or None if failed.
        """
        try:
            # Build audio metadata
            audio_metadata = AudioMetadata(
                sample_rate=sample_rate,
                duration_seconds=duration_seconds,
            )
            
            # Build classification result
            classification_result = ClassificationResult(
                predicted_class=CryClass(predicted_class),
                confidence_score=confidence_score,
                all_probabilities=all_probabilities or {},
                model_version=model_version,
            )
            
            # Build MFCC features if provided
            mfcc = None
            if mfcc_features:
                mfcc = MFCCFeatures(features=mfcc_features)
            
            # Create the classification document
            classification = CryClassification(
                timestamp=datetime.utcnow(),
                device_source=DeviceSource(device_source),
                device_id=device_id,
                session_id=session_id,
                audio_metadata=audio_metadata,
                mfcc_features=mfcc,
                classification=classification_result,
            )
            
            # Save to database
            doc_id = self.classification_repo.create(classification)
            
            if doc_id:
                # Update related records
                if session_id:
                    self.session_repo.add_classification_to_session(session_id, doc_id)
                
                logger.info(f"Classification saved: {predicted_class} (confidence: {confidence_score})")
            
            return doc_id
            
        except Exception as e:
            logger.error(f"Error saving classification: {e}")
            return None
    
    def get_recent_classifications(
        self,
        limit: int = 50,
        device_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get recent classification results.
        
        Args:
            limit: Maximum number of results.
            device_id: Optional filter by device.
            
        Returns:
            List of classification documents.
        """
        return self.classification_repo.find_recent(
            limit=limit,
            device_id=device_id,
        )
    
    def verify_classification(
        self,
        classification_id: str,
        verified_class: str,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Mark a classification as verified by the user.
        
        This is used for improving model accuracy through user feedback.
        
        Args:
            classification_id: The classification document ID.
            verified_class: The correct class as verified by user.
            notes: Optional notes about the verification.
            
        Returns:
            bool: True if update was successful.
        """
        return self.classification_repo.update_verification(
            id=classification_id,
            verified_class=CryClass(verified_class),
            notes=notes,
        )
    
    def get_statistics(
        self,
        device_id: Optional[str] = None,
        days: int = 7,
    ) -> Dict[str, Any]:
        """
        Get classification statistics.
        
        Provides aggregated statistics for analysis and visualization.
        
        Args:
            device_id: Optional filter by device.
            days: Number of days to analyze.
            
        Returns:
            Dictionary with aggregated statistics.
        """
        return self.classification_repo.get_statistics(
            device_id=device_id,
            days=days,
        )


class SessionService:
    """
    Service for managing audio monitoring sessions.
    
    Handles session lifecycle for continuous monitoring mode.
    """
    
    def __init__(self):
        self.session_repo = AudioSessionRepository()
        self.device_repo = DeviceRegistrationRepository()
    
    def start_session(
        self,
        device_id: str,
        device_source: str,
    ) -> Optional[str]:
        """
        Start a new audio monitoring session.
        
        Args:
            device_id: The device starting the session.
            device_source: Device type ('esp32' or 'android').
            
        Returns:
            str: Session ID if created, None if failed.
        """
        try:
            session_id = str(uuid.uuid4())
            
            session = AudioSession(
                session_id=session_id,
                device_id=device_id,
                device_source=DeviceSource(device_source),
            )
            
            doc_id = self.session_repo.create(session)
            if doc_id:
                # Update device last seen
                self.device_repo.update_last_seen(device_id)
                return session_id
            
            return None
            
        except Exception as e:
            logger.error(f"Error starting session: {e}")
            return None
    
    def end_session(self, session_id: str) -> bool:
        """
        End an active session.
        
        Args:
            session_id: The session to end.
            
        Returns:
            bool: True if session was ended.
        """
        return self.session_repo.end_session(session_id)
    
    def get_active_sessions(
        self,
        device_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all active monitoring sessions.
        
        Args:
            device_id: Optional filter by device.
            
        Returns:
            List of active session documents.
        """
        return self.session_repo.find_active_sessions(device_id)


class DeviceService:
    """
    Service for device registration and management.
    """
    
    def __init__(self):
        self.device_repo = DeviceRegistrationRepository()
    
    def register_esp32(
        self,
        device_id: str,
        mac_address: Optional[str] = None,
        firmware_version: Optional[str] = None,
    ) -> Optional[str]:
        """
        Register an ESP32 device.
        
        Args:
            device_id: Unique device identifier.
            mac_address: ESP32 MAC address.
            firmware_version: Firmware version.
            
        Returns:
            str: Device ID if registered.
        """
        device = DeviceRegistration(
            device_id=device_id,
            device_type=DeviceSource.ESP32,
            mac_address=mac_address,
            firmware_version=firmware_version,
        )
        return self.device_repo.register_device(device)
    
    def register_android(
        self,
        device_id: str,
        app_version: Optional[str] = None,
    ) -> Optional[str]:
        """
        Register an Android device.
        
        Args:
            device_id: Unique device identifier.
            app_version: Mobile app version.
            
        Returns:
            str: Device ID if registered.
        """
        device = DeviceRegistration(
            device_id=device_id,
            device_type=DeviceSource.ANDROID,
            platform="android",
            app_version=app_version,
        )
        return self.device_repo.register_device(device)
    
    def get_devices_by_type(self, device_type: str) -> List[Dict[str, Any]]:
        """
        Get all devices of a specific type.
        
        Args:
            device_type: 'esp32' or 'android'.
            
        Returns:
            List of device documents.
        """
        return self.device_repo.find_by_type(DeviceSource(device_type))
    
    def heartbeat(self, device_id: str) -> bool:
        """
        Update device last seen timestamp.
        
        Should be called periodically by devices to indicate activity.
        
        Args:
            device_id: The device ID.
            
        Returns:
            bool: True if updated.
        """
        return self.device_repo.update_last_seen(device_id)
