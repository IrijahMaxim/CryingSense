"""
Database Initialization Script

Sets up MongoDB collections, indexes, and initial data.
Run this script when setting up the database for the first time.
"""

import logging
from datetime import datetime
from pymongo.errors import CollectionInvalid

from .config import get_database, close_connection
from .repository import (
    CryClassificationRepository,
    AudioSessionRepository,
    DeviceRegistrationRepository,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Collection names
COLLECTIONS = [
    "cry_classifications",
    "audio_sessions",
    "device_registrations",
]


def create_collections() -> None:
    """Create all required collections with validation schemas."""
    db = get_database()
    existing = db.list_collection_names()
    
    for collection_name in COLLECTIONS:
        if collection_name not in existing:
            try:
                db.create_collection(collection_name)
                logger.info(f"Created collection: {collection_name}")
            except CollectionInvalid:
                logger.warning(f"Collection already exists: {collection_name}")
        else:
            logger.info(f"Collection exists: {collection_name}")


def create_indexes() -> None:
    """
    Create all required indexes.
    
    Each repository handles its own indexes, but we call them
    here to ensure they're created during initialization.
    """
    logger.info("Creating indexes...")
    
    # Initialize repositories (this triggers index creation)
    CryClassificationRepository()
    AudioSessionRepository()
    DeviceRegistrationRepository()
    
    logger.info("Indexes created successfully")


def setup_validation_schemas() -> None:
    """
    Set up MongoDB validation schemas for collections.
    
    This provides server-side validation for documents.
    """
    db = get_database()
    
    # Cry classifications validation
    cry_classification_schema = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["device_source", "audio_metadata", "classification"],
            "properties": {
                "device_source": {
                    "bsonType": "string",
                    "enum": ["esp32", "android"],
                    "description": "Source device type"
                },
                "classification": {
                    "bsonType": "object",
                    "required": ["predicted_class", "confidence_score"],
                    "properties": {
                        "predicted_class": {
                            "bsonType": "string",
                            "enum": ["hunger", "tired", "discomfort", "belly_pain", "burp"]
                        },
                        "confidence_score": {
                            "bsonType": "double",
                            "minimum": 0,
                            "maximum": 1
                        }
                    }
                }
            }
        }
    }
    
    try:
        db.command({
            "collMod": "cry_classifications",
            "validator": cry_classification_schema,
            "validationLevel": "moderate",
            "validationAction": "warn"
        })
        logger.info("Validation schema applied to cry_classifications")
    except Exception as e:
        logger.warning(f"Could not apply validation schema: {e}")


def initialize_database() -> None:
    """
    Initialize the database with all required setup.
    
    Call this function when deploying or setting up the system.
    """
    logger.info("=" * 50)
    logger.info("CryingSense Database Initialization")
    logger.info("=" * 50)
    
    try:
        # Step 1: Create collections
        logger.info("\n[1/3] Creating collections...")
        create_collections()
        
        # Step 2: Create indexes
        logger.info("\n[2/3] Creating indexes...")
        create_indexes()
        
        # Step 3: Setup validation
        logger.info("\n[3/3] Setting up validation schemas...")
        setup_validation_schemas()
        
        logger.info("\n" + "=" * 50)
        logger.info("Database initialization completed successfully!")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
    finally:
        close_connection()


def reset_database(confirm: bool = False) -> None:
    """
    Reset the database by dropping all collections.
    
    WARNING: This will delete all data!
    
    Args:
        confirm: Must be True to proceed with reset.
    """
    if not confirm:
        logger.warning("Reset aborted. Pass confirm=True to proceed.")
        return
    
    logger.warning("=" * 50)
    logger.warning("RESETTING DATABASE - ALL DATA WILL BE LOST")
    logger.warning("=" * 50)
    
    db = get_database()
    
    for collection_name in COLLECTIONS:
        try:
            db.drop_collection(collection_name)
            logger.info(f"Dropped collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error dropping {collection_name}: {e}")
    
    logger.info("Database reset complete. Run initialize_database() to set up again.")
    close_connection()


if __name__ == "__main__":
    initialize_database()
