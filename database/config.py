"""
MongoDB Configuration Module

Handles database connection settings and initialization.
Supports both local development and production environments.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pymongo import MongoClient

# Load .env file from the database directory
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseConfig:
    """MongoDB configuration settings."""
    
    # Atlas connection string (preferred for cloud deployments)
    MONGO_URI: Optional[str] = os.getenv("MONGO_URI")
    
    # Legacy connection settings (for local development)
    MONGO_HOST: str = os.getenv("MONGO_HOST", "localhost")
    MONGO_PORT: int = int(os.getenv("MONGO_PORT", "27017"))
    MONGO_USERNAME: Optional[str] = os.getenv("MONGO_USERNAME")
    MONGO_PASSWORD: Optional[str] = os.getenv("MONGO_PASSWORD")
    MONGO_DATABASE: str = os.getenv("MONGO_DATABASE", "cryingsense")
    MONGO_AUTH_SOURCE: str = os.getenv("MONGO_AUTH_SOURCE", "admin")
    
    # Connection pool settings
    MAX_POOL_SIZE: int = int(os.getenv("MONGO_MAX_POOL_SIZE", "50"))
    MIN_POOL_SIZE: int = int(os.getenv("MONGO_MIN_POOL_SIZE", "10"))
    
    # Timeout settings (in milliseconds)
    CONNECTION_TIMEOUT_MS: int = int(os.getenv("MONGO_CONNECTION_TIMEOUT", "5000"))
    SERVER_SELECTION_TIMEOUT_MS: int = int(os.getenv("MONGO_SERVER_SELECTION_TIMEOUT", "5000"))
    
    @classmethod
    def get_connection_uri(cls) -> str:
        """Build MongoDB connection URI. Prefers MONGO_URI if set."""
        # Use Atlas URI if provided
        if cls.MONGO_URI:
            return cls.MONGO_URI
        
        # Fall back to host/port for local development
        if cls.MONGO_USERNAME and cls.MONGO_PASSWORD:
            return (
                f"mongodb://{cls.MONGO_USERNAME}:{cls.MONGO_PASSWORD}"
                f"@{cls.MONGO_HOST}:{cls.MONGO_PORT}"
                f"/?authSource={cls.MONGO_AUTH_SOURCE}"
            )
        return f"mongodb://{cls.MONGO_HOST}:{cls.MONGO_PORT}"


class MongoDBConnection:
    """
    Singleton MongoDB connection manager.
    
    Ensures a single database connection is shared across the application.
    """
    
    _instance: Optional["MongoDBConnection"] = None
    _client: Optional[MongoClient] = None
    _database: Optional[Database] = None
    
    def __new__(cls) -> "MongoDBConnection":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def connect(self) -> Database:
        """
        Establish connection to MongoDB.
        
        Returns:
            Database: MongoDB database instance.
            
        Raises:
            ConnectionFailure: If connection cannot be established.
        """
        if self._client is None:
            try:
                uri = DatabaseConfig.get_connection_uri()
                
                self._client = MongoClient(
                    uri,
                    maxPoolSize=DatabaseConfig.MAX_POOL_SIZE,
                    minPoolSize=DatabaseConfig.MIN_POOL_SIZE,
                    connectTimeoutMS=DatabaseConfig.CONNECTION_TIMEOUT_MS,
                    serverSelectionTimeoutMS=DatabaseConfig.SERVER_SELECTION_TIMEOUT_MS,
                )
                
                # Verify connection
                self._client.admin.command("ping")
                
                # Log connection info (mask credentials)
                if DatabaseConfig.MONGO_URI:
                    # Extract host from URI for logging (hide credentials)
                    import re
                    host_match = re.search(r'@([^/]+)', DatabaseConfig.MONGO_URI)
                    host_info = host_match.group(1) if host_match else "Atlas"
                    logger.info(f"Connected to MongoDB Atlas: {host_info}")
                else:
                    logger.info(f"Connected to MongoDB at {DatabaseConfig.MONGO_HOST}:{DatabaseConfig.MONGO_PORT}")
                
                self._database = self._client[DatabaseConfig.MONGO_DATABASE]
                
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                raise
                
        return self._database
    
    def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._database = None
            logger.info("Disconnected from MongoDB")
    
    @property
    def database(self) -> Database:
        """Get the database instance, connecting if necessary."""
        if self._database is None:
            return self.connect()
        return self._database
    
    @property
    def client(self) -> MongoClient:
        """Get the client instance, connecting if necessary."""
        if self._client is None:
            self.connect()
        return self._client


# Global connection instance
_connection = MongoDBConnection()


def get_database() -> Database:
    """
    Get the MongoDB database instance.
    
    Returns:
        Database: MongoDB database instance.
    """
    return _connection.database


def get_collection(collection_name: str) -> Collection:
    """
    Get a specific collection from the database.
    
    Args:
        collection_name: Name of the collection.
        
    Returns:
        Collection: MongoDB collection instance.
    """
    return _connection.database[collection_name]


def close_connection() -> None:
    """Close the database connection."""
    _connection.disconnect()
