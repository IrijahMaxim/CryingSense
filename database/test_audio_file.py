"""
Test script for inserting audio file data into MongoDB

This demonstrates how to store audio file metadata and data
using different storage methods.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from database.models import AudioFile, AudioMetadata, DeviceSource
from database.repository import AudioFileRepository


def test_audio_file_with_path():
    """Example: Store audio file with filesystem path."""
    
    # Create audio metadata
    metadata = AudioMetadata(
        sample_rate=44100,
        duration_seconds=3.5,
        channels=1,
        bit_depth=16,
        file_size_bytes=308700
    )
    
    # Create audio file record
    audio_file = AudioFile(
        file_id="audio_20240301_001",
        original_filename="baby_cry_hunger.wav",
        file_path="/data/audio/recordings/baby_cry_hunger.wav",  # Local path
        audio_metadata=metadata,
        mime_type="audio/wav",
        device_id="ESP32-001",
        session_id="session_20240301_001"
    )
    
    # Store in database
    repo = AudioFileRepository()
    file_id = repo.create(audio_file)
    
    if file_id:
        print(f"✅ Audio file stored successfully!")
        print(f"   File ID: {file_id}")
        print(f"   Path: {audio_file.file_path}")
        
        # Retrieve it back
        retrieved = repo.find_by_file_id(file_id)
        print(f"\n📥 Retrieved audio file:")
        print(f"   Original filename: {retrieved['original_filename']}")
        print(f"   Duration: {retrieved['audio_metadata']['duration_seconds']}s")
        print(f"   Sample rate: {retrieved['audio_metadata']['sample_rate']} Hz")
    else:
        print("❌ Failed to store audio file")


def test_audio_file_with_url():
    """Example: Store audio file with cloud storage URL."""
    
    metadata = AudioMetadata(
        sample_rate=16000,
        duration_seconds=2.8,
        channels=1,
        bit_depth=16,
        file_size_bytes=89600
    )
    
    audio_file = AudioFile(
        file_id="audio_20240301_002",
        original_filename="cry_tired.wav",
        file_url="https://yourbucket.s3.amazonaws.com/audio/cry_tired.wav",  # S3/cloud URL
        audio_metadata=metadata,
        mime_type="audio/wav",
        device_id="android_device_123",
        expires_at=datetime.utcnow() + timedelta(days=30)  # Auto-delete after 30 days
    )
    
    repo = AudioFileRepository()
    file_id = repo.create(audio_file)
    
    if file_id:
        print(f"✅ Audio file stored successfully!")
        print(f"   File ID: {file_id}")
        print(f"   URL: {audio_file.file_url}")
        print(f"   Expires: {audio_file.expires_at}")


def test_audio_file_with_base64():
    """Example: Store small audio file as base64 (for small files only)."""
    
    # Small audio sample (in real case, encode your actual audio)
    sample_base64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="
    
    metadata = AudioMetadata(
        sample_rate=8000,
        duration_seconds=0.5,
        channels=1,
        bit_depth=8,
        file_size_bytes=len(sample_base64)
    )
    
    audio_file = AudioFile(
        file_id="audio_20240301_003",
        original_filename="test_beep.wav",
        audio_data_base64=sample_base64,  # Store directly in DB
        audio_metadata=metadata,
        mime_type="audio/wav",
        device_id="ESP32-002"
    )
    
    repo = AudioFileRepository()
    file_id = repo.create(audio_file)
    
    if file_id:
        print(f"✅ Audio file stored successfully!")
        print(f"   File ID: {file_id}")
        print(f"   Storage: Base64 (embedded in DB)")


def test_find_audio_files():
    """Example: Query audio files."""
    
    repo = AudioFileRepository()
    
    # Find by device
    device_files = repo.find_by_device("ESP32-001", limit=10)
    print(f"\n🔍 Found {len(device_files)} audio files from ESP32-001")
    
    # Find by session
    session_files = repo.find_by_session("session_20240301_001")
    print(f"🔍 Found {len(session_files)} audio files in session")
    
    # Find specific file
    audio = repo.find_by_file_id("audio_20240301_001")
    if audio:
        print(f"🔍 Found audio file: {audio['original_filename']}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Audio File Storage")
    print("=" * 60)
    
    print("\n📁 Test 1: Store audio with filesystem path")
    print("-" * 60)
    test_audio_file_with_path()
    
    print("\n\n📁 Test 2: Store audio with cloud URL")
    print("-" * 60)
    test_audio_file_with_url()
    
    print("\n\n📁 Test 3: Store audio as base64")
    print("-" * 60)
    test_audio_file_with_base64()
    
    print("\n\n📁 Test 4: Query audio files")
    print("-" * 60)
    test_find_audio_files()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
