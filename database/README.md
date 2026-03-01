# CryingSense Database Module

MongoDB database integration for the CryingSense infant cry classification system.

## Overview

This module handles all database operations for storing, retrieving, and analyzing cry classification data from the IoT hardware (ESP32 + Raspberry Pi) and Android mobile application.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CryingSense System                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐              ┌────────────────────┐                   │
│  │    ESP32     │   Audio      │   Raspberry Pi     │                   │
│  │  (INMP441)   │ ──────────▶  │   3 B+             │                   │
│  │              │              │                    │                   │
│  │ Continuous   │              │ • Preprocessing    │                   │
│  │ Audio Capture│              │   - Noise reduction│                   │
│  └──────────────┘              │   - Normalization  │                   │
│                                │ • MFCC Extraction  │                   │
│                                │ • PyTorch CNN      │                   │
│                                │   Inference        │                   │
│                                └─────────┬──────────┘                   │
│                                          │                              │
│                                          │ Classification Results       │
│                                          ▼                              │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                        MongoDB Database                           │  │
│  │                                                                   │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │  │
│  │  │ cry_classifications│ │ audio_sessions │  │device_registrations││  │
│  │  │                 │  │                 │  │                 │    │  │
│  │  │ • predicted_class│  │ • session_id   │  │ • device_id     │    │  │
│  │  │ • confidence    │  │ • device_id     │  │ • device_type   │    │  │
│  │  │ • timestamp     │  │ • start/end     │  │ • last_seen     │    │  │
│  │  │ • audio_metadata│  │ • cry_count     │  │                 │    │  │
│  │  │ • mfcc_features │  │                 │  │                 │    │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘    │  │
│  │                                                                   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                               │ Read/Write                              │
│  ┌──────────────────┐         │                                         │
│  │ Android App      │ ────────┘                                         │
│  │                  │                                                   │
│  │ • Microphone     │                                                   │
│  │   capture        │                                                   │
│  │ • Display results│                                                   │
│  │ • View history   │                                                   │
│  └──────────────────┘                                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Hardware IoT Process

1. **ESP32** runs continuously and captures infant cry audio using INMP441 microphone
2. Audio is sent to **Raspberry Pi 3 B+**
3. **Raspberry Pi** processes the audio:
   - Applies preprocessing (noise reduction, normalization)
   - Extracts MFCC features
   - Runs PyTorch CNN model for classification
   - Sends results to database and connected Android application
4. **Model outputs** stored in database:
   - Predicted class: `hunger`, `tired`, `discomfort`, `belly_pain`, `burp`
   - Confidence score (0-1)
   - Timestamp

### Android Application Process

1. User can hold down to record continuously OR manually record
2. Audio is processed locally or sent to Raspberry Pi
3. Classification results stored in database for history

## Collections

### `cry_classifications`
Primary collection for all classification results.

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | DateTime | When the cry was recorded |
| `device_source` | String | `esp32` or `android` |
| `device_id` | String | Device identifier |
| `session_id` | String | Optional session reference |
| `audio_metadata` | Object | Sample rate, duration, channels |
| `mfcc_features` | Object | Extracted MFCC coefficients |
| `classification` | Object | Predicted class, confidence, probabilities |
| `is_verified` | Boolean | User verification status |
| `verified_class` | String | User-corrected classification |

### `audio_sessions`
Groups continuous monitoring events.

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | String | Unique session identifier |
| `device_id` | String | Device that started session |
| `is_active` | Boolean | Session status |
| `started_at` | DateTime | Session start time |
| `ended_at` | DateTime | Session end time |
| `total_cries_detected` | Number | Count of cries in session |

### `device_registrations`
Registered devices in the system.

| Field | Type | Description |
|-------|------|-------------|
| `device_id` | String | Unique identifier |
| `device_type` | String | `esp32` or `android` |
| `mac_address` | String | ESP32 MAC (optional) |
| `firmware_version` | String | ESP32 firmware version |
| `is_active` | Boolean | Device status |
| `last_seen` | DateTime | Last activity timestamp |

## Installation

### Prerequisites

- Python 3.9+
- MongoDB 6.0+

### Setup

1. Install MongoDB locally or use MongoDB Atlas:
   ```bash
   # Local installation (Ubuntu/Debian)
   sudo apt-get install mongodb
   
   # Or use Docker
   docker run -d -p 27017:27017 --name cryingsense-mongo mongo:6.0
   ```

2. Install Python dependencies:
   ```bash
   pip install -r database/requirements.txt
   ```

3. Configure environment:
   ```bash
   cp database/.env.example database/.env
   # Edit .env with your MongoDB settings
   ```

4. Initialize the database:
   ```bash
   python -m database.init_db
   ```

## Usage

### Saving a Classification (Raspberry Pi)

```python
from database.services import CryClassificationService

service = CryClassificationService()

# Save classification result
doc_id = service.save_classification(
    predicted_class="hunger",
    confidence_score=0.92,
    device_source="esp32",
    duration_seconds=3.5,
    device_id="ESP32-001",
    all_probabilities={
        "hunger": 0.92,
        "tired": 0.04,
        "discomfort": 0.02,
        "belly_pain": 0.01,
        "burp": 0.01
    }
)

print(f"Saved classification: {doc_id}")
```

### Querying Classifications

```python
from database.services import CryClassificationService

service = CryClassificationService()

# Get recent classifications
recent = service.get_recent_classifications(limit=10)

# Get statistics
stats = service.get_statistics(days=7)
print(f"Total cries: {stats['total_classifications']}")
print(f"By class: {stats['by_class']}")
```

### Managing Sessions

```python
from database.services import SessionService

session_service = SessionService()

# Start a session
session_id = session_service.start_session(
    device_id="ESP32-001",
    device_source="esp32"
)

# End the session
session_service.end_session(session_id)
```

### Device Registration

```python
from database.services import DeviceService

device_service = DeviceService()

# Register ESP32
device_service.register_esp32(
    device_id="ESP32-001",
    mac_address="AA:BB:CC:DD:EE:FF",
    firmware_version="2.0.0"
)

# Heartbeat
device_service.heartbeat("ESP32-001")
```

## API for External Services

### Raspberry Pi Integration

The Raspberry Pi should call the database service after each classification:

```python
# raspberry_pi/inference.py
from database.services import CryClassificationService

def on_classification_complete(result, audio_info):
    """Called after model inference completes."""
    service = CryClassificationService()
    
    service.save_classification(
        predicted_class=result["class"],
        confidence_score=result["confidence"],
        device_source="esp32",
        duration_seconds=audio_info["duration"],
        sample_rate=audio_info["sample_rate"],
        device_id=audio_info["device_id"],
        all_probabilities=result["probabilities"],
        mfcc_features=result.get("mfcc")
    )
```

### Android Application Integration

The Android app should communicate via REST API (to be implemented) or direct MongoDB connection:

```python
# REST API endpoint example (to be added)
POST /api/classifications
{
    "predicted_class": "hunger",
    "confidence_score": 0.87,
    "device_source": "android",
    "device_id": "android-user-123",
    "duration_seconds": 4.2
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGO_HOST` | localhost | MongoDB host |
| `MONGO_PORT` | 27017 | MongoDB port |
| `MONGO_DATABASE` | cryingsense | Database name |
| `MONGO_USERNAME` | - | Username (optional) |
| `MONGO_PASSWORD` | - | Password (optional) |
| `MONGO_AUTH_SOURCE` | admin | Auth database |
| `MONGO_MAX_POOL_SIZE` | 50 | Max connections |
| `MONGO_MIN_POOL_SIZE` | 10 | Min connections |

## File Structure

```
database/
├── __init__.py          # Module exports
├── config.py            # Database connection configuration
├── models.py            # Pydantic data models
├── repository.py        # Data access layer (CRUD)
├── services.py          # Business logic layer
├── init_db.py           # Database initialization script
├── requirements.txt     # Python dependencies
├── .env.example         # Environment template
└── README.md            # This file
```

## Best Practices

1. **Use Services Layer**: Always interact through `services.py` for business operations
2. **Connection Management**: The singleton pattern ensures efficient connection pooling
3. **Indexing**: Indexes are automatically created on repository initialization
4. **Validation**: Pydantic models provide type safety and validation
5. **Error Handling**: All repository operations include error handling and logging

## Next Steps

- [ ] Add REST API layer for mobile app communication
- [ ] Implement data export functionality
- [ ] Add backup/restore utilities
- [ ] Create MongoDB Atlas deployment guide
- [ ] Add real-time change streams for live updates
