# Database Integration for CryingSense Model

This document explains how to use the database integration feature for storing cry classifications, audio sessions, and audio files.

## Overview

The enhanced `predict.py` now supports automatic database integration that saves:
1. **Cry Classification** - Model predictions with confidence scores
2. **Audio Session** - Grouping of multiple predictions in a monitoring session
3. **Audio Files** - Original audio files linked to classifications

## Setup

### Prerequisites

1. MongoDB instance running (see `database/README.md`)
2. Database dependencies installed:
   ```bash
   pip install -r database/requirements.txt
   ```

3. Configure database connection in `database/config.py`

## Usage

### Basic Inference with Database Storage

```bash
# Single file with database storage
python predict.py \
  --audio test.wav \
  --model ../saved_models/cryingsense_cnn.pth \
  --save-to-db \
  --device-id ESP32-001 \
  --device-source esp32
```

### Session-Based Monitoring

For continuous monitoring, you can group predictions into sessions:

```bash
# Start with a session ID
python predict.py \
  --audio test.wav \
  --model ../saved_models/cryingsense_cnn.pth \
  --save-to-db \
  --device-id ESP32-001 \
  --device-source esp32 \
  --session-id monitoring-session-001
```

### Custom Audio Storage Location

By default, audio files are stored in `project_root/audio_storage/`. You can customize this:

```bash
python predict.py \
  --audio test.wav \
  --model ../saved_models/cryingsense_cnn.pth \
  --save-to-db \
  --device-id ESP32-001 \
  --audio-storage-dir /path/to/audio/storage
```

### Batch Inference with Database Storage

```bash
python predict.py \
  --batch audio_directory/ \
  --model ../saved_models/cryingsense_cnn.pth \
  --save-to-db \
  --device-id ESP32-001 \
  --session-id batch-001 \
  --recursive
```

## Python API Usage

### Example 1: Single Prediction with Database

```python
from model.inference.predict import CryingSensePredictor

# Initialize predictor with database enabled
predictor = CryingSensePredictor(
    model_path='../saved_models/cryingsense_cnn.pth',
    save_to_db=True,
    device_id='ESP32-001',
    device_source='esp32'
)

# Perform inference (automatically saves to database)
result = predictor.predict_single('baby_cry.wav')

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
print(f"Classification ID: {result.get('classification_id')}")
print(f"Audio File ID: {result.get('audio_file_id')}")
```

### Example 2: Session-Based Monitoring

```python
from model.inference.predict import CryingSensePredictor

# Initialize predictor
predictor = CryingSensePredictor(
    model_path='../saved_models/cryingsense_cnn.pth',
    save_to_db=True,
    device_id='ESP32-001',
    device_source='esp32'
)

# Start a new monitoring session
session_id = predictor.start_new_session()
print(f"Started session: {session_id}")

# Process multiple audio files in the session
audio_files = ['cry1.wav', 'cry2.wav', 'cry3.wav']
for audio_file in audio_files:
    result = predictor.predict_single(audio_file)
    if result['is_cry']:
        print(f"Cry detected: {result['prediction']} ({result['confidence']:.2%})")

# End the session
predictor.end_current_session()
```

### Example 3: Selective Database Saving

```python
from model.inference.predict import CryingSensePredictor

# Initialize predictor with database disabled by default
predictor = CryingSensePredictor(
    model_path='../saved_models/cryingsense_cnn.pth',
    save_to_db=False,  # Default: don't save
    device_id='ESP32-001'
)

# Process some files without saving
result1 = predictor.predict_single('test1.wav')

# Save only specific predictions
result2 = predictor.predict_single('important.wav', save_to_db=True)
```

### Example 4: Real-Time ESP32 Integration

```python
from model.inference.predict import CryingSensePredictor
import time

# Initialize for ESP32 device
predictor = CryingSensePredictor(
    model_path='../saved_models/cryingsense_cnn_quantized.pth',
    save_to_db=True,
    device_id='ESP32-001',
    device_source='esp32',
    confidence_threshold=0.75
)

# Start monitoring session
session_id = predictor.start_new_session()

try:
    # Continuous monitoring loop
    while True:
        # Assume audio is captured and saved as temp file
        audio_path = '/tmp/captured_audio.wav'
        
        # Process audio
        result = predictor.predict_single(audio_path)
        
        # Handle result
        if result['alert']:
            print(f"⚠️  ALERT: {result['prediction']} detected!")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Saved as: {result.get('classification_id')}")
        
        time.sleep(1)  # Adjust based on your needs
        
except KeyboardInterrupt:
    print("\nStopping monitoring...")
    predictor.end_current_session()
```

## Database Schema

### CryClassification Document

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "device_source": "esp32",
  "device_id": "ESP32-001",
  "session_id": "session-123",
  "audio_metadata": {
    "sample_rate": 16000,
    "duration_seconds": 3.5,
    "channels": 1,
    "bit_depth": 16
  },
  "classification": {
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
```

### AudioFile Document

```json
{
  "file_id": "audio_20240115_103000_abc12345",
  "original_filename": "baby_cry.wav",
  "file_path": "/path/to/audio_storage/audio_20240115_103000_abc12345.wav",
  "audio_metadata": {
    "sample_rate": 16000,
    "duration_seconds": 3.5,
    "channels": 1,
    "bit_depth": 16,
    "file_size_bytes": 112000
  },
  "mime_type": "audio/wav",
  "device_id": "ESP32-001",
  "session_id": "session-123",
  "classification_id": "classification_id_here",
  "uploaded_at": "2024-01-15T10:30:00Z"
}
```

## Querying Stored Data

Use the database services to query stored data:

```python
from database.services import CryClassificationService

service = CryClassificationService()

# Get recent classifications
recent = service.get_recent_classifications(limit=20, device_id='ESP32-001')

# Get statistics
stats = service.get_statistics(device_id='ESP32-001', days=7)
print(f"Total classifications: {stats['total_count']}")
print(f"Class distribution: {stats['class_distribution']}")
```

## Important Notes

1. **Storage Space**: Audio files can consume significant disk space. Consider implementing cleanup policies or using cloud storage.

2. **Noise Detection**: The model detects noise but doesn't save it to the database (only actual cry classifications are saved).

3. **Performance**: Database operations add ~10-50ms to inference time depending on your setup.

4. **Error Handling**: If database operations fail, the prediction still completes - errors are logged as warnings.

5. **Privacy**: Ensure proper encryption and access controls for stored audio files containing sensitive data.

## Troubleshooting

### Database Connection Failed
```
Warning: Database modules not available. Running without database integration.
```
**Solution**: Check MongoDB is running and connection settings in `database/config.py`

### Audio File Save Failed
```
Warning: Failed to save audio file: [error]
```
**Solution**: Check write permissions on audio storage directory

### Classification Not Saved
```
Skipping database save (no cry detected)
```
**Note**: This is expected - noise predictions are not saved to avoid cluttering the database.

## See Also

- `database/README.md` - Database setup and configuration
- `database/services.py` - Service layer API documentation
- `model/inference/README.md` - Model inference documentation
