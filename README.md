# CryingSense

AI- and IoT-powered infant cry interpretation system designed to help caregivers understand a baby's needs.

## Overview
CryingSense automatically classifies infant crying sounds into five categories:
- **Hunger** - Baby needs feeding
- **Tiredness** - Baby needs sleep
- **Discomfort** - Baby is uncomfortable (temperature, position, etc.)
- **Belly Pain** - Baby has abdominal discomfort
- **Burp** - Baby needs to burp

## System Architecture
The system consists of three main components:

1. **ESP32 Microphone Module** - Captures infant cry audio
2. **Raspberry Pi Edge Inference** - Runs PyTorch CNN model for real-time classification
3. **Mobile Application** - Displays alerts and interpretations to caregivers

## Tech Stack
- **AI Framework**: PyTorch 2.6.0+
- **Audio Processing**: Librosa (MFCC, Mel-spectrogram, Chroma features)
- **IoT Hardware**: ESP32 + Raspberry Pi 3B+
- **Backend**: Node.js + Express.js
- **Database**: MongoDB
- **Mobile**: React Native

## Project Structure
```
CryingSense/
├── dataset/                    # Audio dataset and processing
│   ├── raw/                   # Original recordings
│   ├── processed/             # Cleaned audio and features
│   └── dataset_info.md        # Dataset documentation
│
├── model/                      # Machine learning models
│   ├── training/              # Training scripts
│   ├── inference/             # Inference/prediction code
│   ├── models/                # Model architectures
│   ├── saved_models/          # Trained model files
│   └── requirements.txt       # Python dependencies
│
├── iot/                        # IoT device code
│   ├── raspberry_pi/          # Raspberry Pi inference server
│   └── ESP32/                 # ESP32 microphone firmware
│
├── experiments/                # Training experiments and results
│   ├── logs/                  # Training logs
│   ├── confusion_matrices/    # Model evaluation visualizations
│   └── performance_reports/   # Performance metrics
│
└── scripts/                    # Processing utilities
    ├── preprocess_audio.py    # Audio preprocessing
    ├── feature_extraction.py  # Feature extraction
    └── dataset_split.py       # Dataset splitting
```

## Getting Started

### Prerequisites
- Python 3.11+
- PyTorch 2.6.0+
- Librosa
- Node.js 20.0.0+
- MongoDB 7.0+

### Installation

1. Clone the repository
```bash
git clone https://github.com/IrijahMaxim/CryingSense.git
cd CryingSense
```

2. Install Python dependencies
```bash
pip install -r model/requirements.txt
```

3. Prepare the dataset
```bash
# Preprocess raw audio
python scripts/preprocess_audio.py

# Extract features
python scripts/feature_extraction.py

# Create train/val/test splits
python scripts/dataset_split.py
```

4. Train the model
```bash
python model/training/train.py
```

## Dataset Processing Pipeline

### 1. Preprocessing (`scripts/preprocess_audio.py`)
- Noise reduction
- Silence trimming
- Amplitude normalization
- Fixed 5-second duration
- Output: `dataset/processed/cleaned/`

### 2. Feature Extraction (`scripts/feature_extraction.py`)
- MFCC: 40 coefficients
- Mel Spectrogram: 128 bands
- Chroma: 12 bins
- Output: `dataset/processed/feature_extraction/`

### 3. Dataset Split (`scripts/dataset_split.py`)
- 70% training, 15% validation, 15% test
- Session-level splitting (no data leakage)
- Output: `dataset/dataset_split.json`

## Model Architecture
Convolutional Neural Network (CNN) operating on:
- Input: Multi-channel acoustic features (MFCC, Mel, Chroma)
- Architecture: CNN layers → Pooling → Fully connected
- Output: 5-class classification (hunger, tired, discomfort, belly_pain, burp)
- Optimization: Quantization for edge deployment

## Deployment

### Edge Inference (Raspberry Pi)
- Model: Quantized PyTorch CNN
- Latency: <500ms per 5-second sample
- Memory: <512MB
- See `iot/raspberry_pi/README.md`

### ESP32 Audio Capture
- Sample Rate: 16kHz
- Format: 16-bit mono
- Buffer: 5-second segments
- See `iot/ESP32/README.md`

## Performance Targets
- **Accuracy**: >85% on test set
- **Inference Latency**: <500ms on Raspberry Pi
- **False Positive Rate**: <10%
- **Memory Usage**: <512MB

## Documentation
- [Dataset Information](dataset/dataset_info.md)
- [Raspberry Pi Setup](iot/raspberry_pi/README.md)
- [ESP32 Setup](iot/ESP32/README.md)
- [Model Inference](model/inference/README.md)

## Contributing
Contributions are welcome! Please ensure:
- Code follows existing style
- Processing scripts maintain data quality
- Models meet performance requirements
- Documentation is updated

## License
This project is licensed under the MIT License.

## Acknowledgments
- Infant cry dataset sources
- PyTorch and Librosa communities
- IoT hardware contributors

## Contact
For questions or issues, please open an issue on GitHub.
