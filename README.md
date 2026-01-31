# 👶 CryingSense

**AI- and IoT-Powered Infant Cry Interpretation System**

CryingSense is an intelligent system that automatically classifies infant crying sounds into five categories—**hunger**, **tiredness**, **discomfort**, **belly pain**, and **need to burp**—helping caregivers quickly understand and respond to a baby's needs.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Node.js](https://img.shields.io/badge/Node.js-20.0-green.svg)](https://nodejs.org)
[![React Native](https://img.shields.io/badge/React%20Native-0.73-61DAFB.svg)](https://reactnative.dev)

---

## 🎯 Overview

CryingSense uses:
- **ESP32** microcontroller with microphone to capture infant cry audio
- **Raspberry Pi 3 B+** running a PyTorch CNN model for real-time cry classification
- **Express.js backend** for data management and API services
- **React Native mobile app** for notifications and visualization

The system operates entirely on edge devices for maximum privacy and minimal latency.

---

## ✨ Key Features

- 🎤 **Real-time Audio Capture** - ESP32 continuously monitors for cry sounds
- 🧠 **Edge AI Classification** - Raspberry Pi runs CNN inference locally
- 📱 **Mobile Notifications** - Instant alerts with cry type and confidence
- 📊 **Historical Tracking** - View and analyze past cry events
- 🔒 **Privacy-First** - All audio processing happens on local devices
- ⚡ **Low Latency** - Near-instant classification and notification

### Cry Categories

| Category | Description | Icon |
|----------|-------------|------|
| **Hunger** | Baby needs feeding | 🍼 |
| **Tired** | Baby needs rest or sleep | 😴 |
| **Discomfort** | General discomfort, diaper change needed | 😣 |
| **Belly Pain** | Gas or digestive discomfort | 😖 |
| **Burp** | Trapped air needs to be released | 💨 |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CRYINGSENSE                           │
│                   System Architecture                         │
└─────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   ESP32 +    │──────│  Raspberry   │──────│   Backend    │
│  Microphone  │ WiFi │   Pi 3 B+    │ HTTP │   Server     │
│ (Transmitter)│      │  (Gateway)   │      │  (Express)   │
└──────────────┘      └──────────────┘      └──────────────┘
       │                     │                      │
   Captures              Runs CNN              Stores Data
   Audio                 Inference            & Coordinates
       │                     │                      │
       └─────────────────────┴──────────────────────┘
                              │
                    ┌─────────┴──────────┐
                    │   Mobile App       │
                    │  (React Native)    │
                    │  Notifications &   │
                    │   Visualization    │
                    └────────────────────┘
```

### Data Flow

1. **Audio Capture**: ESP32 captures 5-second audio samples
2. **Feature Extraction**: Basic features extracted on ESP32
3. **Transmission**: Features sent to Raspberry Pi via HTTP/MQTT
4. **Inference**: Raspberry Pi runs full CNN model on complete audio features
5. **Classification**: Result (cry type + confidence) generated
6. **Storage**: Backend stores event in MongoDB
7. **Notification**: Mobile app receives real-time notification

---

## 🔧 Tech Stack

### AI & Machine Learning
- **PyTorch** 2.6.0 - Deep learning framework
- **Librosa** - Audio feature extraction (MFCC, Mel-spectrogram, Chroma)
- **NumPy** & **SciPy** - Numerical processing
- **scikit-learn** - Model evaluation and metrics

### IoT Layer
- **ESP32-WROOM-32** - Audio capture and edge processing
- **Raspberry Pi 3 B+** - CNN inference and gateway
- **MicroPython** - ESP32 firmware
- **Python 3.11** - Raspberry Pi inference engine

### Backend
- **Node.js** 20.0.0 - JavaScript runtime
- **Express.js** 4.19.2 - REST API framework
- **MongoDB** 7.0 - NoSQL database
- **Mongoose** - MongoDB ODM

### Mobile Application
- **React Native** 0.73 - Mobile framework
- **Expo** 50.0 - Development toolchain
- **React Navigation** - Screen navigation
- **Axios** - HTTP client
- **Expo Notifications** - Push notifications

---

## 📁 Project Structure

```
CryingSense/
│
├── README.md                    # This file
├── .gitignore                   # Git ignore rules
│
├── dataset/                     # Audio dataset
│   ├── raw/                     # Raw audio files (WAV)
│   │   ├── belly_pain/
│   │   ├── burp/
│   │   ├── discomfort/
│   │   ├── hunger/
│   │   └── tired/
│   ├── processed/               # Processed audio and features
│   └── dataset_info.md          # Dataset documentation
│
├── model/                       # AI model components
│   ├── models/                  # Model architectures
│   │   ├── cnn_model.py         # CNN architecture
│   │   └── model_utils.py       # Utility functions
│   ├── training/                # Training scripts
│   │   ├── train.py             # Model training
│   │   ├── validate.py          # Validation
│   │   └── evaluate.py          # Evaluation
│   ├── inference/               # Inference engine
│   │   └── predict.py           # Prediction script
│   ├── saved_models/            # Trained model weights
│   └── requirements.txt         # Python dependencies
│
├── scripts/                     # Data processing scripts
│   ├── preprocess_audio.py      # Audio preprocessing
│   ├── feature_extraction.py    # Feature extraction
│   └── dataset_split.py         # Train/val/test split
│
├── iot/                         # IoT device code
│   ├── firmware/                # ESP32 & RPi firmware
│   │   ├── main.py              # ESP32 main controller
│   │   ├── audio_capture.py     # Audio capture module
│   │   ├── feature_extraction.py # ESP32 feature extraction
│   │   └── inference_engine.py  # RPi inference engine
│   ├── communication/           # Communication modules
│   │   ├── mqtt_client.py       # MQTT client
│   │   └── http_client.py       # HTTP client
│   ├── config/                  # Configuration
│   │   └── device_config.json   # Device settings
│   └── deployment/              # Deployment docs
│       └── flash_instructions.md # Flashing guide
│
├── backend/                     # Backend server
│   ├── server.js                # Express server
│   ├── routes/                  # API routes
│   ├── controllers/             # Business logic
│   ├── models/                  # MongoDB schemas
│   ├── database/                # DB connection
│   ├── package.json             # Node dependencies
│   └── .env.example             # Environment variables
│
├── mobile_app/                  # Mobile application
│   ├── CryingSenseApp/          # React Native app
│   │   ├── App.js               # Main app component
│   │   ├── screens/             # Screen components
│   │   ├── components/          # Reusable components
│   │   ├── services/            # API services
│   │   ├── navigation/          # Navigation config
│   │   ├── package.json         # RN dependencies
│   │   └── app.json             # Expo configuration
│   └── README.md                # Mobile app docs
│
└── experiments/                 # Training experiments
    ├── logs/                    # Training logs
    ├── confusion_matrices/      # Evaluation results
    └── performance_reports/     # Performance metrics
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20.0+
- MongoDB 7.0+
- ESP32-WROOM-32 development board
- Raspberry Pi 3 B+ (or higher)
- INMP441 I2S MEMS microphone

### 1. Setup Dataset

```bash
# Download dataset from Kaggle
kaggle datasets download -d mennaahmed23/baby-crying-dataset

# Unzip to dataset/raw/
unzip baby-crying-dataset.zip -d dataset/raw/

# Preprocess audio
python scripts/preprocess_audio.py

# Extract features
python scripts/feature_extraction.py

# Split dataset
python scripts/dataset_split.py --input dataset/processed/features --output dataset/split
```

### 2. Train Model

```bash
cd model

# Install dependencies
pip install -r requirements.txt

# Train model
python training/train.py

# Evaluate model
python training/evaluate.py
```

### 3. Setup Backend

```bash
cd backend

# Install dependencies
npm install

# Configure environment
cp .env.example .env
# Edit .env with your MongoDB URI

# Start server
npm start
```

### 4. Setup IoT Devices

#### ESP32 Setup
```bash
cd iot/deployment
# Follow flash_instructions.md to flash ESP32 firmware

# Configure WiFi in iot/config/device_config.json
# Upload firmware files to ESP32
```

#### Raspberry Pi Setup
```bash
# On Raspberry Pi
cd iot/firmware

# Install dependencies
pip install -r ../../model/requirements.txt

# Run inference engine
python inference_engine.py
```

### 5. Setup Mobile App

```bash
cd mobile_app/CryingSenseApp

# Install dependencies
npm install

# Start development server
npm start

# Run on device
npm run android  # or npm run ios
```

---

## 📖 Usage

### Training Custom Model

```bash
# Navigate to model directory
cd model

# Train with custom parameters
python training/train.py --epochs 50 --batch-size 32 --lr 0.001

# Validate model
python training/validate.py

# Generate confusion matrix
python training/evaluate.py
```

### Running Inference

```bash
# Using the inference script
python model/inference/predict.py \
  --model model/saved_models/cryingsense_cnn.pth \
  --audio path/to/cry_audio.wav \
  --probabilities
```

### API Endpoints

#### Cry Events
- `POST /api/cry-events` - Create new cry event
- `GET /api/cry-events` - Get all cry events
- `GET /api/cry-events/:id` - Get specific event
- `GET /api/cry-events/stats` - Get statistics

#### Classifications
- `POST /api/classifications` - Create classification
- `GET /api/classifications` - Get all classifications
- `GET /api/classifications/:id` - Get specific classification
- `GET /api/classifications/stats` - Get statistics

#### Devices
- `POST /api/devices` - Register device
- `GET /api/devices` - Get all devices
- `GET /api/devices/:id` - Get device info
- `PUT /api/devices/:id/status` - Update device status

---

## 📊 Model Performance

The CryingSense CNN model achieves:
- **Accuracy**: ~85-92% (varies by class)
- **Inference Time**: ~100-150ms on Raspberry Pi 3 B+
- **Model Size**: ~2-5 MB (quantized for edge deployment)

### Model Architecture

```
CryingSenseCNN (Depthwise Separable Convolutions)
├── Conv2D (4 → 16 channels)
├── MaxPool2D
├── Conv2D (16 → 32 channels)
├── MaxPool2D
├── Conv2D (32 → 64 channels)
├── MaxPool2D
├── Global Average Pooling
├── FC (64 → 128)
├── Dropout (0.3)
└── FC (128 → 5 classes)
```

---

## 🔐 Security Considerations

- ✅ All audio processing happens locally (no cloud upload)
- ✅ API endpoints use rate limiting (100 req/15min per IP)
- ✅ WiFi communication should use WPA2/WPA3 encryption
- ✅ Consider HTTPS for backend communication in production
- ✅ Implement device pairing/registration for IoT devices
- ✅ **Updated PyTorch to 2.6.0** to address critical vulnerabilities:
  - CVE: Heap buffer overflow (fixed in 2.2.0)
  - CVE: Use-after-free vulnerability (fixed in 2.2.0)
  - CVE: Remote code execution via torch.load (fixed in 2.6.0)

### Security Best Practices

1. **Model Loading**: When loading PyTorch models in production, use `weights_only=True`:
   ```python
   torch.load(filepath, map_location=device, weights_only=True)
   ```

2. **API Authentication**: Add JWT authentication for production:
   ```javascript
   // Add to backend/server.js
   const jwt = require('jsonwebtoken');
   ```

3. **HTTPS**: Use TLS/SSL certificates for backend communication

4. **Device Authentication**: Implement device tokens for IoT devices

5. **Regular Updates**: Keep all dependencies up to date with security patches

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Model Improvements**
   - Collect more training data
   - Experiment with different architectures
   - Implement transfer learning

2. **IoT Enhancements**
   - Add MQTT support for real-time streaming
   - Implement battery-powered ESP32 mode
   - Add multiple device support

3. **Mobile App Features**
   - Add analytics dashboard
   - Implement historical trends
   - Add caregiver notes

4. **Backend Features**
   - Add user authentication
   - Implement multi-device management
   - Add data export functionality

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: [Baby Crying Dataset](https://www.kaggle.com/datasets/mennaahmed23/baby-crying-dataset) by Menna Ahmed on Kaggle
- **PyTorch**: Facebook AI Research
- **Librosa**: Audio analysis library
- **Expo**: React Native development platform

---

## 📞 Support

For issues, questions, or contributions:
- 📧 Create an issue on GitHub
- 📖 Check the documentation in each component's directory
- 💬 Refer to individual README files for detailed setup instructions

---

## 🗺️ Roadmap

- [ ] Multi-language support for mobile app
- [ ] Cloud backup option (optional)
- [ ] Integration with smart home systems
- [ ] Voice assistant integration (Alexa, Google Home)
- [ ] Advanced analytics and insights
- [ ] Parent community features

---

**Made with ❤️ for better baby care**
