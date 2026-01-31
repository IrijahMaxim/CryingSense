# CryingSense Dataset Information

## Overview
The CryingSense dataset consists of infant cry audio recordings categorized into five distinct classes based on the baby's needs. The dataset is used to train a Convolutional Neural Network (CNN) for automated cry classification.

## Dataset Source
- **Source**: [Baby Crying Dataset on Kaggle](https://www.kaggle.com/datasets/mennaahmed23/baby-crying-dataset)
- **License**: Please refer to the Kaggle dataset page for license information

## Classes
The dataset contains the following five classes:

1. **belly_pain** - Cries indicating the baby is experiencing belly pain or discomfort
2. **burp** - Cries indicating the baby needs to burp
3. **discomfort** - General discomfort cries (not specifically hunger, tiredness, or pain)
4. **hunger** - Cries indicating the baby is hungry
5. **tired** - Cries indicating the baby is tired and needs sleep

## Directory Structure

```
dataset/
├── raw/
│   ├── belly_pain/    # Raw audio files (.wav)
│   ├── burp/          # Raw audio files (.wav)
│   ├── discomfort/    # Raw audio files (.wav)
│   ├── hunger/        # Raw audio files (.wav)
│   └── tired/         # Raw audio files (.wav)
│
├── processed/
│   ├── cleaned/       # Preprocessed audio files (noise reduced, normalized)
│   │   ├── belly_pain/
│   │   ├── burp/
│   │   ├── discomfort/
│   │   ├── hunger/
│   │   └── tired/
│   │
│   └── features/      # Extracted features (.npy)
│       ├── belly_pain/
│       ├── burp/
│       ├── discomfort/
│       ├── hunger/
│       └── tired/
│
└── dataset_info.md    # This file
```

## Audio Specifications

### Raw Audio
- **Format**: WAV (Waveform Audio File Format)
- **Sample Rate**: 16000 Hz (16 kHz)
- **Duration**: Variable length (typically 1-10 seconds)
- **Channels**: Mono

### Processed Audio
- **Format**: WAV
- **Sample Rate**: 16000 Hz
- **Duration**: Fixed 5 seconds (padded or trimmed)
- **Channels**: Mono
- **Preprocessing**: Noise reduction, normalization, silence trimming

## Feature Extraction

The audio files are processed to extract multiple acoustic features:

### Feature Types
1. **MFCC (Mel-Frequency Cepstral Coefficients)**
   - Dimensions: 40 coefficients
   - Captures spectral envelope and timbral qualities

2. **Mel Spectrogram**
   - Dimensions: 128 mel bands
   - Represents time-frequency energy distribution
   - Converted to dB scale

3. **Chroma Features**
   - Dimensions: 12 chroma bins
   - Captures pitch content and harmonic structure

4. **SpecAugment**
   - Applied to Mel Spectrogram for data augmentation
   - Frequency and time masking

### Feature Dimensions
- **Input Shape**: (4, height, width)
  - 4 channels: [MFCC, Mel Spectrogram, SpecAugment, Chroma]
  - Height: Variable (40 for MFCC, 128 for Mel, 12 for Chroma)
  - Width: 216 time steps (fixed)

### Saved Format
- **File Type**: NumPy array (.npy)
- **Data Type**: float32
- **Shape**: (4, height, 216)

## Data Augmentation

To increase dataset diversity and model robustness, the following augmentation techniques are applied:

1. **Time Stretching** (0.8x - 1.2x)
2. **Pitch Shifting** (-2 to +2 semitones)
3. **Background Noise Addition**
4. **Volume Scaling**
5. **Time Shifting**
6. **SpecAugment** (frequency and time masking)

## Dataset Statistics

### Expected Distribution
- Each class should have a balanced number of samples
- Typical dataset size: 500-2000 samples per class (before augmentation)
- After augmentation: 2x-4x increase in samples

### Recommended Split
- **Training**: 70%
- **Validation**: 15%
- **Test**: 15%

## Preprocessing Pipeline

```
Raw Audio (.wav)
    ↓
Load & Resample (16 kHz)
    ↓
Trim Silence
    ↓
Noise Reduction
    ↓
Normalize Amplitude
    ↓
Pad/Trim to 5 seconds
    ↓
Data Augmentation (optional)
    ↓
Feature Extraction
    ↓
Convert to dB Scale
    ↓
Pad/Crop to Fixed Dimensions
    ↓
Save as NumPy Array (.npy)
```

## Usage

### 1. Download Dataset
```bash
# Download from Kaggle
kaggle datasets download -d mennaahmed23/baby-crying-dataset
unzip baby-crying-dataset.zip -d dataset/raw/
```

### 2. Preprocess Audio
```bash
python scripts/preprocess_audio.py
```

### 3. Extract Features
```bash
python scripts/feature_extraction.py
```

### 4. Split Dataset
```bash
python scripts/dataset_split.py --input dataset/processed/features --output dataset/split
```

## Notes

- The `noise/` folder in raw dataset contains environmental sounds and is **not** used for cry classification
- Audio files should be at least 0.5 seconds long for meaningful feature extraction
- Silence-only or very low-amplitude recordings are filtered during preprocessing
- All features are normalized to improve model convergence

## Citation

If you use this dataset, please cite the original source:
```
Menna Ahmed. (2023). Baby Crying Dataset. Kaggle.
https://www.kaggle.com/datasets/mennaahmed23/baby-crying-dataset
```
