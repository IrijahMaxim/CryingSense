# CryingSense Dataset Information

## Overview
This document describes the CryingSense infant cry audio dataset, including its structure, preprocessing parameters, and feature extraction settings.

## Dataset Classes
The dataset contains six classes of infant sounds:

1. **belly_pain** - Cries indicating abdominal discomfort or pain
2. **burp** - Sounds related to the need to burp
3. **discomfort** - General discomfort cries (e.g., temperature, position)
4. **hunger** - Hunger-related crying
5. **tired** - Fatigue or sleepiness cries
6. **noise** - Background noise and non-cry sounds (for filtering/rejection)

## Directory Structure

```
dataset/
├── raw/                                  # Original unmodified recordings
│   ├── belly_pain/
│   ├── burp/
│   ├── discomfort/
│   ├── hunger/
│   ├── tired/
│   └── noise/
│
├── processed/
│   ├── cleaned/                          # Preprocessed audio files
│   │   ├── belly_pain/
│   │   ├── burp/
│   │   ├── discomfort/
│   │   ├── hunger/
│   │   ├── tired/
│   │   └── noise/
│   │
│   └── feature_extraction/
│       ├── cleaned/                      # Features from cleaned audio
│       │   ├── mfcc/
│       │   │   ├── belly_pain/
│       │   │   ├── burp/
│       │   │   ├── discomfort/
│       │   │   ├── hunger/
│       │   │   ├── tired/
│       │   │   └── noise/
│       │   │
│       │   ├── mel_spectrogram/
│       │   │   ├── belly_pain/
│       │   │   ├── burp/
│       │   │   ├── discomfort/
│       │   │   ├── hunger/
│       │   │   ├── tired/
│       │   │   └── noise/
│       │   │
│       │   └── chroma/
│       │       ├── belly_pain/
│       │       ├── burp/
│       │       ├── discomfort/
│       │       ├── hunger/
│       │       ├── tired/
│       │       └── noise/
│       │
│       └── raw/                          # Features from raw audio (optional)
│           ├── mfcc/
│           ├── mel_spectrogram/
│           └── chroma/
│
└── dataset_split.json                    # Train/validation/test split info
```

## Audio Preprocessing Parameters

### Input Specifications
- **Source Directory**: `dataset/raw/`
- **Output Directory**: `dataset/processed/cleaned/`
- **File Format**: WAV (PCM 16-bit)

### Processing Pipeline
1. **Loading & Resampling**
   - Target Sample Rate: 16,000 Hz
   - Original sample rates are resampled to ensure consistency

2. **Silence Trimming**
   - Top dB Threshold: 20 dB
   - Removes silence from beginning and end while preserving cry audio

3. **Noise Reduction**
   - Algorithm: Spectral gating using noisereduce library
   - Reduces background noise and environmental interference

4. **Amplitude Normalization**
   - Method: Peak normalization to [-1, 1] range
   - Ensures consistent energy levels across all samples

5. **Duration Standardization**
   - Target Duration: 5.0 seconds (80,000 samples at 16kHz)
   - Trimming: Center crop for longer audio
   - Padding: Zero-padding for shorter audio

6. **Output Format**
   - WAV format, 16-bit PCM
   - Sample Rate: 16,000 Hz
   - Filename: Maintains 1:1 mapping with raw files

### Data Augmentation
- **Training-Time Only**: Augmentation is applied during model training, NOT stored permanently
- Techniques used during training:
  - Time stretching (0.8x - 1.2x)
  - Pitch shifting (±2 semitones)
  - Background noise addition
  - Volume scaling
  - SpecAugment (frequency and time masking)

## Feature Extraction Parameters

### Input Specifications
- **Source Directory**: `dataset/processed/cleaned/`
- **Output Base Directory**: `dataset/processed/feature_extraction/cleaned/`
- **File Format**: NumPy arrays (.npy)

### MFCC (Mel-Frequency Cepstral Coefficients)
- **Number of Coefficients**: 40
- **FFT Size**: 1024
- **Hop Length**: 512 samples
- **Window**: Hann window (default)
- **Output Shape**: (40, time_steps)
- **Storage**: `feature_extraction/cleaned/mfcc/`

### Mel Spectrogram
- **Number of Mel Bands**: 128
- **FFT Size**: 1024
- **Hop Length**: 512 samples
- **Conversion**: Power to dB scale (ref=max)
- **Output Shape**: (128, time_steps)
- **Storage**: `feature_extraction/cleaned/mel_spectrogram/`

### Chroma Features
- **Number of Chroma Bins**: 12 (one per semitone)
- **FFT Size**: 1024
- **Hop Length**: 512 samples
- **Type**: Short-Time Fourier Transform (STFT) based
- **Output Shape**: (12, time_steps)
- **Storage**: `feature_extraction/cleaned/chroma/`

### Time Steps Calculation
- Sample Rate: 16,000 Hz
- Duration: 5.0 seconds
- Hop Length: 512 samples
- **Expected Time Steps**: ⌈(16000 × 5) / 512⌉ = 157 frames

### Feature Dimensions
All features are padded or cropped to consistent dimensions:
- MFCC: **(40, 157)**
- Mel Spectrogram: **(128, 157)**
- Chroma: **(12, 157)**

## Dataset Splits

### Split Ratios
- **Training Set**: 70%
- **Validation Set**: 15%
- **Test Set**: 15%

### Split Strategy
- Splitting is performed at the **session level** (not individual files)
- Files from the same recording session/infant are kept together
- Prevents data leakage between train/val/test sets
- Random seed: 42 (for reproducibility)

### Split Information
- Split assignments are stored in `dataset/dataset_split.json`
- Contains file lists for each class and split
- Includes statistics (sample counts per class/split)

## Dataset Statistics

### Raw Dataset (Example - Update after processing)
| Class        | Number of Files | Duration (approx.) |
|--------------|----------------:|-------------------:|
| belly_pain   |             750 |          ~62.5 min |
| burp         |             247 |          ~20.6 min |
| discomfort   |             750 |          ~62.5 min |
| hunger       |             750 |          ~62.5 min |
| tired        |             752 |          ~62.7 min |
| noise        |               0 |              0 min |
| **Total**    |        **3,249**|     **~270.8 min** |

*Note: Update this table after adding noise samples*

## Naming Conventions

### File Naming
- **Format**: Consistent with source files
- **Examples**:
  - `357c_part1.wav` → `357c_part1.npy`
  - `burping_aug_701.wav` → `burping_aug_701.npy`
- **Mapping**: Each processed file maintains exact name correspondence with raw source

### Session Identification
Files from the same recording session share a base identifier:
- `357c_part1.wav`, `357c_part2.wav` → Session: `357c`
- `burping_aug_701.wav`, `burping_aug_702.wav` → Session: `burping`

## Processing Scripts

### 1. Preprocessing Script
- **Location**: `scripts/preprocess_audio.py`
- **Function**: Cleans and standardizes raw audio
- **Usage**: `python scripts/preprocess_audio.py`

### 2. Feature Extraction Script
- **Location**: `scripts/feature_extraction.py`
- **Function**: Extracts MFCC, Mel, and Chroma features
- **Usage**: `python scripts/feature_extraction.py`

### 3. Dataset Split Script
- **Location**: `scripts/dataset_split.py`
- **Function**: Creates train/val/test splits
- **Usage**: `python scripts/dataset_split.py`

## Reproducibility

### Version Control
- All preprocessing parameters are documented in this file
- Processing scripts are version controlled
- Random seeds are fixed (seed=42) for deterministic splits

### Reprocessing
To reprocess the dataset:
1. Never modify files in `dataset/raw/` (preserve originals)
2. Delete `dataset/processed/` directory
3. Run preprocessing: `python scripts/preprocess_audio.py`
4. Run feature extraction: `python scripts/feature_extraction.py`
5. Create splits: `python scripts/dataset_split.py`

## Hardware Constraints

### Target Device: Raspberry Pi 3B+
- **RAM**: 1 GB
- **Storage**: 32 GB SD card
- **Constraints**:
  - Features are optimized for size and inference speed
  - Model must fit within memory constraints
  - Inference latency target: <500ms per sample

### Feature Size Estimates
Per audio file (5 seconds):
- MFCC: 40 × 157 × 4 bytes = ~25 KB
- Mel Spectrogram: 128 × 157 × 4 bytes = ~80 KB
- Chroma: 12 × 157 × 4 bytes = ~7.5 KB
- **Total per file**: ~112.5 KB

## Quality Assurance

### Validation Checks
- [ ] All raw files have corresponding cleaned files
- [ ] All cleaned files are exactly 5 seconds
- [ ] All feature files have consistent shapes
- [ ] No missing labels or corrupted data
- [ ] Train/val/test splits have no overlap
- [ ] Class balance is reasonable for training

### Periodic Audits
- Review processed audio samples manually
- Check spectrograms for proper preprocessing
- Verify feature distributions across classes
- Monitor for data quality issues

## Updates and Versioning

### Version History
- **v1.0** (2024-02-04): Initial dataset structure and processing pipeline

### Adding New Data
When adding new cry recordings:
1. Place raw files in appropriate `dataset/raw/` subdirectory
2. Run full processing pipeline (preprocess → extract → split)
3. Update statistics in this document
4. Retrain model with updated dataset
5. Version control changes to dataset metadata

## Notes

- The `noise` class is optional for model training but useful for rejection/filtering
- Augmentation should ONLY be applied during training, not stored permanently
- Always maintain 1:1 mapping between raw, cleaned, and feature files
- Document any changes to processing parameters in this file
