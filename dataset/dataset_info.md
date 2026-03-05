# CryingSense Dataset Information

## Overview
This document describes the CryingSense infant cry audio dataset, including its structure, preprocessing parameters, and feature extraction settings.

## Dataset Classes
The dataset contains seven classes of infant sounds and non-cry audio:

### Cry Types (Model will alert)
1. **belly_pain** - Cries indicating abdominal discomfort or pain
2. **burp** - Sounds related to the need to burp
3. **discomfort** - General discomfort cries (e.g., temperature, position)
4. **hunger** - Hunger-related crying
5. **tired** - Fatigue or sleepiness cries

### Non-Cry Sounds (Model will ignore)
6. **noise** - Background noise and non-cry sounds (e.g., vacuum, TV, toys)
7. **speech** - Human speech and baby babbling/cooing (treated as noise by model)

**Important**: The model treats **speech** samples the same as **noise** - both are classified as sounds to ignore. This ensures the system only alerts on actual crying, not when parents are talking or baby is babbling.

## Directory Structure

```
dataset/
├── raw/                                  # Original unmodified recordings
│   ├── belly_pain/
│   ├── burp/
│   ├── discomfort/
│   ├── hunger/
│   ├── tired/
│   ├── noise/
│   └── speech/                           # Human speech samples (treated as noise)
│
├── processed/
│   ├── cleaned/                          # Preprocessed audio files
│   │   ├── belly_pain/
│   │   ├── burp/
│   │   ├── discomfort/
│   │   ├── hunger/
│   │   ├── tired/
│   │   ├── noise/
│   │   └── speech/
│   │
│   └── features/
│       ├── mfcc/
│       │   ├── belly_pain/
│       │   ├── burp/
│       │   ├── discomfort/
│       │   ├── hunger/
│       │   ├── tired/
│       │   ├── noise/
│       │   └── speech/
│       │
│       ├── mel_spectrogram/
│       │   ├── belly_pain/
│       │   ├── burp/
│       │   ├── discomfort/
│       │   ├── hunger/
│       │   ├── tired/
│       │   ├── noise/
│       │   └── speech/
│       │
│       └── chroma/
│           ├── belly_pain/
│           ├── burp/
│           ├── discomfort/
│           ├── hunger/
│           ├── tired/
│           ├── noise/
│           └── speech/
│
├── visualizations/                       # Generated visualizations
│   ├── cleaned/                          # Visualizations from cleaned audio
│   │   ├── belly_pain/
│   │   ├── burp/
│   │   ├── discomfort/
│   │   ├── hunger/
│   │   ├── tired/
│   │   ├── noise/
│   │   ├── speech/
│   │   ├── dataset_statistics.png        # Dataset overview statistics
│   │   └── dataset_summary.txt           # Text summary of dataset
│   └── raw/                              # Visualizations from raw audio (optional)
│
└── dataset_split.json                    # Train/validation/test split info
```

## Processing Pipeline

The dataset undergoes several processing stages:

### 1. Audio Preprocessing (`preprocess_audio.py`)
Cleans and normalizes raw audio files:
- **Input**: `dataset/raw/{category}/`
- **Output**: `dataset/processed/cleaned/{category}/`
- Resamples to 16kHz, trims silence, normalizes volume

### 2. Feature Extraction (`feature_extraction.py`)
Extracts acoustic features for model training:
- **Input**: `dataset/processed/cleaned/` (or `dataset/raw/` with `--raw` flag)
- **Output**: `dataset/processed/features/{cleaned|raw}/{mfcc|mel_spectrogram|chroma}/{category}/`
- Saves features as `.npy` files for efficient loading during training
- **Note**: This script focuses ONLY on feature extraction, not visualization

### 3. Visualization (`visualize_dataset.py`)
Creates comprehensive visualizations and statistics:
- **Input**: `dataset/processed/cleaned/` (or `dataset/raw/` with `--raw` flag)
- **Output**: `dataset/visualizations/{cleaned|raw}/`
- Generates audio feature plots (waveform, spectrogram, mel, MFCC, chroma)
- Creates dataset statistics (class distribution, duration analysis)
- Options: visualize all files, sample files, or statistics only

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
| speech       |               0 |              0 min |
| **Total**    |        **3,249**|     **~270.8 min** |

*Note: Update this table after adding noise and speech samples*

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

## Speech Category and Model Behavior

### Purpose of Speech Category
The `speech` category contains human speech and baby babbling/cooing samples that the model should **ignore** (not trigger alerts). This includes:

- Adult speech (parents talking, conversations)
- Baby babbling, cooing, laughing (non-crying vocalizations)
- Children talking
- TV/radio speech
- Phone conversations
- Singing and humming

### Model Training Approach
**Speech is treated identically to environmental noise:**

1. **During Training**: Both `speech` and `noise` samples are labeled as "non-cry" classes
2. **During Inference**: Model learns to distinguish:
   - **Cry types** (belly_pain, burp, discomfort, hunger, tired) → ✅ Alert
   - **Non-cry sounds** (noise, speech) → ❌ No alert

3. **Binary Classification View**:
   ```
   Is Crying?
   ├── YES → Classify cry type (belly_pain, burp, etc.)
   └── NO → Ignore (noise or speech)
   ```

### Model Architecture Considerations
The CNN model can be trained to:

**Option A: Multi-class with 7 classes**
- Classes: belly_pain, burp, discomfort, hunger, tired, noise, speech
- Both noise and speech are ignored classes during deployment

**Option B: Multi-class with 6 classes (speech merged into noise)**
- Classes: belly_pain, burp, discomfort, hunger, tired, noise
- Speech samples are simply labeled as "noise" during training
- Simpler model, same practical outcome

**Recommended**: Option A for better interpretability and debugging

### Why Speech Samples Matter
Without speech training:
- ❌ Model might misclassify parent talking as crying
- ❌ Baby babbling might trigger false alarms
- ❌ TV/radio could cause spurious alerts

With speech training:
- ✅ Model ignores conversations
- ✅ Baby babbling doesn't cause alerts
- ✅ Only actual crying triggers the system

### Adding Speech Samples
See `dataset/raw/speech/README.md` for:
- How to add speech samples
- Recommended sources (LibriSpeech, Common Voice, etc.)
- Recording guidelines
- Quality requirements

### Dataset Balance Recommendations
- **Cry samples**: 70-80% of dataset
- **Noise samples**: 10-15% of dataset
- **Speech samples**: 10-15% of dataset

This ensures the model:
1. Learns cry patterns well (majority class)
2. Rejects environmental noise
3. Ignores human speech and babbling

Avoid overwhelming the model with non-cry samples (keep cry samples as majority).

