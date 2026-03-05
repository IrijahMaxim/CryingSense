# CryingSense Scripts

This folder contains data processing scripts for preparing the CryingSense dataset.

---

## Pipeline Overview

The scripts should be run in this order:

```
0. cleanup_dataset.py    →  (Optional) Clean old processed files
1. preprocess_audio.py   →  Clean and normalize raw audio files
2. feature_extraction.py →  Extract MFCC, Mel, Chroma features for training
3. visualize_dataset.py  →  (Optional) Generate visualizations & statistics
4. dataset_split.py      →  Split into train/val/test sets
```

**Note**: `visualize_dataset.py` can be run at any time after step 1 to create visualizations. It's independent from `feature_extraction.py`.

---

## Quick Start

```bash
# Activate virtual environment
cd "P:\VScode Lobby\CryingSense"
.\venv\Scripts\Activate.ps1

# Navigate to scripts folder
cd scripts

# Optional: Clean old processed data
python cleanup_dataset.py --confirm

# Run full pipeline
python preprocess_audio.py
python feature_extraction.py
python visualize_dataset.py      # Optional: visualize 3 samples per category
python dataset_split.py
```

---

## cleanup_dataset.py

**Purpose**: Removes all processed files before regenerating dataset.

### What It Does
- Deletes all files in `dataset/visualizations/`
- Deletes all files in `dataset/processed/cleaned/`
- Deletes all files in `dataset/processed/features/` (.npy files)
- Deletes all files in `dataset/processed/feature_extraction/` (legacy)
- Preserves folder structure
- Shows file count and size before deletion
- Requires confirmation (unless `--confirm` flag used)

### Usage

```bash
# With confirmation prompt
python cleanup_dataset.py

# Skip confirmation (careful!)
python cleanup_dataset.py --confirm
```

### When to Use
- Before reprocessing the entire dataset
- When preprocessing parameters have changed
- To free up disk space
- After adding new raw audio files

### Safety Features
- Shows what will be deleted before confirmation
- Displays file counts and sizes
- Preserves `dataset/raw/` (never touches source files)
- Keeps folder structure intact

---

## preprocess_audio.py

**Enhanced**: Now includes background noise augmentation.

Cleans and normalizes raw audio files for consistent processing.

### What It Does
- Loads raw audio from `dataset/raw/`
- Resamples to 16kHz
- Trims silence from beginning and end
- Normalizes volume levels
- Generates augmented versions:
  1. **_normalized**: Normalized only
  2. **_shifted**: Time-shifted (up to 20%)
  3. **_both**: Normalized + time-shifted
  4. **_noise**: Normalized + background noise (if noise folder exists)
  5. **_noise_shifted**: Normalized + background noise + time-shifted
- Saves cleaned audio to `dataset/processed/cleaned/`

### Background Noise Augmentation
- Automatically uses files from `dataset/raw/noise/` if available
- Mixes background noise at 2% factor
- Helps model learn to distinguish cries from background sounds
- Skipped if noise directory doesn't exist

### Usage

```bash
python preprocess_audio.py
```

### Output
- Cleaned audio files in `dataset/processed/cleaned/{category}/`
- 3 versions per file (without noise) or 5 versions (with noise)
- Maintains same directory structure as raw files

---

## feature_extraction.py

**Refactored**: Now focuses ONLY on extracting features for model training (no visualization).

### What It Does
- Extracts acoustic features from audio files:
  • **MFCC** (40 coefficients) - Mel-Frequency Cepstral Coefficients
  • **Mel Spectrogram** (128 bands) - Perceptually-weighted frequencies
  • **Chroma** (12 bins) - Pitch/harmonic content
- Saves features as `.npy` files for efficient model training
- Pads/crops features to consistent shape for batch processing

### Usage

```bash
# Extract from cleaned data (default)
python feature_extraction.py

# Extract from raw data
python feature_extraction.py --raw
```

### Output
- `.npy` feature files in `dataset/processed/features/{cleaned|raw}/`
  • `mfcc/{category}/`
  • `mel_spectrogram/{category}/`
  • `chroma/{category}/`

### Parameters
- Sample rate: 16000 Hz
- MFCC coefficients: 40
- Mel bands: 128
- Chroma bins: 12
- FFT size: 1024
- Hop length: 512
- Duration: 5.0 seconds

---

## visualize_dataset.py

**Enhanced**: Comprehensive visualization and statistical analysis tool.

### What It Does
- Creates multi-panel visualizations for audio files:
  1. **Waveform** - Time-domain amplitude
  2. **Spectrogram** - Time-frequency representation (STFT)
  3. **Mel Spectrogram** - Perceptually-weighted frequencies
  4. **MFCC** - Mel-Frequency Cepstral Coefficients
  5. **Chroma Features** - Pitch class representation
  
- Generates dataset statistics:
  • Class distribution (bar chart and pie chart)
  • Audio duration analysis (box plots)
  • Per-category statistics (count, duration, min/max)
  • Text summary report

### Usage

```bash
# Visualize 3 random samples per category (default)
python visualize_dataset.py

# Visualize all audio files (slow!)
python visualize_dataset.py --all

# Visualize 5 random samples per category
python visualize_dataset.py --samples 5

# Generate only statistics (no individual file visualizations)
python visualize_dataset.py --stats-only

# Use raw audio instead of cleaned
python visualize_dataset.py --raw
```

### Output
- Audio visualizations: `dataset/visualizations/{cleaned|raw}/{category}/`
- Statistics: `dataset/visualizations/{cleaned|raw}/dataset_statistics.png`
- Text summary: `dataset/visualizations/{cleaned|raw}/dataset_summary.txt`

### When to Use
- After preprocessing to verify audio quality
- To understand dataset distribution and characteristics
- Before training to identify potential issues
- For documentation and presentations

---

## dataset_split.py

**Enhanced**: Now supports custom split ratios and file count limits with interactive prompts.

Splits dataset into train/validation/test sets for model training.

### What It Does
- Groups files by recording session (prevents data leakage)
- Splits each class independently
- Default: 60% train / 20% val / 20% test
- Custom: Interactive prompt for any ratio **and file count limits**
- Saves split information to JSON file
- Supports cleaned and/or raw features

### Usage

```bash
# Split cleaned features with default ratios (60/20/20, all files)
python dataset_split.py

# Interactive custom split ratios and file limits
python dataset_split.py --custom-split

# Split raw features only
python dataset_split.py --raw-only

# Split both cleaned and raw features
python dataset_split.py --all

# All cleaned features + only noise from raw
python dataset_split.py --noise-raw

# Custom split with raw data
python dataset_split.py --raw-only --custom-split
```

### Custom Split Example
```
$ python dataset_split.py --custom-split

==========================================================
Custom Split Configuration
==========================================================

Default split: 60% train, 20% val, 20% test

Options:
  1. Use default split (60/20/20) with all files
  2. Enter custom configuration
==========================================================

Choose option (1 or 2): 2

----------------------------------------------------------
Enter custom split ratios (percentages)
----------------------------------------------------------
Training set percentage (e.g., 70): 70
Validation set percentage (e.g., 15): 15
Test set percentage (e.g., 15): 15

✓ Custom split: 70.0% train, 15.0% val, 15.0% test

----------------------------------------------------------
File Count Limits (Optional)
----------------------------------------------------------

Limit the number of files used per class?
  1. Use all available files
  2. Set a uniform limit for all classes
  3. Set individual limits per class

Choose option (1, 2, or 3): 2

Enter file limit per class (e.g., 100): 50

✓ Using 50 files per class

==========================================================
Configuration Summary
==========================================================
Split ratios: 70.0% train, 15.0% val, 15.0% test
File limits: 50 files per class
==========================================================

Confirm configuration? (yes/no): yes
```

### File Limit Options

1. **Use all files**: No limits, use entire dataset
2. **Uniform limit**: Same limit for all classes (e.g., 50 files per class)
3. **Individual limits**: Custom limit per class
   ```
   belly_pain: 100
   burp: 50
   discomfort: 80
   hunger: 100
   tired: 60
   noise: 200
   speech: 150
   ```

### Benefits of File Limits
- **Faster development**: Test with small subsets
- **Balanced datasets**: Equalize underrepresented classes
- **Memory management**: Work within hardware constraints
- **Quick experimentation**: Iterate faster with smaller datasets

### Input/Output
| Option | Input | Output |
|--------|-------|--------|
| Default | `dataset/processed/features/cleaned/` | `dataset/dataset_split.json` |
| `--raw-only` | `dataset/processed/features/raw/` | `dataset/dataset_split.json` |
| `--all` | Both cleaned and raw | `dataset/dataset_split.json` |
| `--noise-raw` | Cleaned + raw noise only | `dataset/dataset_split.json` |

### Split Ratios
- **Default**: Train 60% / Val 20% / Test 20%
- **Custom**: Any ratio that sums to 100%
- Use `--custom-split` flag for interactive prompt

### Output JSON Structure
```json
{
  "sources": ["cleaned"],
  "splits": {
    "train": {
      "belly_pain": ["file1.npy", "file2.npy", ...],
      "burp": [...],
      ...
    },
    "val": {...},
    "test": {...}
  },
  "statistics": {
    "train": {"belly_pain": 240, "burp": 170, ...},
    "val": {...},
    "test": {...},
    "total": {...}
  },
  "config": {
    "train_ratio": 0.80,
    "val_ratio": 0.10,
    "test_ratio": 0.10,
    "random_seed": 42,
    "classes": ["belly_pain", "burp", "discomfort", "hunger",
                 "tired", "noise", "speech"]
  }
}
```

---

## Summary: Workflow Comparison

### Two Independent Paths:

**1. Feature Extraction for Training** (Required for model training)
```
preprocess_audio.py → feature_extraction.py → dataset_split.py
```
- Extracts features as `.npy` files
- Ready for model training
- No visualizations created

**2. Visualization & Analysis** (Optional, for understanding data)
```
preprocess_audio.py → visualize_dataset.py
```
- Creates PNG visualizations
- Generates statistics and reports
- Independent from feature extraction
- Can be run anytime after preprocessing

### Benefits of Separation:
- **Faster feature extraction**: No time spent on visualization
- **Flexible visualization**: Run only when needed, with various options
- **Clear separation of concerns**: Training preparation vs. data exploration
- **Disk space efficiency**: Generate visualizations only for samples you need

---

## Directory Structure

### Before Processing
```
dataset/
├── raw/
│   ├── belly_pain/
│   │   ├── cry_001.wav
│   │   └── ...
│   ├── burp/
│   ├── discomfort/
│   ├── hunger/
│   ├── tired/
│   └── noise/
└── processed/
    └── (empty)
```

### After Processing
```
dataset/
├── raw/
│   └── (original files)
├── processed/
│   ├── cleaned/
│   │   ├── belly_pain/
│   │   └── ...
│   └── feature_extraction/
│       ├── cleaned/
│       │   ├── mfcc/
│       │   ├── mel_spectrogram/
│       │   └── chroma/
│       └── raw/
│           └── ...
└── dataset_split.json
```

---

## Class Labels

| Class | Description | Count (typical) |
|-------|-------------|-----------------|
| `belly_pain` | Stomach discomfort cry | ~150 |
| `burp` | Burping sounds | ~100 |
| `discomfort` | General discomfort | ~180 |
| `hunger` | Hungry cry | ~200 |
| `tired` | Sleepy/tired cry | ~170 |
| `noise` | Non-cry sounds (optional) | ~120 |

---

## Troubleshooting

### "Input directory not found"
- Ensure raw audio files exist in `dataset/raw/{class_name}/`
- Run preprocessing before feature extraction

### Memory errors during feature extraction
- Process in smaller batches
- Close other applications

### Uneven class distribution after split
- This is normal; the split preserves session grouping
- Consider collecting more data for underrepresented classes

### noisereduce import error
```bash
pip install noisereduce
```

---

## Dependencies

- **librosa** - Audio loading and feature extraction
- **numpy** - Numerical operations
- **scipy** - WAV file I/O
- **noisereduce** - Noise reduction
- **tqdm** - Progress bars

Install all:
```bash
pip install librosa numpy scipy noisereduce tqdm
```

---

## Notes

- Audio is standardized to **16kHz mono, 5 seconds**
- Features are saved as **NumPy arrays (.npy)**
- Session-based splitting prevents **data leakage**
- Random seed (42) ensures **reproducible splits**
