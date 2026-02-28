# CryingSense Scripts

This folder contains data processing scripts for preparing the CryingSense dataset.

---

## Pipeline Overview

The scripts should be run in this order:

```
1. preprocess_audio.py  →  Clean raw audio files
2. feature_extraction.py →  Extract MFCC, Mel, Chroma features  
3. dataset_split.py     →  Split into train/val/test sets
```

---

## Quick Start

```bash
# Activate virtual environment
cd "P:\VScode Lobby\CryingSense"
.\venv\Scripts\Activate.ps1

# Navigate to scripts folder
cd scripts

# Run full pipeline
python preprocess_audio.py
python feature_extraction.py
python dataset_split.py
```

---

## preprocess_audio.py

Cleans and normalizes raw audio files for training.

### What It Does
- Loads raw `.wav` files from `dataset/raw/`
- Trims silence from beginning and end
- Applies noise reduction
- Normalizes amplitude to [-1, 1]
- Pads or trims to exactly 5 seconds
- Saves to `dataset/processed/cleaned/`

### Usage

```bash
# Process all raw audio
python preprocess_audio.py
```

### Input/Output
| Direction | Path |
|-----------|------|
| Input | `dataset/raw/{class_name}/*.wav` |
| Output | `dataset/processed/cleaned/{class_name}/*.wav` |

### Parameters (in code)
| Parameter | Value | Description |
|-----------|-------|-------------|
| `sample_rate` | 16000 Hz | Target sample rate |
| `duration` | 5.0 seconds | Fixed audio length |
| `top_db` | 20 dB | Silence threshold |

---

## feature_extraction.py

Extracts acoustic features from audio files.

### What It Does
- Extracts MFCC features (40 coefficients)
- Extracts Mel Spectrogram (128 bands)
- Extracts Chroma features (12 bins)
- Saves each feature as `.npy` file

### Usage

```bash
# Extract from cleaned data only (default)
python feature_extraction.py

# Extract from raw data only
python feature_extraction.py --raw-only

# Extract from both cleaned and raw data
python feature_extraction.py --include-raw
```

### Options
| Option | Description |
|--------|-------------|
| `--include-raw` | Also extract features from raw dataset |
| `--raw-only` | Only extract from raw dataset (skip cleaned) |

### Input/Output
| Dataset | Input | Output |
|---------|-------|--------|
| Cleaned | `dataset/processed/cleaned/` | `dataset/processed/feature_extraction/cleaned/` |
| Raw | `dataset/raw/` | `dataset/processed/feature_extraction/raw/` |

### Output Structure
```
feature_extraction/
├── cleaned/
│   ├── mfcc/
│   │   ├── belly_pain/
│   │   ├── burp/
│   │   └── ...
│   ├── mel_spectrogram/
│   │   └── ...
│   └── chroma/
│       └── ...
└── raw/
    ├── mfcc/
    ├── mel_spectrogram/
    └── chroma/
```

### Feature Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_mfcc` | 40 | MFCC coefficients |
| `n_mels` | 128 | Mel frequency bands |
| `n_chroma` | 12 | Chroma bins |
| `n_fft` | 1024 | FFT window size |
| `hop_length` | 512 | Samples between frames |

---

## dataset_split.py

Splits dataset into training, validation, and test sets.

### What It Does
- Groups files by recording session (prevents data leakage)
- Splits each class independently
- Saves split information to JSON file
- Prefixes filenames with source (`cleaned:` or `raw:`) when combining both

### Usage

```bash
# Split cleaned data only (default)
python dataset_split.py

# Split raw data only
python dataset_split.py --raw-only

# Split both cleaned and raw data
python dataset_split.py --all

# All cleaned data + only noise from raw
python dataset_split.py --noise-raw
```

### Options
| Option | Description |
|--------|-------------|
| (default) | Split cleaned dataset only |
| `--raw-only` | Only split raw dataset |
| `--all` | Split both cleaned + raw datasets |
| `--noise-raw` | All cleaned data + only noise class from raw |

### Split Ratios
| Set | Percentage |
|-----|------------|
| Training | 80% |
| Validation | 10% |
| Test | 10% |

### Output
All modes output to: `dataset/dataset_split.json`

### JSON Structure
```json
{
  "sources": ["cleaned", "raw"],
  "splits": {
    "train": {
      "belly_pain": ["cleaned:file1.wav", "raw:file2.wav", ...],
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
    "classes": ["belly_pain", "burp", "discomfort", "hunger", "tired", "noise"]
  }
}
```

**Note:** Filenames are prefixed with `cleaned:` or `raw:` to indicate their source.

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
