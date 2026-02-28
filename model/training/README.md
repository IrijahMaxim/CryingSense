# CryingSense Training Module

This folder contains scripts for training, validating, and evaluating the CryingSense CNN model.

## Scripts

| Script | Purpose |
|--------|---------|
| `train.py` | Train the CNN model with early stopping and learning rate scheduling |
| `validate.py` | Quick validation on the validation set |
| `evaluate.py` | Full evaluation with metrics, confusion matrix, and reports |

## Prerequisites

1. **Feature Extraction**: Run feature extraction first
   ```bash
   python scripts/feature_extraction.py
   ```

2. **Dataset Split** (recommended): Create reproducible train/val/eval splits
   ```bash
   python scripts/dataset_split.py
   ```

## Usage

### Training

```bash
python model/training/train.py
```

**Features:**
- Automatic GPU detection (falls back to CPU)
- Early stopping with patience=10
- Learning rate scheduling (ReduceLROnPlateau)
- AdamW optimizer with weight decay
- Data augmentation (noise, time shift, amplitude scaling)
- Training curves visualization
- Best model checkpoint saving

**Outputs:**
- `model/saved_models/cryingsense_cnn_best.pth` - Best model checkpoint
- `model/saved_models/training_history.json` - Training metrics history
- `model/saved_models/training_curves.png` - Loss/accuracy plots

### Validation

```bash
python model/training/validate.py
```

Quick validation showing classification report and confusion matrix.

### Evaluation

```bash
python model/training/evaluate.py
```

**Features:**
- Comprehensive metrics (accuracy, precision, recall, F1)
- Per-class performance analysis
- Confidence threshold analysis
- Inference time measurement

**Outputs:**
- `experiments/performance_reports/evaluation_results.json`
- `experiments/performance_reports/classification_report.txt`
- `experiments/confusion_matrices/confusion_matrix.png`

## Data Pipeline

```
dataset/dataset_split.json
         │
         ▼
┌────────────────────────────────────────────────┐
│  load_split_from_json()                        │
│  - Reads splits: train, val, eval              │
│  - Supports cleaned + raw feature directories  │
│  - Returns (mfcc_path, base_dir) tuples        │
└────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────┐
│  CryingSenseDataset                            │
│  - Loads MFCC, Mel, Chroma features            │
│  - Computes Delta MFCC                         │
│  - Creates 4-channel input tensor              │
│  - Optional data augmentation                  │
└────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────┐
│  CryingSenseCNN                                │
│  - 4-channel input (MFCC, Mel, Chroma, Delta)  │
│  - Depthwise separable convolutions            │
│  - 6 output classes (includes noise)           │
└────────────────────────────────────────────────┘
```

## Dataset Split Format

`dataset/dataset_split.json`:
```json
{
  "sources": ["cleaned", "raw"],
  "splits": {
    "train": {
      "belly_pain": ["cleaned:file1.npy", "raw:file2.npy"],
      "burp": [...],
      ...
    },
    "val": {...},
    "eval": {...}
  }
}
```

**Entry format:** `"source:filename.npy"` where source is `cleaned` or `raw`

## Feature Directories

```
dataset/processed/feature_extraction/
├── cleaned/
│   ├── mfcc/{class_name}/*.npy
│   ├── mel_spectrogram/{class_name}/*.npy
│   └── chroma/{class_name}/*.npy
└── raw/
    ├── mfcc/{class_name}/*.npy
    ├── mel_spectrogram/{class_name}/*.npy
    └── chroma/{class_name}/*.npy
```

## Model Architecture

- **Input**: 4 channels × H × W (feature dimensions)
- **Backbone**: Depthwise separable convolutions with batch norm
- **Optimizer**: AdamW with weight decay (1e-4)
- **Output**: 6 classes (5 cry types + noise)
- **Dropout**: 0.3 (configurable)

## Classes

| Index | Class Name | Description |
|-------|------------|-------------|
| 0 | belly_pain | Baby crying due to belly pain |
| 1 | burp | Baby needs to burp |
| 2 | discomfort | General discomfort |
| 3 | hunger | Baby is hungry |
| 4 | noise | Environmental/background noise (invisible class) |
| 5 | tired | Baby is tired |

> **Note:** The `noise` class is trained to help the model distinguish environmental sounds from actual cries. During inference, if the model predicts "noise", it returns `"no_cry_detected"` instead - making noise an "invisible" class that doesn't appear as a cry prediction.

## Fallback Behavior

If `dataset_split.json` is not found:
- Scripts fall back to random train/val split (80/20)
- Uses only the `cleaned` feature directory
- Prints warning message recommending JSON split creation

## Requirements

See `model/requirements.txt`:
- torch
- numpy
- scikit-learn
- matplotlib
- seaborn
- tqdm
