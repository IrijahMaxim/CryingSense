# Model Architectures

Core model definitions and export utilities for CryingSense.

## Files

| File | Description |
|------|-------------|
| `cnn_model.py` | CryingSenseCNN architecture |
| `export_model.py` | Export to deployment formats |

---

## CryingSenseCNN

Lightweight CNN with depthwise separable convolutions for infant cry classification.

### Architecture

```
Input: (batch, 4, 128, 216)
    │
    ├─ Conv2d(4→16) + BatchNorm + ReLU + MaxPool
    ├─ DepthwiseSeparable(16→32) + BatchNorm + ReLU + MaxPool
    ├─ DepthwiseSeparable(32→64) + BatchNorm + ReLU + MaxPool
    ├─ AdaptiveAvgPool2d(4, 4)
    ├─ Flatten
    ├─ Linear(→128) + ReLU + Dropout(0.5)
    └─ Linear(128→num_classes)
    
Output: (batch, num_classes)
```

### Usage

```python
from model.models.cnn_model import CryingSenseCNN

# Create model
model = CryingSenseCNN(num_classes=6)

# Input: 4-channel feature tensor
# Channel 0: MFCC (40 coefficients, padded to 128)
# Channel 1: Mel spectrogram (128 bins)
# Channel 2: Chroma (12 bins, padded to 128)
# Channel 3: Delta MFCC

import torch
x = torch.randn(1, 4, 128, 216)
output = model(x)  # Shape: (1, 6)
```

### Key Features

- **Depthwise Separable Convolutions:** Reduces parameters while maintaining accuracy
- **Lazy FC Layer:** `_fc1` is initialized on first forward pass to handle variable input sizes
- **Adaptive Pooling:** Fixed spatial output (4x4) regardless of input dimensions
- **Dropout:** 50% dropout for regularization

---

## Export Model

Export trained models to deployment formats with auto-versioning.

### Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| TorchScript | `.torchscript.pt` | Production, mobile |
| ONNX | `.onnx` | Cross-platform |
| Quantized | `_quantized.pth` | Edge devices |
| State Dict | `.pth` | Standard PyTorch |

### Usage

```bash
# Export with auto-versioning (creates cryingsense_model_beta_v1, v2, etc.)
python -m model.models.export_model

# Specify version
python -m model.models.export_model --version 3

# Select formats
python -m model.models.export_model --formats torchscript,quantized

# Custom base name
python -m model.models.export_model --base-name cryingsense_prod
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `saved_models/cryingsense_cnn_best.pth` | Source checkpoint |
| `--output-dir` | `saved_models/exported` | Output directory |
| `--num-classes` | 6 | Number of classes |
| `--input-shape` | `1,4,128,216` | Input tensor shape |
| `--formats` | `torchscript,onnx,quantized` | Export formats |
| `--version` | auto | Version number |
| `--base-name` | `cryingsense_model_beta` | Base name prefix |

### Versioning

- Auto-detects next version by scanning output directory
- Creates: `{base_name}_v{number}.{format}`
- Example: `cryingsense_model_beta_v1.torchscript.pt`

---

## Model Specs

| Metric | Value |
|--------|-------|
| Parameters | ~12,410 |
| Size (fp32) | ~0.05 MB |
| Input Shape | (B, 4, 128, 216) |
| Output | 6 classes |
