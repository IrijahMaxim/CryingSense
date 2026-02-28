# CryingSense Model Exports

Exported models with auto-incrementing versioning scheme.

## Versioning

Models are exported with naming: `cryingsense_model_beta_v{number}`

- **v1, v2, ...** - Auto-incremented based on existing exports
- Each export creates multiple format variants

## Export Formats

| Format | File Extension | Use Case |
|--------|---------------|----------|
| PyTorch | `.pth` | Standard PyTorch loading with metadata |
| TorchScript | `.torchscript.pt` | Production deployment, mobile |
| Quantized | `_quantized.pth` | Edge devices, reduced size |
| ONNX | `.onnx` | Cross-platform (requires onnx package) |

## Usage

### Export a New Version

```bash
# From project root
python -m model.models.export_model

# With specific options
python -m model.models.export_model --num-classes 6 --formats torchscript,quantized
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `saved_models/cryingsense_cnn_best.pth` | Source checkpoint |
| `--output-dir` | `saved_models/exported` | Output directory |
| `--num-classes` | 6 | Number of output classes |
| `--input-shape` | `1,4,128,216` | Input tensor shape |
| `--formats` | `torchscript,onnx,quantized` | Export formats |
| `--version` | auto | Version number (auto-detect if not set) |
| `--base-name` | `cryingsense_model_beta` | Base name prefix |

### Loading Exported Models

**TorchScript:**
```python
import torch
model = torch.jit.load('cryingsense_model_beta_v1.torchscript.pt')
output = model(input_tensor)
```

**PyTorch State Dict:**
```python
import torch
from model.models.cnn_model import CryingSenseCNN

checkpoint = torch.load('cryingsense_model_beta_v1.pth')
model = CryingSenseCNN(num_classes=checkpoint['metadata']['num_classes'])

# Initialize lazy layers
dummy = torch.randn(1, 4, 128, 216)
model(dummy)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

**Quantized:**
```python
import torch
model = CryingSenseCNN(num_classes=6)
model.load_state_dict(torch.load('cryingsense_model_beta_v1_quantized.pth'))
```

## Current Exports

| Version | Description |
|---------|-------------|
| v1 | Initial beta export from best trained model |

## Model Info

- **Parameters:** ~12,410
- **Size:** ~0.05 MB (fp32)
- **Input:** 4-channel features (MFCC, Mel, Chroma, Delta MFCC)
- **Output:** 6 classes (belly_pain, burp, discomfort, hunger, noise, tired)
