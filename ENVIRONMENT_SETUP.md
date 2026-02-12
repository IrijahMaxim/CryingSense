# PyTorch Environment Setup

## Environment Created
A Python virtual environment has been successfully set up for the CryingSense project with all required dependencies.

## Installed Packages
- **PyTorch**: 2.10.0 (CPU version)
- **TorchAudio**: 2.10.0
- **NumPy**: 2.3.5
- **Librosa**: 0.11.0
- **Scikit-learn**: 1.8.0
- **Matplotlib**: 3.10.8
- **Seaborn**: 0.13.2
- **ONNX & ONNX Runtime**: For model export
- **PyAudio**: For audio capture
- And all other dependencies from `model/requirements.txt`

## How to Activate the Environment

### Windows PowerShell
```powershell
cd "p:\VScode Lobby\CryingSense"
.\venv\Scripts\Activate.ps1
```

### Windows Command Prompt
```cmd
cd "p:\VScode Lobby\CryingSense"
venv\Scripts\activate.bat
```

## Quick Verification
After activation, verify the installation:
```python
python -c "import torch; print('PyTorch:', torch.__version__)"
```

## Training the Model
Once the environment is activated, you can train the model:
```bash
python model/training/train.py
```

## GPU Support
The current installation uses PyTorch CPU version. If you have an NVIDIA GPU and want to use CUDA acceleration:

1. Check your CUDA version: https://developer.nvidia.com/cuda-gpus
2. Uninstall current PyTorch:
   ```bash
   pip uninstall torch torchaudio
   ```
3. Install PyTorch with CUDA support:
   ```bash
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   (Replace `cu118` with your CUDA version, e.g., `cu121` for CUDA 12.1)

## Deactivating the Environment
```bash
deactivate
```

## Notes
- The `wave` module (Python standard library) is available without installation
- All audio processing and ML dependencies are ready for training
- The environment is isolated from your system Python installation
