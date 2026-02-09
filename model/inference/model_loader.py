"""
Model Loading Utilities for CryingSense

Provides utilities to load trained models in various formats:
- Standard PyTorch models (.pth)
- Quantized models for edge devices
- TorchScript models for deployment
"""

import os
import sys
import torch
import torch.nn as nn

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from model.models.cnn_model import CryingSenseCNN


class ModelLoader:
    """Utility class for loading CryingSense models."""
    
    def __init__(self, model_path, num_classes=5, device=None):
        """
        Initialize model loader.
        
        Args:
            model_path: Path to model file
            num_classes: Number of output classes (default: 5)
            device: Device to load model on (cpu/cuda)
        """
        self.model_path = model_path
        self.num_classes = num_classes
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_type = self._detect_model_type()
        
    def _detect_model_type(self):
        """Detect model type from file extension."""
        ext = os.path.splitext(self.model_path)[1].lower()
        
        if ext == '.pt' and 'torchscript' in self.model_path.lower():
            return 'torchscript'
        elif ext == '.onnx':
            return 'onnx'
        elif ext in ['.pth', '.pt']:
            if 'quantized' in self.model_path.lower():
                return 'quantized'
            return 'standard'
        else:
            raise ValueError(f"Unsupported model format: {ext}")
    
    def load(self):
        """
        Load model based on detected type.
        
        Returns:
            Loaded model ready for inference
        """
        if self.model_type == 'torchscript':
            return self._load_torchscript()
        elif self.model_type == 'onnx':
            return self._load_onnx()
        elif self.model_type == 'quantized':
            return self._load_quantized()
        else:
            return self._load_standard()
    
    def _load_standard(self):
        """Load standard PyTorch model."""
        print(f"Loading standard PyTorch model from {self.model_path}")
        
        model = CryingSenseCNN(num_classes=self.num_classes).to(self.device)
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model from epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"Validation accuracy: {checkpoint.get('val_acc', 'unknown'):.4f}")
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        self.model = model
        
        print(f"Model loaded on device: {self.device}")
        return model
    
    def _load_quantized(self):
        """Load quantized PyTorch model."""
        print(f"Loading quantized PyTorch model from {self.model_path}")
        
        model = CryingSenseCNN(num_classes=self.num_classes)
        
        # Apply dynamic quantization
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        self.model = model
        
        print("Quantized model loaded on CPU")
        return model
    
    def _load_torchscript(self):
        """Load TorchScript model."""
        print(f"Loading TorchScript model from {self.model_path}")
        
        model = torch.jit.load(self.model_path, map_location=self.device)
        model.eval()
        self.model = model
        
        print(f"TorchScript model loaded on device: {self.device}")
        return model
    
    def _load_onnx(self):
        """Load ONNX model."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required for ONNX model inference. "
                            "Install with: pip install onnxruntime")
        
        print(f"Loading ONNX model from {self.model_path}")
        
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
        
        session = ort.InferenceSession(self.model_path, providers=providers)
        self.model = session
        
        print(f"ONNX model loaded with providers: {session.get_providers()}")
        return session
    
    def get_model_info(self):
        """Get information about the loaded model."""
        info = {
            'model_path': self.model_path,
            'model_type': self.model_type,
            'device': str(self.device),
            'num_classes': self.num_classes
        }
        
        if self.model and self.model_type in ['standard', 'quantized']:
            total_params = sum(p.numel() for p in self.model.parameters())
            info['total_parameters'] = total_params
            info['model_size_mb'] = total_params * 4 / 1024 / 1024
        
        return info


def load_model(model_path, num_classes=5, device=None):
    """
    Convenience function to load a model.
    
    Args:
        model_path: Path to model file
        num_classes: Number of output classes
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    loader = ModelLoader(model_path, num_classes, device)
    return loader.load()


if __name__ == "__main__":
    # Test model loading
    import argparse
    
    parser = argparse.ArgumentParser(description='Test model loading')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--num-classes', type=int, default=5, help='Number of classes')
    
    args = parser.parse_args()
    
    print("="*70)
    print("CryingSense Model Loader Test")
    print("="*70)
    
    loader = ModelLoader(args.model, args.num_classes)
    model = loader.load()
    
    print("\nModel Information:")
    info = loader.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("Model loaded successfully!")
    print("="*70)
