import torch
import torch.nn as nn
import numpy as np
import os


def count_parameters(model):
    """Count the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, filepath, epoch=None, optimizer=None, metrics=None):
    """
    Save model checkpoint with optional training state.
    
    Args:
        model: PyTorch model to save
        filepath: Path to save the model
        epoch: Current epoch number (optional)
        optimizer: Optimizer state (optional)
        metrics: Dictionary of metrics (optional)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
    }
    
    if epoch is not None:
        save_dict['epoch'] = epoch
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    if metrics is not None:
        save_dict['metrics'] = metrics
    
    torch.save(save_dict, filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath, device='cpu', optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model to load weights into
        filepath: Path to the saved model
        device: Device to load the model on
        optimizer: Optimizer to load state into (optional)
    
    Returns:
        Dictionary containing epoch and metrics if available
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return {
            'epoch': checkpoint.get('epoch'),
            'metrics': checkpoint.get('metrics')
        }
    else:
        # Old format - just state dict
        model.load_state_dict(checkpoint)
        return {}


def quantize_model(model, dtype=torch.qint8):
    """
    Quantize a trained model for deployment on edge devices.
    
    Args:
        model: Trained PyTorch model
        dtype: Quantization data type (default: torch.qint8)
    
    Returns:
        Quantized model
    """
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv2d}, dtype=dtype
    )
    return quantized_model


def get_model_size_mb(model):
    """Calculate the size of a model in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def inference_time(model, input_shape, device='cpu', num_runs=100):
    """
    Measure average inference time of a model.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (batch_size, channels, height, width)
        device: Device to run inference on
        num_runs: Number of inference runs to average
    
    Returns:
        Average inference time in milliseconds
    """
    import time
    
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    avg_time = (time.time() - start_time) / num_runs * 1000  # Convert to ms
    return avg_time


def print_model_summary(model, input_shape=(1, 4, 128, 216)):
    """Print a summary of the model architecture and statistics."""
    print("=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    print(f"Total Parameters: {count_parameters(model):,}")
    print(f"Model Size: {get_model_size_mb(model):.2f} MB")
    print("=" * 80)
    print("\nModel Architecture:")
    print(model)
    print("=" * 80)


if __name__ == "__main__":
    # Test utilities
    from cnn_model import CryingSenseCNN
    
    model = CryingSenseCNN(num_classes=5)
    print_model_summary(model)
    
    # Test inference time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    avg_time = inference_time(model, (1, 4, 128, 216), device=device)
    print(f"\nAverage Inference Time: {avg_time:.2f} ms")
