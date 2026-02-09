"""
Model Export Utilities for CryingSense

Exports trained CNN models to various formats:
- PyTorch (.pt, .pth)
- TorchScript (optimized for deployment)
- ONNX (cross-platform inference)
"""

import os
import sys
import torch
import torch.onnx

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from model.models.cnn_model import CryingSenseCNN


def export_to_torchscript(model, input_shape, save_path, optimize=True):
    """
    Export model to TorchScript format for optimized deployment.
    
    Args:
        model: Trained PyTorch model
        input_shape: Input tensor shape (e.g., (1, 4, 128, 216))
        save_path: Path to save TorchScript model
        optimize: Whether to optimize for inference
    """
    model.eval()
    
    # Create dummy input for tracing
    dummy_input = torch.randn(input_shape)
    
    # Trace the model
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Optimize if requested
    if optimize:
        traced_model = torch.jit.optimize_for_inference(traced_model)
    
    # Save
    torch.jit.save(traced_model, save_path)
    print(f"TorchScript model saved to: {save_path}")
    
    # Verify the model can be loaded
    loaded_model = torch.jit.load(save_path)
    test_output = loaded_model(dummy_input)
    print(f"Verification successful. Output shape: {test_output.shape}")
    
    return traced_model


def export_to_onnx(model, input_shape, save_path, opset_version=14):
    """
    Export model to ONNX format for cross-platform deployment.
    
    Args:
        model: Trained PyTorch model
        input_shape: Input tensor shape (e.g., (1, 4, 128, 216))
        save_path: Path to save ONNX model
        opset_version: ONNX opset version (default: 14)
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Input and output names
    input_names = ['input']
    output_names = ['output']
    
    # Dynamic axes for flexible batch size
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    
    print(f"ONNX model saved to: {save_path}")
    
    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification successful!")
    except ImportError:
        print("Warning: onnx package not found. Skipping verification.")
    except Exception as e:
        print(f"ONNX verification failed: {e}")


def export_state_dict(model, save_path, include_metadata=True, metadata=None):
    """
    Export model state dict with optional metadata.
    
    Args:
        model: Trained PyTorch model
        save_path: Path to save model
        include_metadata: Whether to include training metadata
        metadata: Additional metadata to include
    """
    if include_metadata and metadata:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            **metadata
        }
        torch.save(checkpoint, save_path)
    else:
        torch.save(model.state_dict(), save_path)
    
    print(f"Model state dict saved to: {save_path}")


def quantize_model(model, example_input):
    """
    Quantize model for reduced size and faster inference.
    
    Args:
        model: Trained PyTorch model
        example_input: Example input tensor for calibration
    
    Returns:
        Quantized model
    """
    model.eval()
    
    # Dynamic quantization (good for CPU inference)
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    
    print("Model quantized successfully!")
    
    # Test quantized model
    with torch.no_grad():
        original_output = model(example_input)
        quantized_output = quantized_model(example_input)
        
    print(f"Original output shape: {original_output.shape}")
    print(f"Quantized output shape: {quantized_output.shape}")
    
    return quantized_model


def main():
    """Main export function."""
    import argparse
    
    # Get script directory for resolving relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model = os.path.join(script_dir, '../saved_models/cryingsense_cnn.pth')
    default_output = os.path.join(script_dir, '../saved_models/exported')
    
    parser = argparse.ArgumentParser(description='Export CryingSense CNN model')
    parser.add_argument('--model', type=str, 
                       default=default_model,
                       help='Path to trained model checkpoint (default: ../saved_models/cryingsense_cnn.pth)')
    parser.add_argument('--output-dir', type=str, default=default_output,
                       help='Output directory for exported models')
    parser.add_argument('--num-classes', type=int, default=5,
                       help='Number of classes')
    parser.add_argument('--input-shape', type=str, default='1,4,128,216',
                       help='Input shape as comma-separated values (batch,channels,height,width)')
    parser.add_argument('--formats', type=str, default='torchscript,onnx,quantized',
                       help='Export formats: torchscript, onnx, quantized (comma-separated)')
    
    args = parser.parse_args()
    
    # Resolve model path
    if not os.path.isabs(args.model):
        args.model = os.path.abspath(args.model)
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("\nPlease train a model first or specify the correct path with --model")
        print("Example: python export_model.py --model path/to/your/model.pth")
        return
    
    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("CryingSense Model Export")
    print("="*70)
    print(f"Input model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Input shape: {input_shape}")
    print(f"Export formats: {args.formats}")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    device = torch.device('cpu')  # Export on CPU for compatibility
    model = CryingSenseCNN(num_classes=args.num_classes).to(device)
    
    checkpoint = torch.load(args.model, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Validation accuracy: {checkpoint.get('val_acc', 'unknown'):.4f}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / 1024 / 1024  # fp32
    print(f"Model parameters: {total_params:,}")
    print(f"Estimated size: ~{model_size_mb:.2f} MB")
    
    formats = [f.strip() for f in args.formats.split(',')]
    
    # Export to requested formats
    print("\n" + "="*70)
    print("Exporting Models")
    print("="*70)
    
    if 'torchscript' in formats:
        print("\n[1/3] Exporting to TorchScript...")
        torchscript_path = os.path.join(args.output_dir, 'cryingsense_cnn.torchscript.pt')
        export_to_torchscript(model, input_shape, torchscript_path, optimize=True)
    
    if 'onnx' in formats:
        print("\n[2/3] Exporting to ONNX...")
        onnx_path = os.path.join(args.output_dir, 'cryingsense_cnn.onnx')
        export_to_onnx(model, input_shape, onnx_path)
    
    if 'quantized' in formats:
        print("\n[3/3] Creating quantized model...")
        example_input = torch.randn(input_shape)
        quantized_model = quantize_model(model, example_input)
        quantized_path = os.path.join(args.output_dir, 'cryingsense_cnn_quantized.pth')
        torch.save(quantized_model.state_dict(), quantized_path)
        print(f"Quantized model saved to: {quantized_path}")
    
    print("\n" + "="*70)
    print("Export Complete!")
    print("="*70)
    print(f"\nExported models saved to: {args.output_dir}")
    print("\nUsage:")
    print("  TorchScript: model = torch.jit.load('cryingsense_cnn.torchscript.pt')")
    print("  ONNX: Use onnxruntime or other ONNX-compatible frameworks")
    print("  Quantized: Load like regular PyTorch model with reduced size")
    print("="*70)


if __name__ == "__main__":
    main()
