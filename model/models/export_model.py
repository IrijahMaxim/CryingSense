"""
Model Export Utilities for CryingSense

Exports trained CNN models to various formats:
- PyTorch (.pt, .pth)
- TorchScript (optimized for deployment)
- ONNX (cross-platform inference)

Versioning:
- Auto-incrementing version: cryingsense_model_beta_v1, v2, etc.
- Scans output directory to find next available version
"""

import os
import sys
import re
import glob
import torch
import torch.onnx

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from model.models.cnn_model import CryingSenseCNN


def get_next_version(output_dir: str, base_name: str = "cryingsense_model_beta") -> int:
    """
    Scan output directory to find the next available version number.
    
    Args:
        output_dir: Directory to scan for existing exports
        base_name: Base name prefix for versioned models
        
    Returns:
        Next available version number (starts at 1)
    """
    if not os.path.exists(output_dir):
        return 1
    
    # Pattern to match versioned files: cryingsense_model_beta_v{number}*
    pattern = os.path.join(output_dir, f"{base_name}_v*")
    existing_files = glob.glob(pattern)
    
    if not existing_files:
        return 1
    
    # Extract version numbers from existing files
    version_pattern = re.compile(rf"{re.escape(base_name)}_v(\d+)")
    versions = []
    
    for filepath in existing_files:
        filename = os.path.basename(filepath)
        match = version_pattern.search(filename)
        if match:
            versions.append(int(match.group(1)))
    
    if not versions:
        return 1
    
    return max(versions) + 1


def get_versioned_name(version: int, base_name: str = "cryingsense_model_beta") -> str:
    """
    Generate versioned model name.
    
    Args:
        version: Version number
        base_name: Base name prefix
        
    Returns:
        Versioned name like 'cryingsense_model_beta_v1'
    """
    return f"{base_name}_v{version}"


def export_to_torchscript(model, input_shape, save_path, optimize=True):
    """
    Export model to TorchScript format for optimized deployment.
    
    Args:
        model: Trained PyTorch model
        input_shape: Input tensor shape (e.g., (1, 4, 128, 216))
        save_path: Path to save TorchScript model
        optimize: Whether to optimize for inference (may not work with all models)
    """
    model.eval()
    
    # Create dummy input for tracing
    dummy_input = torch.randn(input_shape)
    
    # Trace the model
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Optimize if requested (skip if it fails)
    if optimize:
        try:
            traced_model = torch.jit.optimize_for_inference(traced_model)
        except Exception as e:
            print(f"Note: Optimization skipped ({e})")
    
    # Save
    torch.jit.save(traced_model, save_path)
    print(f"TorchScript model saved to: {save_path}")
    
    # Verify the model can be loaded
    try:
        loaded_model = torch.jit.load(save_path)
        test_output = loaded_model(dummy_input)
        print(f"Verification successful. Output shape: {test_output.shape}")
    except Exception as e:
        print(f"Note: Verification skipped ({e})")
    
    return traced_model


def export_to_onnx(model, input_shape, save_path, opset_version=14):
    """
    Export model to ONNX format for cross-platform deployment.
    
    Args:
        model: Trained PyTorch model
        input_shape: Input tensor shape (e.g., (1, 4, 128, 216))
        save_path: Path to save ONNX model
        opset_version: ONNX opset version (default: 14)
        
    Returns:
        True if export succeeded, False otherwise
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
    try:
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
    except Exception as e:
        print(f"ONNX export failed: {e}")
        print("To enable ONNX export, install: pip install onnx onnxscript")
        return False
    
    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification successful!")
    except ImportError:
        print("Note: onnx package not found. Skipping verification.")
    except Exception as e:
        print(f"ONNX verification failed: {e}")
    
    return True


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
    default_model = os.path.join(script_dir, '../saved_models/cryingsense_cnn_best.pth')
    default_output = os.path.join(script_dir, '../saved_models/exported')
    
    parser = argparse.ArgumentParser(description='Export CryingSense CNN model')
    parser.add_argument('--model', type=str, 
                       default=default_model,
                       help='Path to trained model checkpoint (default: ../saved_models/cryingsense_cnn_best.pth)')
    parser.add_argument('--output-dir', type=str, default=default_output,
                       help='Output directory for exported models')
    parser.add_argument('--num-classes', type=int, default=6,
                       help='Number of classes (default: 6)')
    parser.add_argument('--input-shape', type=str, default='1,4,128,216',
                       help='Input shape as comma-separated values (batch,channels,height,width)')
    parser.add_argument('--formats', type=str, default='torchscript,onnx,quantized',
                       help='Export formats: torchscript, onnx, quantized (comma-separated)')
    parser.add_argument('--version', type=int, default=None,
                       help='Version number (default: auto-detect next version)')
    parser.add_argument('--base-name', type=str, default='cryingsense_model_beta',
                       help='Base name for versioned exports (default: cryingsense_model_beta)')
    
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
    
    # Determine version number
    if args.version is None:
        version = get_next_version(args.output_dir, args.base_name)
    else:
        version = args.version
    
    versioned_name = get_versioned_name(version, args.base_name)
    
    print("="*70)
    print("CryingSense Model Export")
    print("="*70)
    print(f"Input model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Version: {versioned_name}")
    print(f"Input shape: {input_shape}")
    print(f"Export formats: {args.formats}")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    device = torch.device('cpu')  # Export on CPU for compatibility
    model = CryingSenseCNN(num_classes=args.num_classes).to(device)
    
    # Initialize _fc1 layer by running a dummy forward pass
    # This is required because _fc1 is lazily initialized in the model
    dummy_input = torch.randn(input_shape).to(device)
    with torch.no_grad():
        _ = model(dummy_input)
    
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
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
    
    exported_files = []
    
    if 'torchscript' in formats:
        print("\n[1/3] Exporting to TorchScript...")
        torchscript_path = os.path.join(args.output_dir, f'{versioned_name}.torchscript.pt')
        export_to_torchscript(model, input_shape, torchscript_path, optimize=True)
        exported_files.append(torchscript_path)
    
    if 'onnx' in formats:
        print("\n[2/3] Exporting to ONNX...")
        onnx_path = os.path.join(args.output_dir, f'{versioned_name}.onnx')
        if export_to_onnx(model, input_shape, onnx_path):
            exported_files.append(onnx_path)
    
    if 'quantized' in formats:
        print("\n[3/3] Creating quantized model...")
        example_input = torch.randn(input_shape)
        quantized_model = quantize_model(model, example_input)
        quantized_path = os.path.join(args.output_dir, f'{versioned_name}_quantized.pth')
        torch.save(quantized_model.state_dict(), quantized_path)
        print(f"Quantized model saved to: {quantized_path}")
        exported_files.append(quantized_path)
    
    # Also save a copy of the original state dict with metadata
    metadata_path = os.path.join(args.output_dir, f'{versioned_name}.pth')
    export_state_dict(model, metadata_path, {
        'version': version,
        'base_name': args.base_name,
        'num_classes': args.num_classes,
        'input_shape': input_shape,
        'source_model': args.model
    })
    exported_files.append(metadata_path)
    
    print("\n" + "="*70)
    print("Export Complete!")
    print("="*70)
    print(f"\nVersion: {versioned_name}")
    print(f"Exported models saved to: {args.output_dir}")
    print("\nExported files:")
    for f in exported_files:
        print(f"  - {os.path.basename(f)}")
    print("\nUsage:")
    print(f"  TorchScript: model = torch.jit.load('{versioned_name}.torchscript.pt')")
    print(f"  ONNX: Use onnxruntime or other ONNX-compatible frameworks")
    print(f"  Quantized: Load like regular PyTorch model with reduced size")
    print("="*70)


if __name__ == "__main__":
    main()
