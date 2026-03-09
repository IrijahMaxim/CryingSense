"""
Quick test script to verify trail_run system components.

Tests:
- Configuration loading
- Audio buffer creation
- Model file existence
- Import chain
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        import config
        print("  ✓ config")
        from audio_buffer import AudioBuffer
        print("  ✓ audio_buffer")
        from wifi_receiver import WiFiAudioReceiver
        print("  ✓ wifi_receiver")
        from classifier import CryClassifier
        print("  ✓ classifier")
        from database_handler import DatabaseHandler
        print("  ✓ database_handler")
        from waveform_display import TerminalDisplay
        print("  ✓ waveform_display")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config():
    """Test configuration values."""
    print("\nTesting configuration...")
    try:
        import config
        print(f"  Model path: {config.MODEL_PATH}")
        print(f"  Model exists: {config.MODEL_PATH.exists()}")
        print(f"  Classes: {config.CLASS_NAMES}")
        print(f"  Ignore classes: {config.IGNORE_CLASSES}")
        print(f"  Sample rate: {config.SAMPLE_RATE}")
        print(f"  WiFi port: {config.WIFI_PORT}")
        return True
    except Exception as e:
        print(f"  ✗ Config test failed: {e}")
        return False

def test_audio_buffer():
    """Test audio buffer creation."""
    print("\nTesting audio buffer...")
    try:
        from audio_buffer import AudioBuffer
        buffer = AudioBuffer(max_duration=5.0, sample_rate=16000)
        print(f"  ✓ Buffer created: {buffer.max_samples} samples")
        
        # Test write/read
        import numpy as np
        test_data = np.random.randint(-1000, 1000, 1000, dtype=np.int16)
        buffer.write(test_data)
        print(f"  ✓ Write test: {buffer.samples_available} samples")
        
        read_data = buffer.read_latest(500)
        print(f"  ✓ Read test: {len(read_data)} samples")
        
        return True
    except Exception as e:
        print(f"  ✗ Buffer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """Test that the model can be loaded."""
    print("\nTesting model loading...")
    try:
        import config
        from model.models.cnn_model import CryingSenseCNN
        import torch
        
        if not config.MODEL_PATH.exists():
            print(f"  ⚠ Model file not found: {config.MODEL_PATH}")
            print("  Run training first: python model/training/train.py")
            return False
        
        model = CryingSenseCNN(
            num_classes=config.NUM_CLASSES,
            in_channels=4,
            dropout_rate=0.3,
            use_gap=True
        )
        
        checkpoint = torch.load(str(config.MODEL_PATH), map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Classes: {config.NUM_CLASSES}")
        
        return True
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("CryingSense Trail Run - System Test")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Audio Buffer", test_audio_buffer()))
    results.append(("Model Loading", test_model_loading()))
    
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {name}")
    
    all_passed = all(r[1] for r in results)
    
    print("=" * 60)
    if all_passed:
        print("\n✓ All tests passed! System ready to run.")
        print("  Start the system: python main.py")
    else:
        print("\n✗ Some tests failed. Please fix issues before running.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
