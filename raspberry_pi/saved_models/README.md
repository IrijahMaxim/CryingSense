# Place your exported model files here
#
# Supported formats (in order of preference for RPi 3B+):
#   1. ONNX          — cryingsense_model.onnx   (fastest on ARM via onnxruntime)
#   2. TorchScript   — cryingsense_model.pt      (portable, no Python model def needed)
#   3. Quantized PTH — cryingsense_quantized.pth (smallest file, needs torch)
#
# Generate these from the main project:
#   python model/models/export_model.py --formats onnx,torchscript,quantized
