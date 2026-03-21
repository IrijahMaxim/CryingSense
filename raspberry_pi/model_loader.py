"""
Model Loader — Raspberry Pi 3B+ Pipeline

Loads the quantized CryingSenseCNN in order of preference:
  1. ONNX   — onnxruntime  (fastest on ARM, ~60 MB runtime)
  2. TorchScript — torch.jit  (no model def needed)
  3. Quantized .pth — torch dynamic-quantize (smallest file)

All inference is CPU-only (no CUDA on Pi).
"""

import os
import logging
import numpy as np

log = logging.getLogger(__name__)


class ModelLoader:
    """Auto-detects model format and exposes a unified ``predict(features)`` interface."""

    def __init__(self, model_path: str, num_classes: int = 6):
        self.model_path = model_path
        self.num_classes = num_classes
        self._backend = None  # "onnx" | "torchscript" | "pytorch"
        self._model = None
        self._session = None  # onnxruntime session

    # ── public ───────────────────────────────────────────────────────────

    def load(self):
        """Load the model and return *self* for chaining."""
        if not self.model_path or not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                "Place a model in raspberry_pi/saved_models/ "
                "(see saved_models/README.md)."
            )

        ext = os.path.splitext(self.model_path)[1].lower()

        if ext == ".onnx":
            self._load_onnx()
        elif ext == ".pt":
            self._load_torchscript()
        elif ext == ".pth":
            self._load_quantized_pytorch()
        else:
            raise ValueError(f"Unsupported model format: {ext}")

        log.info("Model loaded  [backend=%s  path=%s]", self._backend, self.model_path)
        return self

    def predict_raw(self, features: np.ndarray) -> np.ndarray:
        """
        Run forward pass.

        Parameters
        ----------
        features : (1, 4, H, W) float32 numpy array.

        Returns
        -------
        (num_classes,) float32 logits.
        """
        if self._backend == "onnx":
            return self._infer_onnx(features)
        else:
            return self._infer_torch(features)

    @property
    def backend(self) -> str:
        return self._backend

    # ── ONNX ─────────────────────────────────────────────────────────────

    def _load_onnx(self):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4          # use all 4 Cortex-A53 cores
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            self.model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._backend = "onnx"

    def _infer_onnx(self, features: np.ndarray) -> np.ndarray:
        input_name = self._session.get_inputs()[0].name
        logits = self._session.run(None, {input_name: features})[0]
        return logits[0]

    # ── TorchScript ──────────────────────────────────────────────────────

    def _load_torchscript(self):
        import torch

        self._model = torch.jit.load(self.model_path, map_location="cpu")
        self._model.eval()
        self._backend = "torchscript"

    # ── Quantized PyTorch (.pth) ─────────────────────────────────────────

    def _load_quantized_pytorch(self):
        import torch
        import torch.nn as nn

        # Inline lightweight model definition to avoid importing from main project
        class _DepthwiseSeparableConv(nn.Module):
            def __init__(self, in_ch, out_ch, ks=3, pad=1):
                super().__init__()
                self.depthwise = nn.Conv2d(in_ch, in_ch, ks, padding=pad, groups=in_ch, bias=False)
                self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
                self.bn = nn.BatchNorm2d(out_ch)

            def forward(self, x):
                return torch.relu(self.bn(self.pointwise(self.depthwise(x))))

        class _CryingSenseCNN(nn.Module):
            def __init__(self, num_classes=6, in_channels=4, dropout_rate=0.3):
                super().__init__()
                self.conv1 = _DepthwiseSeparableConv(in_channels, 16)
                self.conv2 = _DepthwiseSeparableConv(16, 32)
                self.conv3 = _DepthwiseSeparableConv(32, 64)
                self.pool = nn.MaxPool2d(2, 2)
                self.dropout = nn.Dropout(dropout_rate)
                self._fc1 = None
                self.fc2 = nn.Linear(128, num_classes)

            def forward(self, x):
                x = self.pool(self.conv1(x))
                x = self.pool(self.conv2(x))
                x = self.pool(self.conv3(x))
                x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
                x = x.view(x.size(0), -1)
                if self._fc1 is None:
                    self._fc1 = nn.Linear(x.shape[1], 128).to(x.device)
                x = self.dropout(torch.relu(self._fc1(x)))
                return self.fc2(x)

        model = _CryingSenseCNN(num_classes=self.num_classes)
        # Lazy-init fc1
        with torch.no_grad():
            model(torch.randn(1, 4, 128, 157))

        # Dynamic quantisation — shrinks Linear layers to int8
        model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )

        ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        model.eval()

        self._model = model
        self._backend = "pytorch"

    # ── shared torch inference ───────────────────────────────────────────

    def _infer_torch(self, features: np.ndarray) -> np.ndarray:
        import torch

        tensor = torch.from_numpy(features).float()
        with torch.no_grad():
            logits = self._model(tensor)
        return logits.cpu().numpy()[0]
