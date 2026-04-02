"""Compatibility shim: ModelLoader now maps to the ONNX-only model runtime."""

try:
    from .model import CryingSenseModel
except ImportError:
    from model import CryingSenseModel


class ModelLoader(CryingSenseModel):
    """Backward-compatible alias for older imports."""
