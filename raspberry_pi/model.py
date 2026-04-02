"""
ONNX Model Runtime for Raspberry Pi pipeline.

This module is the single model entry point for the Pi runtime and only supports
ONNX inference via onnxruntime.
"""

import os
import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class CryingSenseModel:
    """ONNX-only model wrapper exposing a stable ``predict_raw`` API."""

    def __init__(self, model_path: str, num_threads: int = 4):
        self.model_path = model_path
        self.num_threads = num_threads
        self._session: Optional[object] = None
        self._input_name: str = ""

    @property
    def backend(self) -> str:
        return "onnx"

    def load(self):
        if not self.model_path or not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"ONNX model file not found: {self.model_path}. "
                "Place the model in raspberry_pi/saved_models/ and set MODEL_PATH if needed."
            )

        if os.path.splitext(self.model_path)[1].lower() != ".onnx":
            raise ValueError("Only ONNX models are supported in the Raspberry Pi pipeline.")

        import onnxruntime as ort

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = self.num_threads
        session_options.inter_op_num_threads = 1
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            self.model_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

        log.info("ONNX model loaded  [path=%s]", self.model_path)
        return self

    def predict_raw(self, features: np.ndarray) -> np.ndarray:
        """Run inference and return logits with shape ``(num_classes,)``."""
        if self._session is None:
            raise RuntimeError("Model not loaded. Call load() before predict_raw().")
        logits = self._session.run(None, {self._input_name: features})[0]
        return logits[0]
