"""
Predictor — Raspberry Pi 3B+ Pipeline

Takes a preprocessed + feature-extracted input and produces a classification dict:
  - predicted class (excluding invisible classes like "noise")
  - confidence score
  - all class probabilities
  - is_cry flag
  - alert flag (confidence ≥ threshold)
"""

import time
import logging
from datetime import datetime

import numpy as np

from config import (
    CLASS_NAMES,
    CRY_CLASSES,
    INVISIBLE_CLASSES,
    CONFIDENCE_THRESHOLD,
)

log = logging.getLogger(__name__)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class Predictor:
    """Wraps model loader + result formatting."""

    def __init__(self, model_loader, confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self.model = model_loader
        self.threshold = confidence_threshold

    def predict(self, features: np.ndarray, audio_file: str = "") -> dict:
        """
        Parameters
        ----------
        features : (4, H, W)  float32 numpy array from FeatureExtractor.
        audio_file : optional filename (for logging).

        Returns
        -------
        dict with keys:
            is_cry, prediction, confidence, probabilities,
            cry_probabilities, alert, inference_time_ms, timestamp
        """
        # Add batch dim → (1, 4, H, W)
        batch = features[np.newaxis, ...]

        t0 = time.perf_counter()
        logits = self.model.predict_raw(batch)
        inference_ms = (time.perf_counter() - t0) * 1000

        probs = _softmax(logits)
        idx = int(np.argmax(probs))
        predicted_class = CLASS_NAMES[idx]
        confidence = float(probs[idx])

        is_cry = predicted_class not in INVISIBLE_CLASSES

        result = {
            "is_cry": is_cry,
            "prediction": predicted_class if is_cry else "no_cry_detected",
            "confidence": round(confidence, 4),
            "probabilities": {
                name: round(float(p), 4)
                for name, p in zip(CLASS_NAMES, probs)
            },
            "cry_probabilities": {
                name: round(float(p), 4)
                for name, p in zip(CLASS_NAMES, probs)
                if name not in INVISIBLE_CLASSES
            },
            "alert": is_cry and confidence >= self.threshold,
            "inference_time_ms": round(inference_ms, 2),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "audio_file": audio_file,
        }

        if is_cry:
            log.info(
                "CRY  %-12s  conf=%.2f  inf=%5.1f ms",
                predicted_class, confidence, inference_ms,
            )
        else:
            log.debug("NOISE  conf=%.2f  inf=%5.1f ms", confidence, inference_ms)

        return result
