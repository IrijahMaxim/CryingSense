"""
Audio Preprocessing — Raspberry Pi 3B+ Pipeline

Normalises incoming .wav audio before feature extraction:
  1. Load / resample to 16 kHz mono
  2. Optional noise reduction (spectral gating)
  3. Silence trimming
  4. Peak normalisation
  5. Pad / crop to exactly DURATION seconds
"""

import numpy as np
import librosa

from config import (
    SAMPLE_RATE,
    DURATION,
    TOP_DB,
    NOISE_REDUCE,
)


class AudioPreprocessor:
    """Lightweight audio preprocessor sized for 1 GB RAM."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        duration: float = DURATION,
        top_db: int = TOP_DB,
        noise_reduce: bool = NOISE_REDUCE,
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.top_db = top_db
        self.noise_reduce = noise_reduce
        self.target_length = int(sample_rate * duration)

    # ── public API ───────────────────────────────────────────────────────

    def preprocess_file(self, wav_path: str) -> np.ndarray:
        """Full pipeline from file path → normalised numpy array."""
        audio = self._load(wav_path)
        return self._pipeline(audio)

    def preprocess_array(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Full pipeline from raw numpy array (any sample rate)."""
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        return self._pipeline(audio)

    # ── internals ────────────────────────────────────────────────────────

    def _load(self, path: str) -> np.ndarray:
        audio, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        return audio

    def _pipeline(self, audio: np.ndarray) -> np.ndarray:
        if self.noise_reduce:
            audio = self._reduce_noise(audio)
        audio = self._trim_silence(audio)
        audio = self._normalise(audio)
        audio = self._pad_or_crop(audio)
        return audio

    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        try:
            import noisereduce as nr
            return nr.reduce_noise(
                y=audio,
                sr=self.sample_rate,
                stationary=True,
                prop_decrease=0.8,
            )
        except ImportError:
            return audio

    def _trim_silence(self, audio: np.ndarray) -> np.ndarray:
        trimmed, _ = librosa.effects.trim(audio, top_db=self.top_db)
        if len(trimmed) < self.sample_rate * 0.5:
            return audio
        return trimmed

    @staticmethod
    def _normalise(audio: np.ndarray) -> np.ndarray:
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.9
        return audio

    def _pad_or_crop(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) > self.target_length:
            start = (len(audio) - self.target_length) // 2
            audio = audio[start : start + self.target_length]
        elif len(audio) < self.target_length:
            audio = np.pad(audio, (0, self.target_length - len(audio)))
        return audio
